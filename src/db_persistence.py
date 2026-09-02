"""
src/db_persistence.py — PostgreSQL persistence operations.

Handles all database writes for the inference pipeline:
  - payment event persistence (with idempotency)
  - recovery decision persistence
  - audit record persistence
  - simulated outcome persistence

All operations fail safely. If the database is unavailable,
the inference pipeline continues without persistence.
Credentials are never logged.
"""

import json
import datetime
import uuid
import logging
import os
from typing import Dict, Any, Optional, Tuple

from src.database import get_connection, DatabaseUnavailableError

logger = logging.getLogger(__name__)

# Idempotency key resolution

def resolve_idempotency_key(event: Dict[str, Any]) -> Tuple[str, str]:
    """
    Resolve the idempotency_key for an incoming event.

    Priority:
      1. Caller-supplied 'idempotency_key'           → source = 'caller_supplied'
      2. Caller-supplied 'event_id'                  → source = 'event_id_fallback'
      3. Request-scoped UUID (generated)             → source = 'request_scoped_fallback'
         WARNING: fallback does NOT prevent duplicate processing across retries.

    Returns: (idempotency_key, source_label)
    """
    if event.get('idempotency_key'):
        return str(event['idempotency_key']), 'caller_supplied'
    if event.get('event_id'):
        return str(event['event_id']), 'event_id_fallback'
    return str(uuid.uuid4()), 'request_scoped_fallback'


# Payment event persistence

def persist_payment_event(
    event: Dict[str, Any],
    payment_id: str,
    idempotency_key: str,
) -> Tuple[Optional[str], bool]:
    """
    Persist a payment event. Uses ON CONFLICT to handle duplicates atomically.

    Returns:
        (event_id_str, is_duplicate)
        - is_duplicate = True if the idempotency_key already existed
        - event_id_str = the UUID of the persisted or existing row
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO payment_events (
                    payment_id, idempotency_key, customer_id,
                    payment_amount, failure_reason, payment_method, is_subscription,
                    customer_tenure_months, past_successful_payments, past_failed_payments,
                    historical_success_rate, time_since_last_success_days,
                    days_overdue, recovery_attempts_so_far,
                    raw_event_payload, processing_status
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    %s::jsonb, 'RECEIVED'
                )
                ON CONFLICT ON CONSTRAINT uq_payment_events_idempotency_key
                DO NOTHING
                RETURNING event_id
            """, (
                payment_id,
                idempotency_key,
                event.get('customer_id'),
                float(event['payment_amount']),
                event['failure_reason'],
                event['payment_method'],
                int(event['is_subscription']),
                float(event['customer_tenure_months']),
                int(event['past_successful_payments']),
                int(event['past_failed_payments']),
                float(event['historical_success_rate']),
                float(event['time_since_last_success_days']),
                float(event['days_overdue']),
                int(event['recovery_attempts_so_far']),
                json.dumps(event),
            ))
            row = cur.fetchone()

            if row:
                # New row inserted
                event_id = str(row[0])
                conn.commit()
                return event_id, False
            else:
                # Duplicate: fetch existing event_id
                cur.execute(
                    "SELECT event_id FROM payment_events WHERE idempotency_key = %s",
                    (idempotency_key,)
                )
                existing = cur.fetchone()
                conn.commit()
                return str(existing[0]) if existing else None, True


def update_event_status(event_id: str, status: str) -> None:
    """Update the processing_status of a payment_event row."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE payment_events SET processing_status = %s WHERE event_id = %s::uuid",
                (status, event_id)
            )
        conn.commit()


# Recovery decision persistence

def persist_recovery_decision(
    event_id: str,
    model_version_id: Optional[str],
    engine_result: Dict[str, Any],
    trace: Dict[str, Any],
    processing_time_ms: float,
    strategy_name: str = "Rule-Based Real-Time Default",
) -> Optional[str]:
    """
    Persist a recovery decision row. Returns decision_id as string.
    """
    key_factors = trace.get('key_input_factors', [])
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO recovery_decisions (
                    event_id, model_version_id, processing_time_ms,
                    model_probability, recovery_priority, recommended_action,
                    expected_recovery, decision_threshold, effective_retry_cost,
                    expected_retry_net_value, strategy_name, decision_reason,
                    key_input_factors
                ) VALUES (
                    %s::uuid, %s::uuid, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s::jsonb
                )
                RETURNING decision_id
            """, (
                event_id,
                model_version_id,
                processing_time_ms,
                float(engine_result['recovery_probability']),
                engine_result.get('priority', 'LOW'),
                engine_result.get('recommended_action', 'Unknown'),
                float(engine_result.get('expected_recovery', 0.0)),
                float(trace.get('selected_threshold', 0.05)),
                float(trace.get('effective_retry_cost', 50.0)),
                float(trace.get('expected_retry_net_value', 0.0)),
                strategy_name,
                trace.get('decision_reason', ''),
                json.dumps(key_factors),
            ))
            row = cur.fetchone()
        conn.commit()
    return str(row[0]) if row else None


# Audit record persistence

def persist_audit_record(
    decision_id: str,
    audit_record: Dict[str, Any],
) -> None:
    """
    Persist an audit record. Append-only — never updated after insert.
    audit_generated_at = record creation time, NOT payment execution time.
    """
    key_factors_raw = audit_record.get('key_input_factors', '')
    # key_input_factors may be a pipe-delimited string (from AuditTrail CSV compat) or list
    if isinstance(key_factors_raw, str):
        key_factors = key_factors_raw.split(' | ') if key_factors_raw else []
    else:
        key_factors = list(key_factors_raw) if key_factors_raw else []

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO audit_records (
                    audit_id, decision_id, payment_id, audit_generated_at,
                    strategy_name, model_probability, recovery_priority, recommended_action,
                    decision_threshold, effective_retry_cost, expected_recovery,
                    expected_retry_net_value, decision_reason, key_input_factors
                ) VALUES (
                    %s, %s::uuid, %s, %s::timestamptz,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s::jsonb
                )
                ON CONFLICT ON CONSTRAINT uq_audit_records_audit_id DO NOTHING
            """, (
                audit_record['audit_id'],
                decision_id,
                audit_record['payment_id'],
                audit_record['timestamp'],
                audit_record.get('strategy_name', 'Rule-Based Real-Time Default'),
                float(audit_record.get('model_probability', 0.0)),
                audit_record.get('recovery_priority', 'LOW'),
                audit_record.get('recommended_action', 'Unknown'),
                float(audit_record.get('decision_threshold', 0.05)),
                float(audit_record.get('effective_retry_cost', 50.0)),
                float(audit_record.get('expected_recovery', 0.0)),
                float(audit_record.get('expected_retry_net_value', 0.0)),
                audit_record.get('decision_reason', ''),
                json.dumps(key_factors),
            ))
        conn.commit()


# Outcome persistence

def persist_simulated_outcome(
    decision_id: str,
    engine_result: Dict[str, Any],
    recommended_action: str,
) -> None:
    """
    Persist a simulated outcome. outcome_source is always 'SIMULATED'.
    These are synthetic simulation results, NOT real Razorpay transactions.
    """
    simulated_recovered = engine_result.get('simulated_recovered')
    if simulated_recovered is None:
        return  # No simulated outcome available

    recovery_occurred = bool(simulated_recovered)
    recovered_amount = engine_result.get('simulated_recovered_revenue')
    action_cost = engine_result.get('action_cost')
    net_recovered = engine_result.get('net_recovered_revenue')

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO recovery_outcomes (
                    decision_id, outcome_source, action_executed,
                    recovery_occurred, recovered_amount,
                    action_cost, net_recovered_revenue, outcome_notes
                ) VALUES (
                    %s::uuid, 'SIMULATED', %s,
                    %s, %s,
                    %s, %s, %s
                )
            """, (
                decision_id,
                recommended_action,
                recovery_occurred,
                float(recovered_amount) if recovered_amount is not None else None,
                float(action_cost) if action_cost is not None else None,
                float(net_recovered) if net_recovered is not None else None,
                'Synthetic simulation assumption — not a real Razorpay transaction.',
            ))
        conn.commit()


# Duplicate event retrieval

def get_existing_decision_for_event(event_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve the most recent recovery decision for a payment event (for duplicate responses).
    Returns None if not found.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT d.decision_id, d.model_probability, d.recovery_priority,
                           d.recommended_action, d.expected_recovery,
                           d.decision_reason, d.decided_at
                    FROM recovery_decisions d
                    WHERE d.event_id = %s::uuid
                    ORDER BY d.decided_at DESC
                    LIMIT 1
                """, (event_id,))
                row = cur.fetchone()
        if not row:
            return None
        return {
            'decision_id': str(row[0]),
            'model_probability': float(row[1]),
            'recovery_priority': row[2],
            'recommended_action': row[3],
            'expected_recovery': float(row[4]),
            'decision_reason': row[5],
            'decided_at': row[6].isoformat() if row[6] else None,
        }
    except (DatabaseUnavailableError, Exception):
        return None
