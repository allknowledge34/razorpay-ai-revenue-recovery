import os
import time
import datetime
import uuid
from typing import Dict, Any, Optional

from src.validation.data_validator import DataValidator
from src.domain.recovery_engine import RecoveryEngine
from src.services.explanation.decision_trace import DecisionTracer
from src.services.audit.audit_trail import AuditTrail
from src.services.recovery.recovery_orchestrator import RecoveryOrchestrator


class RecoveryInferenceService:
    """
    Synchronous inference service layer.

    Accepts ONE incoming failed-payment event, validates it, runs ML prediction,
    assigns a recovery decision, generates a trace, and produces an audit record.

    If PostgreSQL is available (DATABASE_URL configured), the pipeline also persists:
      - payment event (with idempotency check)
      - recovery decision
      - audit record
      - simulated outcomes (if present)

    If PostgreSQL is unavailable, the pipeline runs in stateless mode —
    inference still succeeds; only persistence is skipped.

    DISCLAIMER: All predictions and simulated outcomes are based on synthetic
    data. This service does not execute real Razorpay payment actions.
    """

    def __init__(self, simulator_cost: float = 50.0, simulator_threshold: float = 0.05,
                 enable_persistence: bool = True):
        self.validator = DataValidator()
        self.engine = RecoveryEngine()
        self.tracer = DecisionTracer(simulator_cost=simulator_cost, simulator_threshold=simulator_threshold)
        self.auditor = AuditTrail(simulator_cost=simulator_cost, simulator_threshold=simulator_threshold)
        self.orchestrator = RecoveryOrchestrator()
        self.enable_persistence = enable_persistence

        self._model_version = os.environ.get('MODEL_VERSION', 'unversioned')
        self._model_version_id: Optional[str] = None



    def _get_model_version_id(self) -> Optional[str]:
        """Resolve and cache model_version_id from model_versions table."""
        if self._model_version_id is not None:
            return self._model_version_id
        try:
            from src.infrastructure.database.database import get_or_create_model_version
            self._model_version_id = get_or_create_model_version(
                model_name='revenue_recovery_model',
                model_version=self._model_version,
            )
            return self._model_version_id
        except Exception:
            return None  # DB unavailable — proceed without version FK


    def predict_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single failed payment event through the recovery decision pipeline.

        Idempotency key priority (for persistence):
          1. caller-supplied 'idempotency_key'
          2. caller-supplied 'event_id'
          3. request-scoped UUID fallback (does NOT guarantee cross-request deduplication)
        """
        start_time = time.perf_counter()

        payment_id = str(event.get('payment_id', str(uuid.uuid4())))
        customer_id = event.get('customer_id')

        # Resolve idempotency key
        from src.infrastructure.database.db_persistence import resolve_idempotency_key
        idempotency_key, idempotency_key_source = resolve_idempotency_key(event)

        result: Dict[str, Any] = {
            'event_identity': {
                'payment_id': payment_id,
                'customer_id': customer_id,
                'idempotency_key': idempotency_key,
                'idempotency_key_source': idempotency_key_source,
            },
            'validation': {},
            'prediction': {},
            'decision': {},
            'economic_estimate': {},
            'explanation': {},
            'persistence': {'persisted': False, 'is_duplicate': False},
            'processing_metadata': {}
        }

        val_result = self.validator.validate_record(event)
        result['validation'] = {
            'is_valid': val_result.is_valid,
            'errors': val_result.errors
        }

        if not val_result.is_valid:
            result['processing_metadata'] = {
                'status': 'error',
                'error_type': 'validation_error',
                'processing_time_ms': round((time.perf_counter() - start_time) * 1000, 2)
            }
            return result

        try:
            db_event_id: Optional[str] = None
            is_duplicate = False

            if self.enable_persistence:
                try:
                    from src.infrastructure.database.db_persistence import (
                        persist_payment_event, update_event_status,
                        get_existing_decision_for_event, persist_recovery_decision,
                        persist_audit_record, persist_simulated_outcome
                    )
                    from src.infrastructure.database.database import initialize_schema

                    initialize_schema()

                    db_event_id, is_duplicate = persist_payment_event(
                        event=event,
                        payment_id=payment_id,
                        idempotency_key=idempotency_key,
                    )

                    result['persistence']['persisted'] = db_event_id is not None
                    result['persistence']['is_duplicate'] = is_duplicate
                    if db_event_id:
                        result['persistence']['event_id'] = db_event_id

                    if is_duplicate:
                        existing = get_existing_decision_for_event(db_event_id) if db_event_id else None
                        result['persistence']['idempotency_status'] = 'duplicate_detected'
                        result['processing_metadata'] = {
                            'status': 'duplicate',
                            'message': 'Duplicate event detected. Returning previously stored decision.',
                            'idempotency_key_source': idempotency_key_source,
                            'processing_time_ms': round((time.perf_counter() - start_time) * 1000, 2),
                        }
                        if existing:
                            result['decision'] = {
                                'recommended_action': existing['recommended_action'],
                                'recovery_priority': existing['recovery_priority'],
                            }
                            result['prediction'] = {'recovery_probability': existing['model_probability']}
                            result['economic_estimate'] = {'expected_recovery': existing['expected_recovery']}
                        return result

                except Exception:
                    result['persistence']['persisted'] = False
                    result['persistence']['db_error'] = 'Database unavailable. Running in stateless mode.'

            engine_result = self.engine.predict_recovery(event)

            prob = engine_result.get('recovery_probability')
            prob_val = self.validator.validate_prediction_probability(prob)
            if not prob_val.is_valid:
                result['validation']['is_valid'] = False
                result['validation']['errors'].extend(prob_val.errors)
                result['processing_metadata'] = {
                    'status': 'error',
                    'error_type': 'model_output_error',
                    'processing_time_ms': round((time.perf_counter() - start_time) * 1000, 2)
                }
                return result

            trace = self.tracer.generate_trace(event, engine_result)

            # Execute Bounded Recovery Workflow
            event_context = {
                'payment_id': payment_id,
                'recovery_probability': prob,
                'recovery_attempts_so_far': int(event.get('recovery_attempts_so_far', 0)),
                'payment_amount': float(event.get('payment_amount', 0.0)),
                'expected_recovery': engine_result.get('expected_recovery', 0.0),
                'recommended_action': engine_result.get('recommended_action', 'Unknown')
            }

            # Using payment_id hash for deterministic seeded execution
            import hashlib
            seed_bytes = hashlib.sha256(payment_id.encode("utf-8")).digest()
            seed = int.from_bytes(seed_bytes[:8], "big") % (2**32)
            orch_result = self.orchestrator.process_event(event_context, seed=seed)
            result['bounded_recovery'] = orch_result

            # Populate engine_result for persistence compatibility
            engine_result['simulated_recovered'] = orch_result.get('simulated_recovered')
            engine_result['simulated_recovered_revenue'] = orch_result.get('recovered_amount')
            engine_result['action_cost'] = orch_result.get('action_cost')
            engine_result['net_recovered_revenue'] = orch_result.get('net_recovered_revenue')


            result['prediction'] = {'recovery_probability': prob}
            result['decision'] = {
                'recommended_action': engine_result.get('recommended_action', 'Unknown'),
                'recovery_priority': engine_result.get('priority', 'UNKNOWN'),
            }
            result['economic_estimate'] = {
                'expected_recovery': engine_result.get('expected_recovery', 0.0),
                'effective_retry_cost': trace.get('effective_retry_cost', 0.0),
                'expected_retry_net_value': trace.get('expected_retry_net_value', 0.0),
            }
            result['explanation'] = {
                'decision_reason': trace.get('decision_reason', ''),
                'key_input_factors': trace.get('key_input_factors', []),
            }

            audit_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
            event_for_audit = dict(event)
            event_for_audit['payment_id'] = payment_id
            audit_record = self.auditor.create_audit_record(
                record=event_for_audit,
                engine_result=engine_result,
                strategy_name='Rule-Based Real-Time Default',
                audit_timestamp=audit_ts,
            )
            result['audit_record'] = audit_record

            processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            result['processing_metadata'] = {
                'status': 'success',
                'processing_time_ms': processing_time_ms,
                'timestamp_utc': audit_ts,
                'idempotency_key_source': idempotency_key_source,
                'model_version': self._model_version,
            }

            if self.enable_persistence and db_event_id and not is_duplicate:
                try:
                    model_version_id = self._get_model_version_id()

                    update_event_status(db_event_id, 'DECISIONED')

                    decision_id = persist_recovery_decision(
                        event_id=db_event_id,
                        model_version_id=model_version_id,
                        engine_result=engine_result,
                        trace=trace,
                        processing_time_ms=processing_time_ms,
                        strategy_name='Rule-Based Real-Time Default',
                    )

                    if decision_id:
                        result['persistence']['decision_id'] = decision_id
                        persist_audit_record(decision_id=decision_id, audit_record=audit_record)
                        update_event_status(db_event_id, 'AUDIT_WRITTEN')
                        result['persistence']['audit_persisted'] = True

                        persist_simulated_outcome(
                            decision_id=decision_id,
                            engine_result=engine_result,
                            recommended_action=engine_result.get('recommended_action', 'Unknown'),
                        )

                except Exception:
                    result['persistence']['persist_error'] = 'Decision/audit persistence failed. Inference result still valid.'

        except Exception:
            result['processing_metadata'] = {
                'status': 'error',
                'error_type': 'pipeline_exception',
                'error_message': 'Inference pipeline failed. Please retry or inspect server logs.',
                'processing_time_ms': round((time.perf_counter() - start_time) * 1000, 2),
            }

        return result
