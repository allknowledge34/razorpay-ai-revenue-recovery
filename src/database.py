"""
src/database.py — PostgreSQL connection layer.

Reads DATABASE_URL from environment (via .env or system env).
Provides a context-managed connection and a schema initializer.
Fails safely when DATABASE_URL is missing or the database is unreachable.

Credentials are never logged or exposed in error messages.
"""

import os
import json
import logging
from contextlib import contextmanager
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; env vars can be set directly

logger = logging.getLogger(__name__)

# Connection helpers

def get_database_url() -> Optional[str]:
    """Return DATABASE_URL from environment, or None if not configured."""
    return os.environ.get("DATABASE_URL")


@contextmanager
def get_connection():
    """
    Context manager that yields a psycopg connection.

    Usage:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(...)
            conn.commit()

    Raises DatabaseUnavailableError if DATABASE_URL is missing or connection fails.
    Does NOT log the connection string or credentials.
    """
    import psycopg

    url = get_database_url()
    if not url:
        raise DatabaseUnavailableError(
            "DATABASE_URL environment variable is not set. "
            "Copy .env.example to .env and configure your PostgreSQL credentials."
        )

    conn = None
    try:
        conn = psycopg.connect(url)
        yield conn
    except psycopg.OperationalError as e:
        raise DatabaseUnavailableError(
            f"Could not connect to PostgreSQL. Check DATABASE_URL and ensure the database is running."
        ) from e
    finally:
        if conn and not conn.closed:
            conn.close()


def is_database_available() -> bool:
    """Return True if a database connection can be established, False otherwise."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return True
    except (DatabaseUnavailableError, Exception):
        return False


# Schema initialization

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS model_versions (
    model_version_id    UUID                        PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name          TEXT                        NOT NULL,
    model_version       TEXT                        NOT NULL,
    calibration_version TEXT,
    description         TEXT,
    registered_at       TIMESTAMP WITH TIME ZONE    NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_model_versions_name_version UNIQUE (model_name, model_version)
);

CREATE TABLE IF NOT EXISTS payment_events (
    event_id                        UUID                        PRIMARY KEY DEFAULT gen_random_uuid(),
    payment_id                      TEXT                        NOT NULL,
    idempotency_key                 TEXT                        NOT NULL,
    customer_id                     TEXT,
    event_received_at               TIMESTAMP WITH TIME ZONE    NOT NULL DEFAULT NOW(),
    processing_status               TEXT                        NOT NULL DEFAULT 'RECEIVED',
    payment_amount                  NUMERIC(14, 2)              NOT NULL,
    failure_reason                  TEXT                        NOT NULL,
    payment_method                  TEXT                        NOT NULL,
    is_subscription                 SMALLINT                    NOT NULL,
    customer_tenure_months          NUMERIC(8, 2)               NOT NULL,
    past_successful_payments        INTEGER                     NOT NULL,
    past_failed_payments            INTEGER                     NOT NULL,
    historical_success_rate         NUMERIC(5, 4)               NOT NULL,
    time_since_last_success_days    NUMERIC(8, 2)               NOT NULL,
    days_overdue                    NUMERIC(8, 2)               NOT NULL,
    recovery_attempts_so_far        INTEGER                     NOT NULL,
    raw_event_payload               JSONB,
    CONSTRAINT uq_payment_events_idempotency_key UNIQUE (idempotency_key),
    CONSTRAINT chk_payment_events_amount         CHECK (payment_amount > 0),
    CONSTRAINT chk_payment_events_is_subscription CHECK (is_subscription IN (0, 1)),
    CONSTRAINT chk_payment_events_failure_reason CHECK (failure_reason IN (
        'insufficient_funds', 'invalid_card', 'technical_error', 'limit_exceeded')),
    CONSTRAINT chk_payment_events_payment_method CHECK (payment_method IN (
        'credit_card', 'debit_card', 'upi', 'bank_transfer')),
    CONSTRAINT chk_payment_events_success_rate   CHECK (historical_success_rate BETWEEN 0 AND 1),
    CONSTRAINT chk_payment_events_status         CHECK (processing_status IN (
        'RECEIVED','VALIDATION_FAILED','PREDICTED','DECISIONED','AUDIT_WRITTEN',
        'ACTION_PENDING','ACTION_EXECUTED','RECOVERED','UNRECOVERED'))
);

CREATE INDEX IF NOT EXISTS idx_payment_events_payment_id   ON payment_events (payment_id);
CREATE INDEX IF NOT EXISTS idx_payment_events_received_at  ON payment_events (event_received_at);
CREATE INDEX IF NOT EXISTS idx_payment_events_status       ON payment_events (processing_status);

CREATE TABLE IF NOT EXISTS recovery_decisions (
    decision_id              UUID                        PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id                 UUID                        NOT NULL REFERENCES payment_events(event_id),
    model_version_id         UUID                        REFERENCES model_versions(model_version_id),
    decided_at               TIMESTAMP WITH TIME ZONE    NOT NULL DEFAULT NOW(),
    processing_time_ms       NUMERIC(10, 3),
    model_probability        NUMERIC(6, 5)               NOT NULL,
    recovery_priority        TEXT                        NOT NULL,
    recommended_action       TEXT                        NOT NULL,
    expected_recovery        NUMERIC(14, 2)              NOT NULL,
    decision_threshold       NUMERIC(5, 4)               NOT NULL,
    effective_retry_cost     NUMERIC(10, 2)              NOT NULL,
    expected_retry_net_value NUMERIC(14, 2)              NOT NULL,
    strategy_name            TEXT                        NOT NULL,
    decision_reason          TEXT                        NOT NULL,
    key_input_factors        JSONB,
    CONSTRAINT chk_rd_probability CHECK (model_probability BETWEEN 0 AND 1),
    CONSTRAINT chk_rd_priority    CHECK (recovery_priority IN ('HIGH', 'MEDIUM', 'LOW'))
);

CREATE INDEX IF NOT EXISTS idx_recovery_decisions_event    ON recovery_decisions (event_id);
CREATE INDEX IF NOT EXISTS idx_recovery_decisions_action   ON recovery_decisions (recommended_action);
CREATE INDEX IF NOT EXISTS idx_recovery_decisions_priority ON recovery_decisions (recovery_priority);
CREATE INDEX IF NOT EXISTS idx_recovery_decisions_model    ON recovery_decisions (model_version_id);

CREATE TABLE IF NOT EXISTS audit_records (
    audit_record_id          UUID                        PRIMARY KEY DEFAULT gen_random_uuid(),
    audit_id                 TEXT                        NOT NULL,
    decision_id              UUID                        NOT NULL REFERENCES recovery_decisions(decision_id),
    payment_id               TEXT                        NOT NULL,
    audit_generated_at       TIMESTAMP WITH TIME ZONE    NOT NULL,
    strategy_name            TEXT                        NOT NULL,
    model_probability        NUMERIC(6, 5)               NOT NULL,
    recovery_priority        TEXT                        NOT NULL,
    recommended_action       TEXT                        NOT NULL,
    decision_threshold       NUMERIC(5, 4)               NOT NULL,
    effective_retry_cost     NUMERIC(10, 2)              NOT NULL,
    expected_recovery        NUMERIC(14, 2)              NOT NULL,
    expected_retry_net_value NUMERIC(14, 2)              NOT NULL,
    decision_reason          TEXT                        NOT NULL,
    key_input_factors        JSONB,
    CONSTRAINT uq_audit_records_audit_id UNIQUE (audit_id)
);

CREATE INDEX IF NOT EXISTS idx_audit_records_payment_id    ON audit_records (payment_id);
CREATE INDEX IF NOT EXISTS idx_audit_records_generated_at  ON audit_records (audit_generated_at);
CREATE INDEX IF NOT EXISTS idx_audit_records_decision      ON audit_records (decision_id);

CREATE TABLE IF NOT EXISTS recovery_outcomes (
    outcome_id               UUID                        PRIMARY KEY DEFAULT gen_random_uuid(),
    decision_id              UUID                        NOT NULL REFERENCES recovery_decisions(decision_id),
    outcome_recorded_at      TIMESTAMP WITH TIME ZONE    NOT NULL DEFAULT NOW(),
    outcome_source           TEXT                        NOT NULL,
    action_executed          TEXT,
    recovery_occurred        BOOLEAN,
    recovered_amount         NUMERIC(14, 2),
    action_cost              NUMERIC(10, 2),
    net_recovered_revenue    NUMERIC(14, 2),
    outcome_notes            TEXT,
    CONSTRAINT chk_ro_source CHECK (outcome_source IN ('SIMULATED', 'REAL'))
);

CREATE INDEX IF NOT EXISTS idx_outcomes_decision          ON recovery_outcomes (decision_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_source            ON recovery_outcomes (outcome_source);
CREATE INDEX IF NOT EXISTS idx_outcomes_recovery_occurred ON recovery_outcomes (recovery_occurred);
"""


def initialize_schema() -> None:
    """
    Create all five tables and indexes if they do not yet exist.
    Safe to call on every startup (idempotent via IF NOT EXISTS).
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA_SQL)
        conn.commit()
    logger.info("Database schema initialized successfully.")


# Model version helpers

def get_or_create_model_version(model_name: str, model_version: str,
                                 calibration_version: Optional[str] = None) -> str:
    """
    Return the model_version_id UUID for the given model_name + model_version.
    Creates a new row if one does not exist.
    Returns the UUID as a string.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO model_versions (model_name, model_version, calibration_version)
                VALUES (%s, %s, %s)
                ON CONFLICT ON CONSTRAINT uq_model_versions_name_version
                DO NOTHING
                RETURNING model_version_id
            """, (model_name, model_version, calibration_version))
            row = cur.fetchone()
            if row:
                model_version_id = str(row[0])
            else:
                cur.execute("""
                    SELECT model_version_id FROM model_versions
                    WHERE model_name = %s AND model_version = %s
                """, (model_name, model_version))
                model_version_id = str(cur.fetchone()[0])
        conn.commit()
    return model_version_id


# Stats helpers for Streamlit

def get_table_counts() -> dict:
    """Return row counts for all five tables. Returns zeros if DB unavailable."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                counts = {}
                for table in ['payment_events', 'recovery_decisions', 'audit_records',
                               'recovery_outcomes', 'model_versions']:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    counts[table] = cur.fetchone()[0]
        return counts
    except (DatabaseUnavailableError, Exception):
        return {t: 0 for t in ['payment_events', 'recovery_decisions', 'audit_records',
                                'recovery_outcomes', 'model_versions']}


# Custom exceptions

class DatabaseUnavailableError(Exception):
    """Raised when the database cannot be reached or DATABASE_URL is not set."""
    pass
