-- =============================================================================
-- AI Revenue Recovery Engine — PostgreSQL Schema Design
-- Stage 16: Database Design Reference
--
-- IMPORTANT: This is a DESIGN/REFERENCE file only.
-- Do NOT execute this SQL directly.
-- Do NOT connect to a database with this file.
-- This file documents the intended schema for implementation in a later stage.
-- =============================================================================


-- =============================================================================
-- 1. MODEL VERSIONS
-- Lookup table for model artifact attribution.
-- Register one row per deployed model artifact version.
-- =============================================================================

CREATE TABLE model_versions (
    model_version_id    UUID                        PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name          TEXT                        NOT NULL,
    model_version       TEXT                        NOT NULL,   -- e.g. 'v1.0', or 'unversioned' if MODEL_VERSION env var not set
    calibration_version TEXT,                                   -- Optional calibration artifact version
    description         TEXT,
    registered_at       TIMESTAMP WITH TIME ZONE    NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_model_versions_name_version UNIQUE (model_name, model_version)
);

COMMENT ON TABLE model_versions IS
    'Registry of deployed ML model artifact versions. Every inference row references a model_version_id.';
COMMENT ON COLUMN model_versions.model_version IS
    'Populated from MODEL_VERSION environment variable. Use ''unversioned'' if not configured.';


-- =============================================================================
-- 2. PAYMENT EVENTS
-- One row per unique incoming failed-payment event.
-- This is the idempotency anchor for the entire pipeline.
-- Immutable once written (except processing_status).
--
-- IDENTITY CONCEPTS (IMPORTANT):
--   event_id         — Database surrogate PK (UUID). Identifies the row.
--   payment_id       — Business/payment identifier supplied by caller.
--                      Identifies the payment, NOT a unique event.
--                      Multiple events for the same payment are possible
--                      (e.g. re-assessment after partial remediation).
--   idempotency_key  — Stable key uniquely identifying ONE event-processing
--                      request. UNIQUE constraint is placed on this column.
--                      Supplied by caller, or derived as a documented fallback.
--
-- DO NOT use (payment_id, DATE(event_received_at)) as the uniqueness boundary.
-- =============================================================================

CREATE TABLE payment_events (
    event_id                        UUID                        PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Event identity and idempotency
    payment_id                      TEXT                        NOT NULL,
    idempotency_key                 TEXT                        NOT NULL,
    customer_id                     TEXT,
    event_received_at               TIMESTAMP WITH TIME ZONE    NOT NULL DEFAULT NOW(),
    processing_status               TEXT                        NOT NULL DEFAULT 'RECEIVED',

    -- Input features (mirroring DataValidator.REQUIRED_FEATURES)
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

    -- Forensic audit: full original event payload
    raw_event_payload               JSONB,

    -- -------------------------------------------------------------------------
    -- Constraints
    -- -------------------------------------------------------------------------
    CONSTRAINT uq_payment_events_idempotency_key
        UNIQUE (idempotency_key),

    CONSTRAINT chk_payment_events_amount
        CHECK (payment_amount > 0),

    CONSTRAINT chk_payment_events_is_subscription
        CHECK (is_subscription IN (0, 1)),

    CONSTRAINT chk_payment_events_failure_reason
        CHECK (failure_reason IN (
            'insufficient_funds',
            'invalid_card',
            'technical_error',
            'limit_exceeded'
        )),

    CONSTRAINT chk_payment_events_payment_method
        CHECK (payment_method IN (
            'credit_card',
            'debit_card',
            'upi',
            'bank_transfer'
        )),

    CONSTRAINT chk_payment_events_success_rate
        CHECK (historical_success_rate BETWEEN 0 AND 1),

    CONSTRAINT chk_payment_events_status
        CHECK (processing_status IN (
            'RECEIVED',
            'VALIDATION_FAILED',
            'PREDICTED',
            'DECISIONED',
            'AUDIT_WRITTEN',
            'ACTION_PENDING',    -- FUTURE
            'ACTION_EXECUTED',   -- FUTURE
            'RECOVERED',         -- FUTURE
            'UNRECOVERED'        -- FUTURE
        ))
);

-- Idempotency: unique constraint on idempotency_key (defined inline above).
-- Duplicate detection pattern (design only):
--
--   INSERT INTO payment_events (idempotency_key, payment_id, ...)
--   ON CONFLICT ON CONSTRAINT uq_payment_events_idempotency_key
--   DO NOTHING
--   RETURNING event_id;
--
--   Empty RETURNING → duplicate detected → return previously stored result.
--   Non-empty RETURNING → new event → continue with inference pipeline.

-- Lookup index: payment_id for business-level payment history queries.
-- payment_id is NOT the uniqueness boundary; use idempotency_key for that.
CREATE INDEX idx_payment_events_payment_id
    ON payment_events (payment_id);

CREATE INDEX idx_payment_events_received_at
    ON payment_events (event_received_at);

CREATE INDEX idx_payment_events_status
    ON payment_events (processing_status);

COMMENT ON TABLE payment_events IS
    'One row per unique incoming failed-payment event. idempotency_key is the uniqueness anchor, not payment_id.';
COMMENT ON COLUMN payment_events.event_id IS
    'Database surrogate PK (UUID). Identifies the database row, not the business event.';
COMMENT ON COLUMN payment_events.payment_id IS
    'Business/payment identifier. Identifies the payment, not a unique event. Multiple events for the same payment are valid. Indexed for lookup but NOT the idempotency boundary.';
COMMENT ON COLUMN payment_events.idempotency_key IS
    'Stable key identifying one unique event-processing request. UNIQUE constraint prevents duplicate processing. Supplied by caller (preferred). If absent, the service derives a request-scoped UUID fallback — this fallback does NOT provide strong distributed idempotency.';
COMMENT ON COLUMN payment_events.processing_status IS
    'Event lifecycle state. Only RECEIVED through AUDIT_WRITTEN are in current scope. ACTION_PENDING onward are FUTURE states.';
COMMENT ON COLUMN payment_events.raw_event_payload IS
    'Full original event dict preserved as JSONB for forensic audit. Allows retrospective analysis if schema changes.';




-- =============================================================================
-- 3. RECOVERY DECISIONS
-- One row per inference attempt on an event.
-- Combines ML inference output and rule-based recovery decision (always 1:1).
-- Immutable once written.
-- =============================================================================

CREATE TABLE recovery_decisions (
    decision_id             UUID                        PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id                UUID                        NOT NULL REFERENCES payment_events(event_id),
    model_version_id        UUID                        REFERENCES model_versions(model_version_id),
    decided_at              TIMESTAMP WITH TIME ZONE    NOT NULL DEFAULT NOW(),
    processing_time_ms      NUMERIC(10, 3),

    -- ML inference output
    model_probability       NUMERIC(6, 5)               NOT NULL,

    -- Rule-based recovery decision
    recovery_priority       TEXT                        NOT NULL,
    recommended_action      TEXT                        NOT NULL,

    -- Economic estimates (simulation assumptions — NOT real financial outcomes)
    expected_recovery       NUMERIC(14, 2)              NOT NULL,
    decision_threshold      NUMERIC(5, 4)               NOT NULL,
    effective_retry_cost    NUMERIC(10, 2)              NOT NULL,
    expected_retry_net_value NUMERIC(14, 2)             NOT NULL,

    -- Strategy and explanation
    strategy_name           TEXT                        NOT NULL,
    decision_reason         TEXT                        NOT NULL,
    key_input_factors       JSONB,

    -- -------------------------------------------------------------------------
    -- Constraints
    -- -------------------------------------------------------------------------
    CONSTRAINT chk_recovery_decisions_probability
        CHECK (model_probability BETWEEN 0 AND 1),

    CONSTRAINT chk_recovery_decisions_priority
        CHECK (recovery_priority IN ('HIGH', 'MEDIUM', 'LOW'))
);

CREATE INDEX idx_recovery_decisions_event
    ON recovery_decisions (event_id);

CREATE INDEX idx_recovery_decisions_action
    ON recovery_decisions (recommended_action);

CREATE INDEX idx_recovery_decisions_priority
    ON recovery_decisions (recovery_priority);

CREATE INDEX idx_recovery_decisions_model
    ON recovery_decisions (model_version_id);

COMMENT ON TABLE recovery_decisions IS
    'Merged inference + decision table. ML probability and rule-based action always coexist 1:1 and are stored together.';
COMMENT ON COLUMN recovery_decisions.expected_recovery IS
    'payment_amount * model_probability. This is an expected value estimate under simulation assumptions, not a guarantee.';
COMMENT ON COLUMN recovery_decisions.key_input_factors IS
    'JSON array of human-readable factor strings from DecisionTracer. Stored as JSONB for queryability.';


-- =============================================================================
-- 4. AUDIT RECORDS
-- Immutable, append-only audit log.
-- Maps 1:1 to a recovery_decisions row under current design.
-- Preserves all existing AuditTrail fields.
-- Self-describing: denormalised fields allow reading without joins.
-- =============================================================================

CREATE TABLE audit_records (
    audit_record_id         UUID                        PRIMARY KEY DEFAULT gen_random_uuid(),
    audit_id                TEXT                        NOT NULL,    -- Application SHA-256 hash from AuditTrail._generate_audit_id()
    decision_id             UUID                        NOT NULL REFERENCES recovery_decisions(decision_id),
    payment_id              TEXT                        NOT NULL,    -- Denormalised for self-contained audit lookup
    audit_generated_at      TIMESTAMP WITH TIME ZONE    NOT NULL,    -- WHEN THE AUDIT RECORD WAS GENERATED, not payment time

    -- Denormalised decision fields for audit self-containment
    strategy_name           TEXT                        NOT NULL,
    model_probability       NUMERIC(6, 5)               NOT NULL,
    recovery_priority       TEXT                        NOT NULL,
    recommended_action      TEXT                        NOT NULL,
    decision_threshold      NUMERIC(5, 4)               NOT NULL,
    effective_retry_cost    NUMERIC(10, 2)              NOT NULL,
    expected_recovery       NUMERIC(14, 2)              NOT NULL,
    expected_retry_net_value NUMERIC(14, 2)             NOT NULL,
    decision_reason         TEXT                        NOT NULL,
    key_input_factors       JSONB,

    CONSTRAINT uq_audit_records_audit_id UNIQUE (audit_id)
);

CREATE INDEX idx_audit_records_payment_id
    ON audit_records (payment_id);

CREATE INDEX idx_audit_records_generated_at
    ON audit_records (audit_generated_at);

CREATE INDEX idx_audit_records_decision
    ON audit_records (decision_id);

COMMENT ON TABLE audit_records IS
    'Immutable, append-only audit log. Never updated or deleted. Self-describing by design (denormalised fields).';
COMMENT ON COLUMN audit_records.audit_id IS
    'Application-level SHA-256 hash generated by AuditTrail._generate_audit_id(). Preserved for cross-referencing with CSV exports.';
COMMENT ON COLUMN audit_records.audit_generated_at IS
    'CRITICAL: This is the time the audit record was generated, NOT payment execution time. This semantic must be preserved.';
COMMENT ON COLUMN audit_records.key_input_factors IS
    'Stored as JSON array. The current CSV export uses pipe-delimited strings; this column uses proper JSONB for queryability.';


-- =============================================================================
-- 5. RECOVERY OUTCOMES
-- Tracks simulated or real recovery outcomes per decision.
-- outcome_source IN ('SIMULATED', 'REAL') is always explicit.
-- SIMULATED and REAL outcomes must never be mixed in reporting without filtering.
-- =============================================================================

CREATE TABLE recovery_outcomes (
    outcome_id              UUID                        PRIMARY KEY DEFAULT gen_random_uuid(),
    decision_id             UUID                        NOT NULL REFERENCES recovery_decisions(decision_id),
    outcome_recorded_at     TIMESTAMP WITH TIME ZONE    NOT NULL DEFAULT NOW(),
    outcome_source          TEXT                        NOT NULL,   -- 'SIMULATED' or 'REAL'

    -- What actually happened (nullable until known)
    action_executed         TEXT,
    recovery_occurred       BOOLEAN,                                -- NULL = pending/unknown
    recovered_amount        NUMERIC(14, 2),
    action_cost             NUMERIC(10, 2),
    net_recovered_revenue   NUMERIC(14, 2),
    outcome_notes           TEXT,                                   -- e.g. 'Synthetic simulation assumption'

    -- -------------------------------------------------------------------------
    -- Constraints
    -- -------------------------------------------------------------------------
    CONSTRAINT chk_recovery_outcomes_source
        CHECK (outcome_source IN ('SIMULATED', 'REAL'))
);

CREATE INDEX idx_outcomes_decision
    ON recovery_outcomes (decision_id);

CREATE INDEX idx_outcomes_source
    ON recovery_outcomes (outcome_source);

CREATE INDEX idx_outcomes_recovery_occurred
    ON recovery_outcomes (recovery_occurred);

COMMENT ON TABLE recovery_outcomes IS
    'Tracks simulated or real recovery outcomes. outcome_source must always be explicit. Never mix SIMULATED and REAL in reports without filtering.';
COMMENT ON COLUMN recovery_outcomes.outcome_source IS
    'SIMULATED = from OutcomeSimulator (synthetic). REAL = from actual payment gateway (future). Always explicit.';
COMMENT ON COLUMN recovery_outcomes.recovery_occurred IS
    'NULL means outcome is unknown or pending. false means recovery was attempted but failed. true means recovery confirmed.';


-- =============================================================================
-- END OF DESIGN SCHEMA
-- =============================================================================
-- This file is for design reference only.
-- Do not execute this file against a live database without a migration framework.
-- Do not include credentials in this file.
-- =============================================================================
