# PostgreSQL Persistence Architecture
## Stage 16 — Design Document

---

## 1. Purpose

This document defines the PostgreSQL persistence architecture for the AI Revenue Recovery Engine.

Stage 15 delivers a stateless synchronous inference service. Each call to `predict_event()` produces a fully structured result — validation status, model probability, recovery decision, economic estimate, decision explanation, and a transient in-memory audit record — but **nothing is persisted**. Every result is lost when the response is returned.

This design specifies how all meaningful state should be durably persisted in PostgreSQL, enabling:

- Idempotent event processing (duplicate payment events do not create duplicate decisions)
- Durable audit history (immutable per-decision record)
- Outcome feedback tracking (to compare simulated vs real outcomes in future)
- Model version attribution (every inference is traceable to the model that produced it)
- Event lifecycle management (state transitions from received to decisioned)

---

## 2. Why PostgreSQL Is Now Justified

| Concern | Stage 15 | Stage 16+ |
|---|---|---|
| Event deduplication | None — every call is stateless | PostgreSQL unique constraint on payment_events |
| Audit durability | Transient, in-memory only | Persisted audit_records table |
| Recovery history | CSV file export only | Queryable relational tables |
| Outcome tracking | Synthetic simulation only | Real outcome rows with outcome_source flag |
| Model attribution | Not tracked | model_version on every inference row |
| Idempotency key | Request-scoped UUID per call, not stored, no duplicate protection | `idempotency_key` column with UNIQUE constraint; caller-supplied preferred; fallback UUID documented as weak |


PostgreSQL is the correct choice because:
- The data is **relational** (events have decisions, decisions have audit records, outcomes belong to decisions)
- We need **ACID transactions** (idempotency check + insertion must be atomic)
- We need **structured queries** (audit history filtering, outcome aggregation, model performance analysis)
- PostgreSQL's JSONB type handles semi-structured fields without losing queryability

---

## 3. Architecture Diagram

```mermaid
flowchart TD
    A["Incoming Payment Event\n(payment_id, features)"] --> B["DataValidator"]
    B -->|Invalid| ERR["Return Validation Error\n(not persisted)"]
    B -->|Valid| C["Idempotency Check\n(PostgreSQL: payment_events)"]
    C -->|Duplicate| DUP["Return Stored Result\n(no new rows)"]
    C -->|New| D["RecoveryEngine\n(ML Prediction)"]
    D --> E["DecisionTracer\n(Explanation)"]
    E --> F[("PostgreSQL Transaction")]

    subgraph TX["Atomic Transaction"]
        F --> G["INSERT payment_events"]
        G --> H["INSERT recovery_decisions"]
        H --> I["INSERT audit_records"]
    end

    I --> K["Return Enriched Response\n(includes row IDs)"]
    K --> OUT["[FUTURE] Outcome Feedback\nINSERT recovery_outcomes"]

    style OUT stroke-dasharray: 5 5
    style ERR fill:#f99
    style DUP fill:#ffd
```

---

## 4. Entity Model

Five tables are proposed. `inference_results` and `recovery_decisions` are **combined** into a single table.

**Rationale for merging**: Every time the model runs on an event, it produces exactly one probability and exactly one decision simultaneously. There is no scenario where a probability exists without a decision, or vice versa. A single `recovery_decisions` table with both inference and decision columns avoids a mandatory 1:1 JOIN on every read.

| Entity | Table |
|---|---|
| Payment Event (idempotency anchor) | `payment_events` |
| Inference + Decision (merged) | `recovery_decisions` |
| Audit Record (immutable) | `audit_records` |
| Recovery Outcome (updateable) | `recovery_outcomes` |
| Model Version (lookup) | `model_versions` |

---

## 5. Table Definitions

### 5.1 model_versions

| Column | Type | Nullable | Description |
|---|---|---|---|
| model_version_id | UUID PK | NOT NULL | Surrogate key |
| model_name | TEXT | NOT NULL | e.g. revenue_recovery_model |
| model_version | TEXT | NOT NULL | e.g. v1.0 or 'unversioned' from env var MODEL_VERSION |
| calibration_version | TEXT | NULL | Optional calibration artifact version |
| description | TEXT | NULL | Free-text notes |
| registered_at | TIMESTAMPTZ | NOT NULL | When this version was registered |

> Note: If MODEL_VERSION env var is not set, register with model_version = 'unversioned'. Never hard-code a fake version.

---

### 5.2 payment_events

| Column | Type | Nullable | Immutable | Description |
|---|---|---|---|---|
| event_id | UUID PK | NOT NULL | Yes | Service-generated surrogate key |
| payment_id | TEXT | NOT NULL | Yes | Caller-supplied or service UUID; idempotency boundary |
| customer_id | TEXT | NULL | Yes | Optional caller identifier |
| event_received_at | TIMESTAMPTZ | NOT NULL | Yes | UTC timestamp when event arrived |
| processing_status | TEXT | NOT NULL | No | Lifecycle state (RECEIVED, VALIDATION_FAILED, DECISIONED, AUDIT_WRITTEN) |
| payment_amount | NUMERIC(14,2) | NOT NULL | Yes | Payment amount — NOT float |
| failure_reason | TEXT | NOT NULL | Yes | Validated categorical |
| payment_method | TEXT | NOT NULL | Yes | Validated categorical |
| is_subscription | SMALLINT | NOT NULL | Yes | 0 or 1 |
| customer_tenure_months | NUMERIC(8,2) | NOT NULL | Yes | |
| past_successful_payments | INTEGER | NOT NULL | Yes | |
| past_failed_payments | INTEGER | NOT NULL | Yes | |
| historical_success_rate | NUMERIC(5,4) | NOT NULL | Yes | 0.0000–1.0000 |
| time_since_last_success_days | NUMERIC(8,2) | NOT NULL | Yes | |
| days_overdue | NUMERIC(8,2) | NOT NULL | Yes | |
| recovery_attempts_so_far | INTEGER | NOT NULL | Yes | |
| raw_event_payload | JSONB | NULL | Yes | Full original event dict for forensic audit |

---

### 5.3 recovery_decisions

| Column | Type | Nullable | Immutable | Description |
|---|---|---|---|---|
| decision_id | UUID PK | NOT NULL | Yes | Surrogate key |
| event_id | UUID FK | NOT NULL | Yes | References payment_events |
| model_version_id | UUID FK | NULL | Yes | References model_versions |
| decided_at | TIMESTAMPTZ | NOT NULL | Yes | When the decision was computed |
| processing_time_ms | NUMERIC(10,3) | NULL | Yes | Local inference latency |
| model_probability | NUMERIC(6,5) | NOT NULL | Yes | 0.00000–1.00000 |
| recovery_priority | TEXT | NOT NULL | Yes | HIGH, MEDIUM, LOW |
| recommended_action | TEXT | NOT NULL | Yes | Action string from RecoveryEngine |
| expected_recovery | NUMERIC(14,2) | NOT NULL | Yes | payment_amount × probability |
| decision_threshold | NUMERIC(5,4) | NOT NULL | Yes | Threshold used at decision time |
| effective_retry_cost | NUMERIC(10,2) | NOT NULL | Yes | Assumed retry cost |
| expected_retry_net_value | NUMERIC(14,2) | NOT NULL | Yes | expected_recovery − effective_retry_cost |
| strategy_name | TEXT | NOT NULL | Yes | Strategy label |
| decision_reason | TEXT | NOT NULL | Yes | Human-readable explanation |
| key_input_factors | JSONB | NULL | Yes | JSON array of factor strings |

---

### 5.4 audit_records (immutable, append-only)

| Column | Type | Nullable | Immutable | Description |
|---|---|---|---|---|
| audit_record_id | UUID PK | NOT NULL | Yes | Surrogate key |
| audit_id | TEXT UNIQUE | NOT NULL | Yes | Application SHA-256 hash from AuditTrail |
| decision_id | UUID FK | NOT NULL | Yes | References recovery_decisions |
| payment_id | TEXT | NOT NULL | Yes | Denormalised for self-contained lookup |
| audit_generated_at | TIMESTAMPTZ | NOT NULL | Yes | When audit record was generated (NOT payment time) |
| strategy_name | TEXT | NOT NULL | Yes | |
| model_probability | NUMERIC(6,5) | NOT NULL | Yes | Denormalised for self-containment |
| recovery_priority | TEXT | NOT NULL | Yes | |
| recommended_action | TEXT | NOT NULL | Yes | |
| decision_threshold | NUMERIC(5,4) | NOT NULL | Yes | |
| effective_retry_cost | NUMERIC(10,2) | NOT NULL | Yes | |
| expected_recovery | NUMERIC(14,2) | NOT NULL | Yes | |
| expected_retry_net_value | NUMERIC(14,2) | NOT NULL | Yes | |
| decision_reason | TEXT | NOT NULL | Yes | |
| key_input_factors | JSONB | NULL | Yes | JSON array (not pipe-delimited string) |

> CRITICAL: audit_generated_at means when the audit record was written. It does NOT represent payment execution time.

---

### 5.5 recovery_outcomes

| Column | Type | Nullable | Immutable | Description |
|---|---|---|---|---|
| outcome_id | UUID PK | NOT NULL | Yes | Surrogate key |
| decision_id | UUID FK | NOT NULL | Yes | References recovery_decisions |
| outcome_recorded_at | TIMESTAMPTZ | NOT NULL | Yes | When outcome was recorded |
| outcome_source | TEXT | NOT NULL | Yes | SIMULATED or REAL — always explicit |
| action_executed | TEXT | NULL | Yes | Actual action taken |
| recovery_occurred | BOOLEAN | NULL | Yes | NULL = unknown/pending |
| recovered_amount | NUMERIC(14,2) | NULL | Yes | Amount recovered |
| action_cost | NUMERIC(10,2) | NULL | Yes | Cost of action |
| net_recovered_revenue | NUMERIC(14,2) | NULL | Yes | recovered_amount − action_cost |
| outcome_notes | TEXT | NULL | Yes | e.g. 'Synthetic simulation assumption' |

> outcome_source IN ('SIMULATED', 'REAL') is enforced by a CHECK constraint. SIMULATED and REAL outcomes are never mixed in reporting without explicit filtering.

---

## 6. Relationships

```
model_versions (1) ──────── (N) recovery_decisions
payment_events (1) ──────── (N) recovery_decisions
recovery_decisions (1) ──── (1) audit_records
recovery_decisions (1) ──── (N) recovery_outcomes
```

---

## 7. Idempotency Strategy

### Identity Concept Separation

Three distinct identifiers serve different purposes and must not be confused:

| Identifier | Purpose | Uniqueness guarantee |
|---|---|---|
| `event_id` | Database surrogate PK — identifies the row in `payment_events` | Globally unique (UUID generated by DB) |
| `payment_id` | Business/payment identifier — identifies the payment being recovered | NOT assumed globally unique across events |
| `idempotency_key` | Event-processing identity — identifies one unique invocation of the recovery pipeline | UNIQUE constraint in PostgreSQL |

`payment_id` identifies **the payment**. Multiple events for the same payment are valid and expected (e.g. the same payment re-assessed on a different day after partial remediation). Using `payment_id` as the uniqueness anchor would incorrectly block legitimate re-assessments.

`idempotency_key` identifies **the event-processing request**. The UNIQUE constraint lives here.

### Uniqueness boundary
```
UNIQUE (idempotency_key)
```

No date window. No payment_id-based boundary.

### Idempotency key sourcing

| Source | Priority | Notes |
|---|---|---|
| Caller-supplied `idempotency_key` | **Preferred** | Stable, caller-controlled, strong guarantee |
| Caller-supplied `event_id` (if distinct from payment_id) | Acceptable | Use as idempotency_key if no explicit key provided |
| Service-generated request-scoped UUID | **Fallback only** | Generated fresh per request — provides NO cross-request idempotency; callers must be explicitly informed of this limitation |

When the service derives a request-scoped UUID fallback, it must document this in the response (`idempotency_key_source: "request_scoped_fallback"`). This fallback does **not** prevent duplicate processing if the caller retries without supplying a stable key.

### Duplicate behaviour

```
Event A:
  payment_id = pay_123
  idempotency_key = evt-abc-001 (stable, caller-supplied)
  → Check: no existing row with idempotency_key = 'evt-abc-001'
  → INSERT → process → persist → SUCCESS

Same Event A arrives again (retry, same idempotency_key):
  payment_id = pay_123
  idempotency_key = evt-abc-001
  → Check: existing row found
  → idempotency_status = "duplicate_detected"
  → Return previously stored decision — NO new INSERTs

New legitimate event for same payment (different event):
  payment_id = pay_123
  idempotency_key = evt-abc-002 (different key)
  → Check: no existing row with idempotency_key = 'evt-abc-002'
  → INSERT → process → persist → SUCCESS (new decision row created)
```

### PostgreSQL pattern (design only)
```sql
INSERT INTO payment_events (idempotency_key, payment_id, ...)
ON CONFLICT ON CONSTRAINT uq_payment_events_idempotency_key
DO NOTHING
RETURNING event_id;
-- Empty RETURNING → duplicate detected → return previously stored result
-- Non-empty RETURNING → new event → continue with inference pipeline
```

> **Critical**: `payment_id` is indexed for business-level lookup but is NOT the uniqueness boundary. `idempotency_key` is the sole uniqueness anchor.

---



## 8. Event Lifecycle

| State | Meaning | Current scope |
|---|---|---|
| RECEIVED | Idempotency passed, event stored | YES |
| VALIDATION_FAILED | Rejected by DataValidator — not persisted | YES (response only) |
| PREDICTED | ML model returned probability | YES |
| DECISIONED | Recovery action assigned | YES |
| AUDIT_WRITTEN | Audit record persisted | YES |
| ACTION_PENDING | Action queued for execution | FUTURE |
| ACTION_EXECUTED | Action dispatched to gateway | FUTURE |
| RECOVERED | Payment confirmed recovered | FUTURE |
| UNRECOVERED | Action taken, not recovered | FUTURE |

---

## 9. Transaction Boundaries

### New event (atomic)
```
BEGIN;
  INSERT INTO payment_events ...;
  INSERT INTO recovery_decisions ...;
  INSERT INTO audit_records ...;
COMMIT;
```
All three inserts succeed or all roll back. No partial state.

### Outcome update (separate transaction)
```
BEGIN;
  INSERT INTO recovery_outcomes ...;
  UPDATE payment_events SET processing_status = 'AUDIT_WRITTEN' WHERE event_id = ...;
COMMIT;
```

### Idempotency
Use INSERT ... ON CONFLICT (atomic). Never use SELECT-then-INSERT under concurrent load.

---

## 10. Indexes

| Index | Table | Columns | Purpose |
|---|---|---|---|
| idx_payment_events_payment_id | payment_events | payment_id | Business-level payment lookup (NOT the idempotency boundary) |
| idx_payment_events_received_at | payment_events | event_received_at | Time-range queries |
| idx_payment_events_status | payment_events | processing_status | Lifecycle filtering |
| uq_payment_events_idempotency_key | payment_events | idempotency_key | **Primary idempotency UNIQUE constraint** |
| idx_recovery_decisions_event | recovery_decisions | event_id | FK join |
| idx_recovery_decisions_action | recovery_decisions | recommended_action | Filter by action type |
| idx_recovery_decisions_priority | recovery_decisions | recovery_priority | HIGH/MEDIUM/LOW filtering |
| idx_recovery_decisions_model | recovery_decisions | model_version_id | Model performance analysis |
| idx_audit_records_payment_id | audit_records | payment_id | Direct audit lookup |
| idx_audit_records_generated_at | audit_records | audit_generated_at | Time-range audit queries |
| uq_audit_records_audit_id | audit_records | audit_id | Application hash uniqueness |
| idx_outcomes_decision | recovery_outcomes | decision_id | FK join |
| idx_outcomes_source | recovery_outcomes | outcome_source | SIMULATED vs REAL split |
| idx_outcomes_recovery | recovery_outcomes | recovery_occurred | Success analysis |

---

## 11. Constraints

| Constraint | Table | Rule |
|---|---|---|
| chk_payment_amount | payment_events | payment_amount > 0 |
| chk_is_subscription | payment_events | is_subscription IN (0, 1) |
| chk_failure_reason | payment_events | failure_reason IN ('insufficient_funds', 'invalid_card', 'technical_error', 'limit_exceeded') |
| chk_payment_method | payment_events | payment_method IN ('credit_card', 'debit_card', 'upi', 'bank_transfer') |
| chk_historical_success_rate | payment_events | historical_success_rate BETWEEN 0 AND 1 |
| chk_model_probability | recovery_decisions | model_probability BETWEEN 0 AND 1 |
| chk_recovery_priority | recovery_decisions | recovery_priority IN ('HIGH', 'MEDIUM', 'LOW') |
| chk_outcome_source | recovery_outcomes | outcome_source IN ('SIMULATED', 'REAL') |

---

## 12. Security

- Application connects via a dedicated role (e.g. `recovery_app`) with SELECT, INSERT, UPDATE only.
- No DROP, TRUNCATE, CREATE for the application user.
- `recovery_admin` role handles schema migrations only.
- Credentials via environment variable: `DATABASE_URL=postgresql://user:password@host:port/dbname`
- `.env` file in `.gitignore`. No passwords in Git.
- Development and production use separate credentials and separate database instances.

---

## 13. Retention

- payment_events, recovery_decisions, audit_records, model_versions: Retain indefinitely. Immutable.
- recovery_outcomes: Retain indefinitely. Required for recovery rate analysis.
- Future archival of rows older than a configurable threshold is out of scope for Stage 16.

---

## 14. Local Development Approach

**Recommended: Docker PostgreSQL**

```bash
docker run --name recovery-postgres \
  -e POSTGRES_USER=recovery_app \
  -e POSTGRES_PASSWORD=localdevpassword \
  -e POSTGRES_DB=recovery_db \
  -p 5432:5432 \
  -d postgres:16
```

Docker is recommended (not native installation) because:
- Production will use Docker + Render (PostgreSQL-as-a-service). Docker locally provides closest parity.
- No persistent system-level installation.
- Easy teardown and schema reset for development iteration.

**Do NOT install PostgreSQL natively. Do NOT run this command yet. This is design only.**

---

## 15. Limitations

1. All current predictions are based on synthetic data — no real Razorpay transactions.
2. The persistence layer records decisions and recommendations. It does not trigger real payment actions.
3. Idempotency relies on `idempotency_key`. If callers do not supply a stable key, the service falls back to a request-scoped UUID that provides NO cross-request duplicate protection. This limitation must be communicated to all callers.
4. Audit records are immutable by convention in this design; enforcement via triggers is a future enhancement.
5. Model versioning requires MODEL_VERSION environment variable to be configured. If absent, records as 'unversioned'.

