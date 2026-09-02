# PostgreSQL Persistence — Implementation Guide
## Stage 16 Phase 2

---

## IMPORTANT DISCLAIMERS

- This project uses **synthetic data only**. No real Razorpay transactions are involved.
- PostgreSQL persistence records synthetic ML inference decisions, not real payment executions.
- **Razorpay API integration is intentionally not part of this stage.**
- Simulated recovery outcomes are always labelled `outcome_source = 'SIMULATED'`.

---

## Primary Database: Neon PostgreSQL (Hosted)

**Neon PostgreSQL** is the primary hosted database for this project.

Neon provides standard PostgreSQL with SSL, so the existing five-table schema and all persistence logic work without modification.

### Setup

#### Step 1: Create a Neon database
1. Go to [https://neon.tech](https://neon.tech) and create a free account.
2. Create a new **Project** and a **Database** (e.g. `recovery_db`).
3. From the Neon dashboard, copy the **Connection String**. It will look like:
   ```
   postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require
   ```

#### Step 2: Configure environment
```bash
cp .env.example .env
# Edit .env and paste your Neon connection string as DATABASE_URL:
# DATABASE_URL=postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require
```

**Never commit `.env` to Git.** It is already listed in `.gitignore`.

#### Step 3: Run the application
```bash
source .venv/bin/activate
PYTHONPATH=. streamlit run app/streamlit_app.py
```

The schema is **auto-initialized on first connection** using idempotent `CREATE TABLE IF NOT EXISTS` DDL — no manual migration step is needed.

#### Step 4: Verify in the dashboard
Open the **PostgreSQL Status** tab in the Streamlit dashboard. It will show:
- `PostgreSQL: Connected` with row counts for all five tables.
- **Note:** The tab never displays credentials, DATABASE_URL, or passwords.

---

## Optional: Local Docker PostgreSQL

Docker PostgreSQL may be used as a local development alternative to Neon.
It is **optional** — the application connects through `DATABASE_URL` regardless of whether it points to Neon or a local container.

```bash
# Configure local credentials in .env
POSTGRES_DB=recovery_db
POSTGRES_USER=recovery_app
POSTGRES_PASSWORD=your_secure_local_password_here
DATABASE_URL=postgresql://recovery_app:your_secure_local_password_here@localhost:5432/recovery_db

# Start local PostgreSQL
docker compose up -d

# Verify container is healthy
docker compose ps
```

> **Primary**: Neon PostgreSQL (hosted, with SSL)
> **Optional local fallback**: Docker PostgreSQL

The application does not depend on `localhost`. It connects only via `DATABASE_URL`.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | **Yes** (for persistence) | None | Neon (or Docker) PostgreSQL connection string |
| `MODEL_VERSION` | No | `unversioned` | Model version label for tracking |
| `POSTGRES_DB` | Only for Docker | `recovery_db` | Docker container DB name |
| `POSTGRES_USER` | Only for Docker | `recovery_app` | Docker container DB user |
| `POSTGRES_PASSWORD` | Only for Docker | **must be set** | Docker container DB password |
| `POSTGRES_PORT` | Only for Docker | `5432` | Docker host port |
| `TEST_DATABASE_URL` | Only for integration tests | None | Separate test database — never use production DB |

**Neon connection strings must include `?sslmode=require`.** The application driver (`psycopg`) honours the connection string SSL parameters automatically.

---

## Application Flow with Persistence

```
Incoming Payment Event (dict)
  │
  ├─ DataValidator.validate_record()
  │      └── INVALID → return error response (NOT persisted)
  │
  ├─ Idempotency key resolution:
  │      1. caller-supplied 'idempotency_key'   → source = 'caller_supplied'
  │      2. caller-supplied 'event_id'          → source = 'event_id_fallback'
  │      3. service-generated UUID              → source = 'request_scoped_fallback'
  │         ⚠ fallback does NOT prevent cross-request duplicates
  │
  ├─ initialize_schema() [idempotent, runs once]
  │
  ├─ persist_payment_event() [INSERT ... ON CONFLICT ON idempotency_key]
  │      ├── New event → continue
  │      └── Duplicate detected → return stored result, skip ML inference
  │
  ├─ RecoveryEngine.predict_recovery()  [ML prediction, unchanged]
  ├─ DecisionTracer.generate_trace()   [explanation, unchanged]
  ├─ AuditTrail.create_audit_record()  [in-memory audit, unchanged]
  │
  ├─ persist_recovery_decision()       [INSERT recovery_decisions]
  ├─ persist_audit_record()            [INSERT audit_records, append-only]
  └─ persist_simulated_outcome()       [INSERT recovery_outcomes, source='SIMULATED']
```

### Graceful degradation

If `DATABASE_URL` is not set or the database is unreachable:
- Persistence is silently skipped.
- The inference pipeline runs in **stateless mode** — predictions, decisions, and explanations still work.
- `response['persistence']['persisted'] = False`
- The Streamlit dashboard shows `PostgreSQL: Unavailable` in Tab 8.

---

## Idempotency Behavior

**Uniqueness boundary:** `UNIQUE (idempotency_key)` — no date window, no `payment_id` boundary.

| Priority | Source | Guarantee |
|---|---|---|
| 1 | `event['idempotency_key']` (caller-supplied) | Strong — UNIQUE constraint |
| 2 | `event['event_id']` (caller-supplied) | Strong — if stable per event |
| 3 | Service-generated UUID per request | **Weak** — no cross-request duplicate protection |

When fallback UUID is used, `idempotency_key_source: "request_scoped_fallback"` is disclosed in the response.

---

## Persistence Tables

| Table | Purpose | Immutable? |
|---|---|---|
| `model_versions` | ML model artifact registry | Yes |
| `payment_events` | Idempotency anchor + input features | Yes (except `processing_status`) |
| `recovery_decisions` | ML inference + rule-based decision | Yes |
| `audit_records` | Compliance-grade audit log | Yes |
| `recovery_outcomes` | Simulated or real outcomes (`outcome_source` required) | Yes once written |

`audit_generated_at` = when the record was created, **not** payment execution time.

---

## Security

- `DATABASE_URL` is read from environment only — never hard-coded in source code.
- Credentials are never logged, even on connection failure.
- `.env` is in `.gitignore`. Only `.env.example` (with placeholders) is committed.
- The Streamlit dashboard never displays `DATABASE_URL`, passwords, or hostnames.
- The application role needs only `SELECT`, `INSERT`, `UPDATE` — not `DROP` or `CREATE`.
- No Razorpay credentials are required or stored.

---

## Testing

### Unit tests (no DB required)
```bash
PYTHONPATH=. pytest -q
# 90 passed, 6 skipped (integration tests — require TEST_DATABASE_URL)
```

### Integration tests (require a dedicated test database)

> **IMPORTANT:** Never set `TEST_DATABASE_URL` to your production Neon database.
> Use a separate Neon project/database or an isolated Docker instance for tests.

```bash
# Create a separate Neon test database, then:
export TEST_DATABASE_URL="postgresql://USER:PASSWORD@HOST/TESTDB?sslmode=require"
PYTHONPATH=. pytest -q
# All 96 tests should pass (6 integration tests now run)
```

Integration tests are **automatically skipped** unless `TEST_DATABASE_URL` is explicitly provided.
The application's `DATABASE_URL` is **never used** for integration tests.

---

## Future Deployment (Render)

The application is structured to accept `DATABASE_URL` through deployment environment variables.
When deploying to Render:

1. Set `DATABASE_URL` in the Render environment settings (pointing to Neon).
2. Set `MODEL_VERSION` as needed.
3. Do **not** commit credentials to Git.

Render deployment is a **future stage** and is not part of Stage 16.

---

## Limitations

1. **Synthetic data only.** No real customer or payment data.
2. **No real Razorpay execution.** The system recommends recovery actions; it does not call Razorpay APIs.
3. **Request-scoped UUID fallback** provides no cross-request duplicate protection.
4. **Schema managed in code** via `IF NOT EXISTS` DDL. No migration framework yet.
5. **Not production-ready.** Neon connectivity does not make this a production payment system.
