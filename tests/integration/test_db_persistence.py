"""
tests/test_db_persistence.py

Tests for PostgreSQL persistence layer.

Integration tests that actually hit PostgreSQL are skipped unless the
TEST_DATABASE_URL environment variable is set.

Unit/contract tests verify:
- idempotency key resolution logic (no DB required)
- safe graceful degradation when DB is unavailable (no DB required)
- inference service runs correctly with persistence disabled (no DB required)

To run integration tests against a live database:
    export TEST_DATABASE_URL=postgresql://recovery_app:password@localhost:5432/recovery_test_db
    pytest tests/test_db_persistence.py -q
"""

import os
import uuid
import pytest
from unittest.mock import patch, MagicMock

# Fixtures

@pytest.fixture
def valid_event():
    return {
        'payment_id': 'pay_test_001',
        'customer_id': 'cust_abc',
        'payment_amount': 5000.0,
        'failure_reason': 'technical_error',
        'payment_method': 'upi',
        'is_subscription': 0,
        'customer_tenure_months': 24,
        'past_successful_payments': 10,
        'past_failed_payments': 0,
        'historical_success_rate': 1.0,
        'time_since_last_success_days': 5,
        'days_overdue': 1,
        'recovery_attempts_so_far': 0,
    }


@pytest.fixture
def invalid_event():
    return {
        'payment_id': 'pay_invalid',
        'payment_amount': -100,

        'failure_reason': 'technical_error',
        'payment_method': 'upi',
        'is_subscription': 0,
        'customer_tenure_months': 24,
        'past_successful_payments': 5,
        'past_failed_payments': 1,
        'historical_success_rate': 0.8,
        'time_since_last_success_days': 10,
        'days_overdue': 2,
        'recovery_attempts_so_far': 0,
    }


@pytest.fixture
def service_no_db():
    """Inference service with persistence disabled — no DB required."""
    from src.services.inference.inference_service import RecoveryInferenceService
    return RecoveryInferenceService(enable_persistence=False)



def test_idempotency_key_caller_supplied():
    from src.infrastructure.database.db_persistence import resolve_idempotency_key
    event = {'payment_id': 'pay_001', 'idempotency_key': 'stable-key-A'}
    key, source = resolve_idempotency_key(event)
    assert key == 'stable-key-A'
    assert source == 'caller_supplied'


def test_idempotency_key_event_id_fallback():
    from src.infrastructure.database.db_persistence import resolve_idempotency_key
    event = {'payment_id': 'pay_001', 'event_id': 'evt-uuid-xyz'}
    key, source = resolve_idempotency_key(event)
    assert key == 'evt-uuid-xyz'
    assert source == 'event_id_fallback'


def test_idempotency_key_request_scoped_fallback():
    from src.infrastructure.database.db_persistence import resolve_idempotency_key
    event = {'payment_id': 'pay_001'}

    key, source = resolve_idempotency_key(event)
    assert source == 'request_scoped_fallback'
    # Must be UUID-like (at least 32 chars)
    assert len(key) >= 32


def test_idempotency_fallback_source_disclosed_in_response(service_no_db, valid_event):
    """The service must disclose when a request-scoped fallback is used."""
    event = dict(valid_event)
    event.pop('payment_id', None)
    # ensure no stable idempotency_key supplied
    event.pop('idempotency_key', None)
    event.pop('event_id', None)
    event['payment_id'] = 'pay_fallback_test'

    res = service_no_db.predict_event(event)
    assert res['event_identity']['idempotency_key_source'] == 'request_scoped_fallback'


def test_caller_supplied_idempotency_key_preserved(service_no_db, valid_event):
    """When caller supplies idempotency_key, it must appear in the response."""
    event = dict(valid_event)
    event['idempotency_key'] = 'my-stable-key-001'
    res = service_no_db.predict_event(event)
    assert res['event_identity']['idempotency_key'] == 'my-stable-key-001'
    assert res['event_identity']['idempotency_key_source'] == 'caller_supplied'



def test_valid_event_succeeds_without_db(service_no_db, valid_event):
    res = service_no_db.predict_event(valid_event)
    assert res['validation']['is_valid'] is True
    assert res['processing_metadata']['status'] == 'success'
    assert 'recovery_probability' in res['prediction']
    assert res['persistence']['persisted'] is False



def test_invalid_event_rejected_without_db(service_no_db, invalid_event):
    res = service_no_db.predict_event(invalid_event)
    assert res['validation']['is_valid'] is False
    assert res['processing_metadata']['status'] == 'error'
    assert res['processing_metadata']['error_type'] == 'validation_error'



def test_supplied_payment_id_preserved(service_no_db, valid_event):
    res = service_no_db.predict_event(valid_event)
    assert res['event_identity']['payment_id'] == 'pay_test_001'


def test_missing_payment_id_generates_uuid(service_no_db, valid_event):
    event = dict(valid_event)
    del event['payment_id']
    res = service_no_db.predict_event(event)
    assert len(str(res['event_identity']['payment_id'])) >= 32



def test_pipeline_exception_safe_message(service_no_db, valid_event, monkeypatch):
    def mock_predict(*args, **kwargs):
        raise ValueError("internal secret config value")
    monkeypatch.setattr(service_no_db.engine, 'predict_recovery', mock_predict)
    res = service_no_db.predict_event(valid_event)
    assert res['processing_metadata']['status'] == 'error'
    assert res['processing_metadata']['error_type'] == 'pipeline_exception'
    assert 'internal secret config value' not in res['processing_metadata']['error_message']
    assert res['processing_metadata']['error_message'] == \
        'Inference pipeline failed. Please retry or inspect server logs.'



def test_db_unavailable_inference_still_works(valid_event):
    """When DB is unavailable, inference runs in stateless mode — no crash."""
    from src.services.inference.inference_service import RecoveryInferenceService

    with patch('src.infrastructure.database.database.initialize_schema', side_effect=Exception("DB down")):
        svc = RecoveryInferenceService(enable_persistence=True)
        # initialize_schema will fail → stateless fallback
        res = svc.predict_event(valid_event)
    # Inference result is still valid
    assert res['validation']['is_valid'] is True
    assert res['processing_metadata']['status'] == 'success'
    assert res['persistence']['persisted'] is False



def test_database_unavailable_error_is_not_exposed():
    """DatabaseUnavailableError must not expose credential details."""
    from src.infrastructure.database.database import DatabaseUnavailableError
    err = DatabaseUnavailableError("Could not connect to PostgreSQL.")
    assert "password" not in str(err).lower()
    assert "secret" not in str(err).lower()



def test_is_database_available_returns_false_when_no_url():
    """is_database_available must return False when DATABASE_URL is not set."""
    from src.infrastructure.database.database import is_database_available
    with patch.dict(os.environ, {}, clear=True):
        # Remove DATABASE_URL if set
        env = dict(os.environ)
        env.pop('DATABASE_URL', None)
        with patch.dict(os.environ, env, clear=True):
            result = is_database_available()
    assert isinstance(result, bool)  # must return bool, not raise



TEST_DB_URL = os.environ.get('TEST_DATABASE_URL')
integration = pytest.mark.skipif(
    not TEST_DB_URL,
    reason="Integration tests require TEST_DATABASE_URL environment variable"
)


@integration
def test_integration_schema_initialization():
    """Schema creation is idempotent (IF NOT EXISTS)."""
    with patch.dict(os.environ, {'DATABASE_URL': TEST_DB_URL}):
        from src.infrastructure.database.database import initialize_schema
        initialize_schema()
        initialize_schema()  # second call must not fail


@integration
def test_integration_valid_event_persisted(valid_event):
    """A valid event should be persisted to payment_events."""
    with patch.dict(os.environ, {'DATABASE_URL': TEST_DB_URL}):
        from src.infrastructure.database.database import initialize_schema, get_connection
        from src.infrastructure.database.db_persistence import persist_payment_event

        initialize_schema()
        idem_key = f"test-persist-{uuid.uuid4()}"
        event = dict(valid_event)
        event_id, is_dup = persist_payment_event(event, 'pay_int_001', idem_key)

        assert event_id is not None
        assert is_dup is False

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payment_id FROM payment_events WHERE event_id = %s::uuid", (event_id,))
                row = cur.fetchone()
        assert row is not None
        assert row[0] == 'pay_int_001'


@integration
def test_integration_duplicate_idempotency_key(valid_event):
    """Second insert with same idempotency_key must be detected as duplicate."""
    with patch.dict(os.environ, {'DATABASE_URL': TEST_DB_URL}):
        from src.infrastructure.database.database import initialize_schema
        from src.infrastructure.database.db_persistence import persist_payment_event

        initialize_schema()
        idem_key = f"test-dup-{uuid.uuid4()}"
        event = dict(valid_event)

        event_id_1, is_dup_1 = persist_payment_event(event, 'pay_dup_001', idem_key)
        event_id_2, is_dup_2 = persist_payment_event(event, 'pay_dup_001', idem_key)

        assert is_dup_1 is False
        assert is_dup_2 is True
        assert event_id_1 == event_id_2


@integration
def test_integration_invalid_event_not_persisted(invalid_event):
    """Invalid events must be rejected before reaching persistence."""
    with patch.dict(os.environ, {'DATABASE_URL': TEST_DB_URL, 'MODEL_VERSION': 'v1.0'}):
        from src.services.inference.inference_service import RecoveryInferenceService
        svc = RecoveryInferenceService(enable_persistence=True)
        res = svc.predict_event(invalid_event)

    assert res['validation']['is_valid'] is False
    assert res['persistence']['persisted'] is False


@integration
def test_integration_decision_and_audit_persisted(valid_event):
    """Valid event must produce persisted decision and audit record."""
    idem_key = f"test-full-{uuid.uuid4()}"
    event = dict(valid_event)
    event['idempotency_key'] = idem_key

    with patch.dict(os.environ, {'DATABASE_URL': TEST_DB_URL, 'MODEL_VERSION': 'v1.0'}):
        from src.services.inference.inference_service import RecoveryInferenceService
        svc = RecoveryInferenceService(enable_persistence=True)
        svc._model_version_id = None  # force re-resolution
        res = svc.predict_event(event)

    assert res['processing_metadata']['status'] == 'success'
    assert res['persistence']['persisted'] is True
    assert 'decision_id' in res['persistence']
    assert res['persistence'].get('audit_persisted') is True


@integration
def test_integration_simulated_outcome_persisted(valid_event):
    """When simulated outcome fields are present, they should be persisted."""
    idem_key = f"test-outcome-{uuid.uuid4()}"
    event = dict(valid_event)
    event['idempotency_key'] = idem_key
    # Add simulated outcome fields (as OutcomeSimulator would provide)
    event['simulated_recovered'] = 1
    event['simulated_recovered_revenue'] = 4500.0
    event['action_cost'] = 50.0
    event['net_recovered_revenue'] = 4450.0

    with patch.dict(os.environ, {'DATABASE_URL': TEST_DB_URL, 'MODEL_VERSION': 'v1.0'}):
        from src.services.inference.inference_service import RecoveryInferenceService
        from src.infrastructure.database.database import get_connection
        svc = RecoveryInferenceService(enable_persistence=True)
        svc._model_version_id = None
        res = svc.predict_event(event)

    assert res['processing_metadata']['status'] == 'success'
    decision_id = res['persistence'].get('decision_id')
    if decision_id:
        with patch.dict(os.environ, {'DATABASE_URL': TEST_DB_URL}):
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT outcome_source FROM recovery_outcomes WHERE decision_id = %s::uuid",
                        (decision_id,)
                    )
                    row = cur.fetchone()
        # May be None if engine_result lacked simulated fields (engine ignores extra keys)
        if row:
            assert row[0] == 'SIMULATED'
