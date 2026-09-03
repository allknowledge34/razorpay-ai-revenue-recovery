import pytest
import math
from src.services.inference.inference_service import RecoveryInferenceService

@pytest.fixture
def service():
    return RecoveryInferenceService(simulator_cost=50.0, simulator_threshold=0.05)

@pytest.fixture
def valid_base_event():
    return {
        'payment_id': 'pay_111',
        'customer_id': 'cust_222',
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
        'recovery_attempts_so_far': 0
    }

def test_valid_high_probability_event(service, valid_base_event):
    # This event is essentially identical to the 'Technical Error' high prob example
    res = service.predict_event(valid_base_event)
    assert res['validation']['is_valid'] is True
    assert res['processing_metadata']['status'] == 'success'
    assert res['prediction']['recovery_probability'] > 0.65
    assert res['decision']['recommended_action'] == 'Retry Payment'
    assert 'audit_record' in res
    assert res['event_identity']['payment_id'] == 'pay_111'

def test_valid_medium_probability_event(service, valid_base_event):
    event = valid_base_event.copy()
    event.update({
        'payment_amount': 2500.0,
        'failure_reason': 'insufficient_funds',
        'payment_method': 'debit_card',
        'is_subscription': 1,
        'customer_tenure_months': 12,
        'past_successful_payments': 3,
        'past_failed_payments': 2,
        'historical_success_rate': 0.6,
        'time_since_last_success_days': 30,
        'days_overdue': 3,
        'recovery_attempts_so_far': 1
    })
    res = service.predict_event(event)
    assert res['validation']['is_valid'] is True
    assert res['processing_metadata']['status'] == 'success'
    assert res['decision']['recommended_action'] == 'Payment Method Reminder'

def test_valid_low_probability_event(service, valid_base_event):
    event = valid_base_event.copy()
    event.update({
        'payment_amount': 8000.0,
        'failure_reason': 'invalid_card',
        'payment_method': 'credit_card',
        'is_subscription': 0,
        'customer_tenure_months': 2,
        'past_successful_payments': 0,
        'past_failed_payments': 4,
        'historical_success_rate': 0.0,
        'time_since_last_success_days': 999,
        'days_overdue': 15,
        'recovery_attempts_so_far': 3
    })
    res = service.predict_event(event)
    assert res['validation']['is_valid'] is True
    assert res['processing_metadata']['status'] == 'success'
    assert res['decision']['recommended_action'] == 'Manual Review / Stop Automatic Retry'

def test_missing_required_field(service, valid_base_event):
    event = valid_base_event.copy()
    del event['payment_amount']
    res = service.predict_event(event)
    assert res['validation']['is_valid'] is False
    assert res['processing_metadata']['status'] == 'error'
    assert 'Missing required feature: payment_amount' in res['validation']['errors']

def test_invalid_categorical_value(service, valid_base_event):
    event = valid_base_event.copy()
    event['failure_reason'] = 'stolen_card_unsupported'
    res = service.predict_event(event)
    assert res['validation']['is_valid'] is False
    assert any('Unsupported failure_reason' in err for err in res['validation']['errors'])

def test_negative_payment_amount(service, valid_base_event):
    event = valid_base_event.copy()
    event['payment_amount'] = -100
    res = service.predict_event(event)
    assert res['validation']['is_valid'] is False
    assert any('must be greater than 0' in err for err in res['validation']['errors'])

def test_invalid_historical_success_rate(service, valid_base_event):
    event = valid_base_event.copy()
    event['historical_success_rate'] = 1.5
    res = service.predict_event(event)
    assert res['validation']['is_valid'] is False
    assert any('must be between 0 and 1' in err for err in res['validation']['errors'])

def test_invalid_is_subscription(service, valid_base_event):
    event = valid_base_event.copy()
    event['is_subscription'] = 2
    res = service.predict_event(event)
    assert res['validation']['is_valid'] is False
    assert any('is_subscription must be 0 or 1' in err for err in res['validation']['errors'])

def test_nan_input(service, valid_base_event):
    event = valid_base_event.copy()
    event['payment_amount'] = float('nan')
    res = service.predict_event(event)
    assert res['validation']['is_valid'] is False
    assert any('must be finite' in err for err in res['validation']['errors'])

def test_output_contains_probability(service, valid_base_event):
    res = service.predict_event(valid_base_event)
    assert 'recovery_probability' in res['prediction']
    assert isinstance(res['prediction']['recovery_probability'], float)

def test_output_contains_recommended_action(service, valid_base_event):
    res = service.predict_event(valid_base_event)
    assert 'recommended_action' in res['decision']
    assert res['decision']['recommended_action'] in ['Retry Payment', 'Payment Method Reminder', 'Manual Review / Stop Automatic Retry', 'Do Nothing']

def test_output_contains_decision_trace(service, valid_base_event):
    res = service.predict_event(valid_base_event)
    assert 'decision_reason' in res['explanation']
    assert len(res['explanation']['decision_reason']) > 0

def test_payment_id_preserved_when_supplied(service, valid_base_event):
    res = service.predict_event(valid_base_event)
    assert res['event_identity']['payment_id'] == 'pay_111'


def test_missing_payment_id_generates_uuid(service, valid_base_event):
    event = valid_base_event.copy()
    del event['payment_id']
    res = service.predict_event(event)
    assert 'payment_id' in res['event_identity']
    # Very basic check that it looks like a uuid (string of at least 32 chars)
    assert len(str(res['event_identity']['payment_id'])) >= 32

def test_pipeline_exception_returns_safe_message(service, valid_base_event, monkeypatch):
    # Force an exception inside predict_recovery
    def mock_predict_recovery(event):
        raise ValueError("Secret database password is password123")
    
    monkeypatch.setattr(service.engine, "predict_recovery", mock_predict_recovery)
    
    res = service.predict_event(valid_base_event)
    assert res['processing_metadata']['status'] == 'error'
    assert res['processing_metadata']['error_type'] == 'pipeline_exception'
    assert res['processing_metadata']['error_message'] == 'Inference pipeline failed. Please retry or inspect server logs.'
    assert "Secret database password" not in res['processing_metadata']['error_message']
