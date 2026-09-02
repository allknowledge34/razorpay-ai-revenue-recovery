import pytest
import math
from src.data_validator import DataValidator

def get_valid_record():
    return {
        'payment_amount': 5000.0,
        'failure_reason': 'technical_error',
        'payment_method': 'upi',
        'is_subscription': 0,
        'customer_tenure_months': 12,
        'past_successful_payments': 5,
        'past_failed_payments': 1,
        'historical_success_rate': 0.8,
        'time_since_last_success_days': 15,
        'days_overdue': 2,
        'recovery_attempts_so_far': 0
    }

def test_valid_record():
    record = get_valid_record()
    res = DataValidator.validate_record(record)
    assert res.is_valid is True
    assert len(res.errors) == 0

def test_missing_feature():
    record = get_valid_record()
    del record['payment_amount']
    res = DataValidator.validate_record(record)
    assert res.is_valid is False
    assert "Missing required feature: payment_amount" in res.errors

def test_payment_amount_zero():
    record = get_valid_record()
    record['payment_amount'] = 0
    res = DataValidator.validate_record(record)
    assert res.is_valid is False
    assert "payment_amount must be greater than 0" in res.errors

def test_payment_amount_negative():
    record = get_valid_record()
    record['payment_amount'] = -100
    res = DataValidator.validate_record(record)
    assert res.is_valid is False
    assert "payment_amount must be greater than 0" in res.errors

def test_payment_amount_nan():
    record = get_valid_record()
    record['payment_amount'] = float('nan')
    res = DataValidator.validate_record(record)
    assert res.is_valid is False
    assert "payment_amount must be finite" in res.errors

def test_payment_amount_inf():
    record = get_valid_record()
    record['payment_amount'] = float('inf')
    res = DataValidator.validate_record(record)
    assert res.is_valid is False
    assert "payment_amount must be finite" in res.errors

def test_invalid_historical_success_rate_high():
    record = get_valid_record()
    record['historical_success_rate'] = 1.5
    res = DataValidator.validate_record(record)
    assert res.is_valid is False
    assert "historical_success_rate must be between 0 and 1" in res.errors

def test_invalid_historical_success_rate_low():
    record = get_valid_record()
    record['historical_success_rate'] = -0.5
    res = DataValidator.validate_record(record)
    assert res.is_valid is False
    assert "historical_success_rate must be between 0 and 1" in res.errors
    
def test_historical_success_rate_nan():
    record = get_valid_record()
    record['historical_success_rate'] = float('nan')
    res = DataValidator.validate_record(record)
    assert res.is_valid is False
    assert "historical_success_rate must be finite" in res.errors

def test_negative_days_overdue():
    record = get_valid_record()
    record['days_overdue'] = -2
    res = DataValidator.validate_record(record)
    assert res.is_valid is False
    assert "days_overdue must be >= 0" in res.errors
    
def test_days_overdue_inf():
    record = get_valid_record()
    record['days_overdue'] = float('inf')
    res = DataValidator.validate_record(record)
    assert res.is_valid is False
    assert "days_overdue must be finite" in res.errors

def test_negative_recovery_attempts():
    record = get_valid_record()
    record['recovery_attempts_so_far'] = -1
    res = DataValidator.validate_record(record)
    assert res.is_valid is False
    assert "recovery_attempts_so_far must be >= 0" in res.errors

def test_unsupported_failure_reason():
    record = get_valid_record()
    record['failure_reason'] = 'stolen_card'
    res = DataValidator.validate_record(record)
    assert res.is_valid is False
    assert "Unsupported failure_reason: stolen_card" in res.errors

def test_unsupported_payment_method():
    record = get_valid_record()
    record['payment_method'] = 'crypto'
    res = DataValidator.validate_record(record)
    assert res.is_valid is False
    assert "Unsupported payment_method: crypto" in res.errors

def test_malformed_numeric():
    record = get_valid_record()
    record['payment_amount'] = 'five_thousand'
    res = DataValidator.validate_record(record)
    assert res.is_valid is False
    assert "payment_amount must be numeric" in res.errors

def test_probability_validation():
    res = DataValidator.validate_prediction_probability(0.5)
    assert res.is_valid is True

    res = DataValidator.validate_prediction_probability(-0.1)
    assert res.is_valid is False

    res = DataValidator.validate_prediction_probability(1.1)
    assert res.is_valid is False

    res = DataValidator.validate_prediction_probability(float('nan'))
    assert res.is_valid is False

def test_invalid_is_subscription_value_2():
    record = get_valid_record()
    record['is_subscription'] = 2
    res = DataValidator.validate_record(record)
    assert res.is_valid is False
    assert "is_subscription must be 0 or 1" in res.errors

def test_invalid_is_subscription_value_fractional():
    record = get_valid_record()
    record['is_subscription'] = 0.5
    res = DataValidator.validate_record(record)
    assert res.is_valid is False
    assert "is_subscription must be an integer" in res.errors

def test_invalid_is_subscription_value_string():
    record = get_valid_record()
    record['is_subscription'] = "yes"
    res = DataValidator.validate_record(record)
    assert res.is_valid is False
    assert "is_subscription must be numeric" in res.errors

def test_fractional_past_successful_payments():
    record = get_valid_record()
    record['past_successful_payments'] = 5.5
    res = DataValidator.validate_record(record)
    assert res.is_valid is False
    assert "past_successful_payments must be an integer" in res.errors

def test_fractional_past_failed_payments():
    record = get_valid_record()
    record['past_failed_payments'] = 1.2
    res = DataValidator.validate_record(record)
    assert res.is_valid is False
    assert "past_failed_payments must be an integer" in res.errors

def test_fractional_recovery_attempts_so_far():
    record = get_valid_record()
    record['recovery_attempts_so_far'] = 2.7
    res = DataValidator.validate_record(record)
    assert res.is_valid is False
    assert "recovery_attempts_so_far must be an integer" in res.errors
