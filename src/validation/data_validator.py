from dataclasses import dataclass
from typing import List, Dict, Any
import math

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]

class DataValidator:
    SUPPORTED_FAILURE_REASONS = {'insufficient_funds', 'invalid_card', 'technical_error', 'limit_exceeded'}
    SUPPORTED_PAYMENT_METHODS = {'credit_card', 'debit_card', 'upi', 'bank_transfer'}

    REQUIRED_FEATURES = [
        'payment_amount',
        'failure_reason',
        'payment_method',
        'is_subscription',
        'customer_tenure_months',
        'past_successful_payments',
        'past_failed_payments',
        'historical_success_rate',
        'time_since_last_success_days',
        'days_overdue',
        'recovery_attempts_so_far'
    ]

    @staticmethod
    def _is_finite_numeric(value, field_name: str, errors: List[str]) -> bool:
        try:
            val = float(value)
            if math.isnan(val) or math.isinf(val):
                errors.append(f"{field_name} must be finite")
                return False
            return True
        except (ValueError, TypeError):
            errors.append(f"{field_name} must be numeric")
            return False

    @staticmethod
    def _is_integer_like(value, field_name: str, errors: List[str]) -> bool:
        if not DataValidator._is_finite_numeric(value, field_name, errors):
            return False
        val = float(value)
        if not val.is_integer():
            errors.append(f"{field_name} must be an integer")
            return False
        return True

    @classmethod
    def validate_record(cls, record: Dict[str, Any]) -> ValidationResult:
        errors = []
        
        for feature in cls.REQUIRED_FEATURES:
            if feature not in record:
                errors.append(f"Missing required feature: {feature}")
            elif record[feature] is None:
                errors.append(f"{feature} cannot be null")
                
        if errors:
            return ValidationResult(is_valid=False, errors=errors)

        if cls._is_finite_numeric(record['payment_amount'], 'payment_amount', errors):
            amount = float(record['payment_amount'])
            if amount <= 0:
                errors.append("payment_amount must be greater than 0")

        if cls._is_finite_numeric(record['customer_tenure_months'], 'customer_tenure_months', errors):
            tenure = float(record['customer_tenure_months'])
            if tenure < 0:
                errors.append("customer_tenure_months must be >= 0")

        if cls._is_integer_like(record['past_successful_payments'], 'past_successful_payments', errors):
            successes = float(record['past_successful_payments'])
            if successes < 0:
                errors.append("past_successful_payments must be >= 0")

        if cls._is_integer_like(record['past_failed_payments'], 'past_failed_payments', errors):
            fails = float(record['past_failed_payments'])
            if fails < 0:
                errors.append("past_failed_payments must be >= 0")

        if cls._is_finite_numeric(record['historical_success_rate'], 'historical_success_rate', errors):
            rate = float(record['historical_success_rate'])
            if not (0 <= rate <= 1):
                errors.append("historical_success_rate must be between 0 and 1")

        if cls._is_finite_numeric(record['time_since_last_success_days'], 'time_since_last_success_days', errors):
            time_since = float(record['time_since_last_success_days'])
            if time_since < 0:
                errors.append("time_since_last_success_days must be >= 0")

        if cls._is_finite_numeric(record['days_overdue'], 'days_overdue', errors):
            overdue = float(record['days_overdue'])
            if overdue < 0:
                errors.append("days_overdue must be >= 0")

        if cls._is_integer_like(record['recovery_attempts_so_far'], 'recovery_attempts_so_far', errors):
            attempts = float(record['recovery_attempts_so_far'])
            if attempts < 0:
                errors.append("recovery_attempts_so_far must be >= 0")
                
        if cls._is_integer_like(record['is_subscription'], 'is_subscription', errors):
            is_sub = float(record['is_subscription'])
            if is_sub not in [0.0, 1.0]:
                errors.append("is_subscription must be 0 or 1")

        if record['failure_reason'] not in cls.SUPPORTED_FAILURE_REASONS:
            errors.append(f"Unsupported failure_reason: {record['failure_reason']}")

        if record['payment_method'] not in cls.SUPPORTED_PAYMENT_METHODS:
            errors.append(f"Unsupported payment_method: {record['payment_method']}")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    @staticmethod
    def validate_prediction_probability(prob: float) -> ValidationResult:
        errors = []
        try:
            p = float(prob)
            if math.isnan(p) or math.isinf(p):
                errors.append("prediction probability must be finite")
            elif not (0 <= p <= 1):
                errors.append("prediction probability must be between 0 and 1")
        except (ValueError, TypeError):
            errors.append("prediction probability must be numeric")
            
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
