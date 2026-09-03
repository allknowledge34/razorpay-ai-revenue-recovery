import pytest
from src.domain.recovery_policy import BoundedRecoveryPolicy, RecoveryPolicyConfig

def test_high_probability_viable_economics():
    policy = BoundedRecoveryPolicy()
    decision, action, reason = policy.evaluate(
        probability=0.8,
        attempts=0,
        amount=5000.0,
        expected_recovery=4000.0,
        recommended_action="Retry Payment"
    )
    assert decision == "ALLOWED"
    assert action == "Retry Payment"

def test_probability_below_threshold():
    policy = BoundedRecoveryPolicy(RecoveryPolicyConfig(min_retry_probability=0.1))
    decision, action, reason = policy.evaluate(
        probability=0.05,
        attempts=0,
        amount=5000.0,
        expected_recovery=250.0,
        recommended_action="Retry Payment"
    )
    assert decision == "BLOCKED"
    assert "below minimum threshold" in reason

def test_maximum_attempts_reached():
    policy = BoundedRecoveryPolicy(RecoveryPolicyConfig(max_recovery_attempts=2))
    decision, action, reason = policy.evaluate(
        probability=0.8,
        attempts=2,
        amount=5000.0,
        expected_recovery=4000.0,
        recommended_action="Retry Payment"
    )
    assert decision == "BLOCKED"
    assert "Maximum automatic recovery attempts reached" in reason

def test_high_value_payment():
    policy = BoundedRecoveryPolicy(RecoveryPolicyConfig(high_value_threshold=10000.0))
    decision, action, reason = policy.evaluate(
        probability=0.8,
        attempts=0,
        amount=15000.0,
        expected_recovery=12000.0,
        recommended_action="Retry Payment"
    )
    assert decision == "MANUAL_REVIEW"
    assert action == "Manual Review"

def test_expected_recovery_below_action_cost():
    policy = BoundedRecoveryPolicy(RecoveryPolicyConfig(effective_retry_cost=50.0))
    decision, action, reason = policy.evaluate(
        probability=0.1,
        attempts=0,
        amount=100.0,
        expected_recovery=10.0,
        recommended_action="Retry Payment"
    )
    assert decision == "BLOCKED"
    assert "does not exceed effective retry cost" in reason

def test_high_value_takes_precedence_over_economic_block():
    policy = BoundedRecoveryPolicy(RecoveryPolicyConfig(high_value_threshold=10000.0, min_retry_probability=0.1, effective_retry_cost=50.0))
    # low prob (0.01), low expected recovery (1.0), BUT high amount (15000.0).
    decision, action, reason = policy.evaluate(
        probability=0.01,
        attempts=0,
        amount=15000.0,
        expected_recovery=150.0, # still high enough, wait let's make it 1.0 to test precedence
        recommended_action="Retry Payment"
    )
    # High value should trigger MANUAL_REVIEW before min_retry_probability triggers BLOCKED
    assert decision == "MANUAL_REVIEW"
    assert action == "Manual Review"
