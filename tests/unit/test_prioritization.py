import pytest
import pandas as pd
import numpy as np
from src.domain.prioritization import RecoveryPrioritizer

@pytest.fixture
def prioritizer():
    return RecoveryPrioritizer(
        critical_threshold=10000.0,
        high_threshold=2500.0,
        medium_threshold=500.0,
        subscription_multiplier=1.5
    )

def test_probability_times_amount(prioritizer):
    res = prioritizer.calculate_priority("p1", 1000.0, 0.5, 0)
    assert res['base_expected_recovery'] == 500.0
    assert res['priority_value'] == 500.0
    assert res['priority_tier'] == 'MEDIUM'

def test_subscription_multiplier(prioritizer):
    res = prioritizer.calculate_priority("p2", 1000.0, 0.5, 1)
    assert res['base_expected_recovery'] == 500.0
    assert res['subscription_multiplier'] == 1.5
    assert res['priority_value'] == 750.0

def test_non_subscription_multiplier(prioritizer):
    res = prioritizer.calculate_priority("p3", 1000.0, 0.5, 0)
    assert res['subscription_multiplier'] == 1.0

def test_critical_boundary(prioritizer):
    res = prioritizer.calculate_priority("p4", 20000.0, 0.5, 0)
    assert res['priority_value'] == 10000.0
    assert res['priority_tier'] == 'CRITICAL'
    
    res_below = prioritizer.calculate_priority("p5", 19999.0, 0.5, 0)
    assert res_below['priority_tier'] == 'HIGH'

def test_high_boundary(prioritizer):
    res = prioritizer.calculate_priority("p6", 5000.0, 0.5, 0)
    assert res['priority_value'] == 2500.0
    assert res['priority_tier'] == 'HIGH'

def test_medium_boundary(prioritizer):
    res = prioritizer.calculate_priority("p7", 1000.0, 0.5, 0)
    assert res['priority_value'] == 500.0
    assert res['priority_tier'] == 'MEDIUM'

def test_low_boundary(prioritizer):
    res = prioritizer.calculate_priority("p8", 999.0, 0.5, 0)
    assert res['priority_value'] == 499.5
    assert res['priority_tier'] == 'LOW'

def test_invalid_probability(prioritizer):
    res = prioritizer.calculate_priority("p9", 1000.0, 1.5, 0)
    assert res['priority_tier'] == 'LOW'
    assert 'Invalid input' in res['priority_explanation']

def test_invalid_amount(prioritizer):
    res = prioritizer.calculate_priority("p10", -10.0, 0.5, 0)
    assert res['priority_tier'] == 'LOW'

def test_invalid_subscription_flag(prioritizer):
    # Valid
    res_0 = prioritizer.calculate_priority("p11a", 1000.0, 0.5, 0)
    assert res_0['priority_value'] == 500.0
    
    res_1 = prioritizer.calculate_priority("p11b", 1000.0, 0.5, 1)
    assert res_1['priority_value'] == 750.0
    
    res_false = prioritizer.calculate_priority("p11c", 1000.0, 0.5, False)
    assert res_false['priority_value'] == 500.0
    
    res_true = prioritizer.calculate_priority("p11d", 1000.0, 0.5, True)
    assert res_true['priority_value'] == 750.0

    # Invalid
    for invalid_val in [1.5, 2, -1, "1", "0", "True"]:
        res_inv = prioritizer.calculate_priority("inv", 1000.0, 0.5, invalid_val)
        assert res_inv['priority_tier'] == 'LOW'
        assert res_inv['priority_value'] == 0.0
        assert 'Invalid input parameters' in res_inv['priority_explanation']

def test_deterministic_repeated_calculation(prioritizer):
    res1 = prioritizer.calculate_priority("p12", 2000.0, 0.8, 1)
    res2 = prioritizer.calculate_priority("p12", 2000.0, 0.8, 1)
    assert res1 == res2

def test_ranking_order_and_tie_breaking(prioritizer):
    df = pd.DataFrame([
        {'payment_id': 'B', 'payment_amount': 1000.0, 'recovery_probability': 0.5, 'is_subscription': 0}, # EV 500
        {'payment_id': 'C', 'payment_amount': 2000.0, 'recovery_probability': 0.5, 'is_subscription': 0}, # EV 1000
        {'payment_id': 'A', 'payment_amount': 1000.0, 'recovery_probability': 0.5, 'is_subscription': 0}, # EV 500
    ])
    ranked = prioritizer.batch_prioritize(df)
    assert ranked.iloc[0]['payment_id'] == 'C'
    assert ranked.iloc[1]['payment_id'] == 'A' # A before B due to tie-breaker
    assert ranked.iloc[2]['payment_id'] == 'B'

def test_conceptual_case(prioritizer):
    # Payment A: ₹100 x 0.95 = ₹95
    # Payment B: ₹10,000 x 0.40 = ₹4,000
    # B must receive higher business-value priority despite lower probability.
    res_a = prioritizer.calculate_priority("A", 100.0, 0.95, 0)
    res_b = prioritizer.calculate_priority("B", 10000.0, 0.40, 0)
    assert res_b['priority_value'] > res_a['priority_value']
    assert res_b['priority_tier'] == 'HIGH'
    assert res_a['priority_tier'] == 'LOW'

def test_no_double_counting_documentation():
    # Explicitly test that prioritization does not manually reapply model features 
    # such as attempts, overdue days, historical success rate.
    # Those remain inputs to the ML probability model.
    # This test acts as a structural validation of the formula parameters.
    import inspect
    sig = inspect.signature(RecoveryPrioritizer.calculate_priority)
    params = list(sig.parameters.keys())
    assert 'historical_success_rate' not in params
    assert 'recovery_attempts_so_far' not in params
    assert 'days_overdue' not in params
    assert 'failure_reason' not in params
