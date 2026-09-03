import pytest
from src.services.recovery.recovery_orchestrator import RecoveryOrchestrator, RecoveryStateMachine

def test_successful_retry():
    orchestrator = RecoveryOrchestrator()
    context = {'payment_id': 'test_succ', 'recovery_probability': 0.99, 'recovery_attempts_so_far': 0, 'payment_amount': 5000.0, 'expected_recovery': 4950.0, 'recommended_action': 'Retry Payment'}
    res = orchestrator.process_event(context, seed=42)
    assert res['policy_decision'] == 'ALLOWED'
    assert res['simulated_recovered'] is True
    assert res['verification_status'] == 'VERIFIED'
    assert res['final_state'] == 'CLOSED'

def test_failed_retry():
    orchestrator = RecoveryOrchestrator()
    context = {'payment_id': 'test_fail', 'recovery_probability': 0.1, 'recovery_attempts_so_far': 0, 'payment_amount': 5000.0, 'expected_recovery': 500.0, 'recommended_action': 'Retry Payment'}
    res = orchestrator.process_event(context, seed=1) # seed 1 -> rng.random() ~ 0.13 > 0.1, so fails
    assert res['simulated_recovered'] is False
    assert res['final_state'] == 'STOPPED'

def test_payment_method_reminder():
    orchestrator = RecoveryOrchestrator()
    context = {'payment_id': 'test_remind', 'recovery_probability': 0.99, 'recovery_attempts_so_far': 0, 'payment_amount': 5000.0, 'expected_recovery': 4950.0, 'recommended_action': 'Payment Method Reminder'}
    res = orchestrator.process_event(context, seed=42)
    assert res['selected_action'] == 'Payment Method Reminder'
    assert res['action_cost'] == 1.0

def test_manual_review():
    orchestrator = RecoveryOrchestrator()
    context = {'payment_id': 'test_manual', 'recovery_probability': 0.9, 'recovery_attempts_so_far': 0, 'payment_amount': 50000.0, 'expected_recovery': 45000.0, 'recommended_action': 'Retry Payment'}
    res = orchestrator.process_event(context, seed=42)
    assert res['policy_decision'] == 'MANUAL_REVIEW'
    assert res['final_state'] == 'MANUAL_REVIEW'

def test_probability_below_threshold():
    orchestrator = RecoveryOrchestrator()
    context = {'payment_id': 'test_prob', 'recovery_probability': 0.01, 'recovery_attempts_so_far': 0, 'payment_amount': 500.0, 'expected_recovery': 5.0, 'recommended_action': 'Retry Payment'}
    res = orchestrator.process_event(context, seed=42)
    assert res['policy_decision'] == 'BLOCKED'
    assert res['final_state'] == 'STOPPED'

def test_maximum_attempts_reached():
    orchestrator = RecoveryOrchestrator()
    context = {'payment_id': 'test_max', 'recovery_probability': 0.9, 'recovery_attempts_so_far': 2, 'payment_amount': 5000.0, 'expected_recovery': 4500.0, 'recommended_action': 'Retry Payment'}
    res = orchestrator.process_event(context, seed=42)
    assert res['policy_decision'] == 'BLOCKED'
    assert res['final_state'] == 'STOPPED'

def test_expected_recovery_below_cost():
    orchestrator = RecoveryOrchestrator()
    context = {'payment_id': 'test_cost', 'recovery_probability': 0.1, 'recovery_attempts_so_far': 0, 'payment_amount': 100.0, 'expected_recovery': 10.0, 'recommended_action': 'Retry Payment'}
    res = orchestrator.process_event(context, seed=42)
    assert res['policy_decision'] == 'BLOCKED'
    assert res['final_state'] == 'STOPPED'

def test_invalid_state_transition():
    sm = RecoveryStateMachine()
    with pytest.raises(ValueError):
        sm.transition('CLOSED')

def test_deterministic_seed():
    orchestrator1 = RecoveryOrchestrator()
    orchestrator2 = RecoveryOrchestrator()
    context = {'payment_id': 'test_seed', 'recovery_probability': 0.5, 'recovery_attempts_so_far': 0, 'payment_amount': 5000.0, 'expected_recovery': 2500.0, 'recommended_action': 'Retry Payment'}
    res1 = orchestrator1.process_event(context, seed=123)
    res2 = orchestrator2.process_event(context, seed=123)
    assert res1 == res2

def test_different_seeds():
    orchestrator = RecoveryOrchestrator()
    context = {'payment_id': 'test_diff', 'recovery_probability': 0.5, 'recovery_attempts_so_far': 0, 'payment_amount': 5000.0, 'expected_recovery': 2500.0, 'recommended_action': 'Retry Payment'}
    # 0.5 probability should be deterministic based on seed.
    res1 = orchestrator.process_event(context, seed=42) # random.Random(42).random() = 0.639
    res2 = orchestrator.process_event(context, seed=7)  # random.Random(7).random() = 0.323
    assert res1['simulated_recovered'] != res2['simulated_recovered']

def test_attempt_number_calculation():
    orchestrator = RecoveryOrchestrator()
    context = {'payment_id': 'test_att', 'recovery_probability': 0.9, 'recovery_attempts_so_far': 0, 'payment_amount': 5000.0, 'expected_recovery': 4500.0, 'recommended_action': 'Retry Payment'}
    res = orchestrator.process_event(context, seed=42)
    assert res['attempt_number'] == 1 # Allowed -> incremented
    
    context2 = {'payment_id': 'test_att2', 'recovery_probability': 0.9, 'recovery_attempts_so_far': 2, 'payment_amount': 5000.0, 'expected_recovery': 4500.0, 'recommended_action': 'Retry Payment'}
    res2 = orchestrator.process_event(context2, seed=42)
    assert res2['attempt_number'] == 2 # Blocked -> not incremented

def test_stop_automatic_recovery_workflow():
    orchestrator = RecoveryOrchestrator()
    context = {'payment_id': 'test_stop', 'recovery_probability': 0.01, 'recovery_attempts_so_far': 0, 'payment_amount': 5000.0, 'expected_recovery': 50.0, 'recommended_action': 'Retry Payment'}
    res = orchestrator.process_event(context, seed=42)
    assert res['policy_decision'] == 'BLOCKED'
    assert res['selected_action'] == 'Stop Automatic Recovery'
    assert res['final_state'] == 'STOPPED'
