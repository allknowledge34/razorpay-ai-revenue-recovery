from typing import Dict, Any, List
import random
import datetime

from src.domain.recovery_policy import BoundedRecoveryPolicy, RecoveryPolicyConfig

class RecoveryStateMachine:
    VALID_STATES = {
        'FAILED',
        'ASSESSED',
        'ACTION_SELECTED',
        'ACTION_EXECUTED',
        'RECOVERED',
        'FAILED_RECOVERY',
        'VERIFIED',
        'CLOSED',
        'STOPPED',
        'MANUAL_REVIEW'
    }
    
    VALID_TRANSITIONS = {
        'FAILED': ['ASSESSED'],
        'ASSESSED': ['ACTION_SELECTED', 'STOPPED', 'MANUAL_REVIEW'],
        'ACTION_SELECTED': ['ACTION_EXECUTED'],
        'ACTION_EXECUTED': ['RECOVERED', 'FAILED_RECOVERY'],
        'RECOVERED': ['VERIFIED'],
        'FAILED_RECOVERY': ['VERIFIED'],
        'VERIFIED': ['CLOSED', 'STOPPED'],
        'CLOSED': [],
        'STOPPED': [],
        'MANUAL_REVIEW': []
    }

    def __init__(self, initial_state: str = 'FAILED'):
        self.state = initial_state
        self.history = [initial_state]

    def transition(self, to_state: str):
        if to_state not in self.VALID_STATES:
            raise ValueError(f"Invalid state: {to_state}")
        if to_state not in self.VALID_Transitions(self.state):
            raise ValueError(f"Invalid state transition from {self.state} to {to_state}")
        self.state = to_state
        self.history.append(to_state)
        
    def VALID_Transitions(self, state: str) -> List[str]:
        return self.VALID_TRANSITIONS.get(state, [])


class RecoveryOrchestrator:
    """
    Coordinates policy evaluation, synthetic action execution, and outcome generation/verification.
    """
    def __init__(self, policy: BoundedRecoveryPolicy = None):
        self.policy = policy or BoundedRecoveryPolicy()
        
    def execute_simulated_action(self, action: str, probability: float, amount: float, seed: int = None) -> Dict[str, Any]:
        """
        Executes a synthetic action based on economic assumptions.
        """
        rng = random.Random(seed) if seed is not None else random.Random()
            
        assumptions = {
            'Retry Payment': {'cost': 50.0, 'multiplier': 1.0},
            'Payment Method Reminder': {'cost': 1.0, 'multiplier': 0.5},
            'Manual Review': {'cost': 100.0, 'multiplier': 0.75},
            'Stop Automatic Recovery': {'cost': 0.0, 'multiplier': 0.0},
            'Unknown': {'cost': 0.0, 'multiplier': 0.0}
        }
        
        act = assumptions.get(action, assumptions['Unknown'])
        action_cost = act['cost']
        adj_prob = probability * act['multiplier']
        
        simulated_recovered = rng.random() < adj_prob
        recovered_amount = amount if simulated_recovered else 0.0
        net_recovered_revenue = recovered_amount - action_cost
        
        return {
            'simulated_recovered': simulated_recovered,
            'action_cost': action_cost,
            'recovered_amount': recovered_amount,
            'net_recovered_revenue': net_recovered_revenue
        }

    def process_event(self, event_context: Dict[str, Any], seed: int = None) -> Dict[str, Any]:
        """
        Orchestrates the bounded recovery workflow.
        """
        sm = RecoveryStateMachine('FAILED')
        
        # 1. ASSESSED
        sm.transition('ASSESSED')
        
        probability = event_context.get('recovery_probability', 0.0)
        attempts = event_context.get('recovery_attempts_so_far', 0)
        amount = event_context.get('payment_amount', 0.0)
        expected_recovery = event_context.get('expected_recovery', 0.0)
        recommended_action = event_context.get('recommended_action', 'Unknown')
        
        # 2. Evaluate Policy
        policy_decision, final_action, decision_reason = self.policy.evaluate(
            probability, attempts, amount, expected_recovery, recommended_action
        )
        
        result = {
            'payment_id': event_context.get('payment_id'),
            'initial_state': 'FAILED',
            'attempt_number': attempts + 1 if policy_decision == 'ALLOWED' else attempts,
            'policy_decision': policy_decision,
            'selected_action': final_action,
            'decision_reason': decision_reason,
            'simulated_recovered': False,
            'action_cost': 0.0,
            'recovered_amount': 0.0,
            'net_recovered_revenue': 0.0,
            'verification_status': 'PENDING'
        }
        
        if policy_decision == 'BLOCKED':
            sm.transition('STOPPED')
            result['final_state'] = sm.state
            result['state_history'] = sm.history
            return result
        elif policy_decision == 'MANUAL_REVIEW':
            sm.transition('MANUAL_REVIEW')
            result['final_state'] = sm.state
            result['state_history'] = sm.history
            return result
            
        sm.transition('ACTION_SELECTED')
        
        # 3. Simulate Execution
        sm.transition('ACTION_EXECUTED')
        sim_result = self.execute_simulated_action(final_action, probability, amount, seed)
        result.update(sim_result)
        
        if sim_result['simulated_recovered']:
            sm.transition('RECOVERED')
        else:
            sm.transition('FAILED_RECOVERY')
            
        # 4. Verify
        sm.transition('VERIFIED')
        result['verification_status'] = 'VERIFIED'
        
        if sim_result['simulated_recovered']:
            sm.transition('CLOSED')
        else:
            sm.transition('STOPPED')
            
        result['final_state'] = sm.state
        result['state_history'] = sm.history
        
        return result

