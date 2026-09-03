from typing import Dict, Any, Tuple
import dataclasses

@dataclasses.dataclass
class RecoveryPolicyConfig:
    min_retry_probability: float = 0.05
    max_recovery_attempts: int = 2
    high_value_threshold: float = 25000.0
    effective_retry_cost: float = 50.0

class BoundedRecoveryPolicy:
    """
    Deterministic bounded recovery policy.
    Evaluates business rules against ML predictions and payment context to allow or block automatic actions.
    """
    def __init__(self, config: RecoveryPolicyConfig = None):
        self.config = config or RecoveryPolicyConfig()

    def evaluate(self, 
                 probability: float, 
                 attempts: int, 
                 amount: float, 
                 expected_recovery: float, 
                 recommended_action: str) -> Tuple[str, str, str]:
        """
        Evaluates the recovery policy.
        
        Returns:
            Tuple of (policy_decision, final_action, decision_reason)
            policy_decision in ['ALLOWED', 'BLOCKED', 'MANUAL_REVIEW']
        """
        if attempts >= self.config.max_recovery_attempts:
            return "BLOCKED", "Stop Automatic Recovery", f"Maximum automatic recovery attempts reached ({self.config.max_recovery_attempts})."
            
        if amount > self.config.high_value_threshold:
            return "MANUAL_REVIEW", "Manual Review", f"Payment amount (₹{amount:.2f}) exceeds high-value threshold (₹{self.config.high_value_threshold:.2f})."
            
        if probability < self.config.min_retry_probability:
            return "BLOCKED", "Stop Automatic Recovery", f"Recovery probability ({probability:.3f}) is below minimum threshold ({self.config.min_retry_probability})."
            
        if expected_recovery <= self.config.effective_retry_cost:
            return "BLOCKED", "Stop Automatic Recovery", f"Expected recovery (₹{expected_recovery:.2f}) does not exceed effective retry cost (₹{self.config.effective_retry_cost:.2f})."
            
        return "ALLOWED", recommended_action, "Policy checks passed. Automatic recovery allowed."

