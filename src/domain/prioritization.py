import math
import pandas as pd
from typing import Dict, Any

class RecoveryPrioritizer:
    """
    Independent deterministic Recovery Prioritization layer.
    Calculates business-adjusted expected recovery to sequence actions,
    but DOES NOT override RecoveryPolicy or RecoveryOrchestrator safety boundaries.
    """
    def __init__(self, critical_threshold=10000.0, high_threshold=2500.0, medium_threshold=500.0, subscription_multiplier=1.5):
        self.critical_threshold = critical_threshold
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.subscription_multiplier = subscription_multiplier

    def _is_valid(self, prob: Any, amount: Any, is_sub: Any) -> bool:
        try:
            p = float(prob)
            a = float(amount)
        except (ValueError, TypeError):
            return False
            
        if math.isnan(p) or math.isnan(a):
            return False
        if math.isinf(p) or math.isinf(a):
            return False
        if p < 0.0 or p > 1.0:
            return False
        if a <= 0.0:
            return False
            
        # Strict validation for is_subscription
        if not isinstance(is_sub, (int, bool)):
            return False
        if isinstance(is_sub, bool):
            s = 1 if is_sub else 0
        else:
            s = is_sub
            
        if s not in (0, 1):
            return False
            
        return True

    def calculate_priority(self, payment_id: str, amount: float, prob: float, is_sub: int) -> Dict[str, Any]:
        """Calculates deterministic business priority value."""
        if not self._is_valid(prob, amount, is_sub):
            safe_amt = amount if isinstance(amount, (int, float)) and not math.isnan(amount) else 0.0
            safe_prob = prob if isinstance(prob, (int, float)) and not math.isnan(prob) else 0.0
            return {
                'payment_id': payment_id,
                'payment_amount': float(safe_amt),
                'recovery_probability': float(safe_prob),
                'is_subscription': 0,
                'base_expected_recovery': 0.0,
                'subscription_multiplier': 1.0,
                'business_adjusted_expected_recovery': 0.0,
                'priority_value': 0.0,
                'priority_tier': 'LOW',
                'priority_explanation': 'Invalid input parameters; defaulting to safe LOW priority.'
            }
            
        p = float(prob)
        a = float(amount)
        s = 1 if is_sub is True else (0 if is_sub is False else int(is_sub))
        
        sub_mult = self.subscription_multiplier if s == 1 else 1.0
        base_expected = a * p
        business_adj = base_expected * sub_mult
        
        tier = 'LOW'
        if business_adj >= self.critical_threshold:
            tier = 'CRITICAL'
        elif business_adj >= self.high_threshold:
            tier = 'HIGH'
        elif business_adj >= self.medium_threshold:
            tier = 'MEDIUM'
        
        if s == 1:
            explanation = f"{tier} priority. Subscription recovery receives the configured synthetic business-value multiplier ({sub_mult}x). Business-adjusted recovery value is ₹{business_adj:,.2f}."
        else:
            explanation = f"{tier} priority based on base expected recovery. Business-adjusted recovery value is ₹{business_adj:,.2f}."

        return {
            'payment_id': str(payment_id),
            'payment_amount': a,
            'recovery_probability': p,
            'is_subscription': s,
            'base_expected_recovery': base_expected,
            'subscription_multiplier': sub_mult,
            'business_adjusted_expected_recovery': business_adj,
            'priority_value': business_adj,
            'priority_tier': tier,
            'priority_explanation': explanation
        }
        
    def batch_prioritize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies prioritization to a batch of scored payments."""
        results = []
        for _, row in df.iterrows():
            pid = str(row.get('payment_id', 'unknown'))
            amt = row.get('payment_amount', 0.0)
            prob = row.get('recovery_probability', 0.0)
            is_sub = row.get('is_subscription', 0)
            results.append(self.calculate_priority(pid, amt, prob, is_sub))
            
        out_df = pd.DataFrame(results)
        out_df.sort_values(by=['priority_value', 'payment_id'], ascending=[False, True], inplace=True)
        out_df.reset_index(drop=True, inplace=True)
        out_df['priority_rank'] = out_df.index + 1
        return out_df
