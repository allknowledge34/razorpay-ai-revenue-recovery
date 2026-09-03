class DecisionTracer:
    def __init__(self, simulator_cost=50.0, simulator_threshold=0.05):
        """
        DecisionTracer explains the model's recommendation and provides an economic
        justification using simulation cost assumptions.
        """
        self.simulator_cost = simulator_cost
        self.simulator_threshold = simulator_threshold

    def generate_trace(self, record, engine_result):
        prob = engine_result['recovery_probability']
        amount = record['payment_amount']
        expected_rec = engine_result['expected_recovery']
        
        # Rule-based action logic from the engine
        action = engine_result['recommended_action']
        
        expected_retry_net_value = expected_rec - self.simulator_cost

        # Generating Explanation
        explanation = []
        explanation.append("The model estimates the probability of recovery based on payment characteristics.")
        
        if action == "Retry Payment":
            if expected_retry_net_value > 0:
                explanation.append(f"Under the simulation assumptions, the expected recovery (₹{expected_rec:,.2f}) exceeds the effective retry cost (₹{self.simulator_cost:,.2f}), so an automated retry is economically justified.")
            else:
                explanation.append(f"Under the simulation assumptions, the expected recovery (₹{expected_rec:,.2f}) is lower than the effective retry cost (₹{self.simulator_cost:,.2f}), meaning an automated retry is not mathematically optimal despite the priority band.")
        elif action == "Payment Method Reminder":
            explanation.append("The Stage 5 rule-based engine places this payment in the reminder band. The simulator threshold is a separate strategy-analysis control.")
            explanation.append(f"(Hypothetically, an automated direct retry would yield a net expected value of ₹{expected_retry_net_value:,.2f} under simulation assumptions).")
        else:
            explanation.append("The Stage 5 rule-based engine places this payment below the automatic-retry band, so automatic retry is stopped. The model estimates very low recovery probability.")

        # Key Input Factors
        factors = [
            f"Payment amount: ₹{amount:,.2f}",
            f"Historical success rate: {record.get('historical_success_rate', 0)*100:.0f}%",
            f"Past successful payments: {record.get('past_successful_payments', 0)}",
            f"Past failed payments: {record.get('past_failed_payments', 0)}",
            f"Failure reason: {record.get('failure_reason', 'unknown')}",
            f"Recovery attempts so far: {record.get('recovery_attempts_so_far', 0)}",
            f"Days overdue: {record.get('days_overdue', 0)}"
        ]

        return {
            'recovery_probability': prob,
            'payment_amount': amount,
            'expected_recovery': expected_rec,
            'selected_threshold': self.simulator_threshold,
            'effective_retry_cost': self.simulator_cost,
            'expected_retry_net_value': expected_retry_net_value,
            'recommended_action': action,
            'decision_reason': " ".join(explanation),
            'key_input_factors': factors
        }

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.domain.recovery_engine import RecoveryEngine
    
    engine = RecoveryEngine()
    tracer = DecisionTracer(simulator_cost=50.0, simulator_threshold=0.05)
    
    # Example records
    record_a = {'payment_amount': 5000, 'failure_reason': 'technical_error', 'payment_method': 'upi', 'is_subscription': 0, 'customer_tenure_months': 24, 'past_successful_payments': 10, 'past_failed_payments': 0, 'historical_success_rate': 1.0, 'time_since_last_success_days': 5, 'days_overdue': 1, 'recovery_attempts_so_far': 0}
    record_b = {'payment_amount': 2500, 'failure_reason': 'insufficient_funds', 'payment_method': 'debit_card', 'is_subscription': 1, 'customer_tenure_months': 12, 'past_successful_payments': 3, 'past_failed_payments': 2, 'historical_success_rate': 0.6, 'time_since_last_success_days': 30, 'days_overdue': 3, 'recovery_attempts_so_far': 1}
    record_c = {'payment_amount': 8000, 'failure_reason': 'invalid_card', 'payment_method': 'credit_card', 'is_subscription': 0, 'customer_tenure_months': 2, 'past_successful_payments': 0, 'past_failed_payments': 4, 'historical_success_rate': 0.0, 'time_since_last_success_days': 999, 'days_overdue': 15, 'recovery_attempts_so_far': 3}

    for name, rec in [("A", record_a), ("B", record_b), ("C", record_c)]:
        res = engine.predict_recovery(rec)
        trace = tracer.generate_trace(rec, res)
        print(f"\n--- Case {name} ---")
        print(f"Prob: {trace['recovery_probability']:.3f}, Expected: {trace['expected_recovery']:.2f}, Action: {trace['recommended_action']}")
        print(f"Threshold: {trace['selected_threshold']:.2f}, Retry Cost: {trace['effective_retry_cost']:.2f}, Hypothetical Net Retry Value: {trace['expected_retry_net_value']:.2f}")
        print(f"Reason: {trace['decision_reason']}")
