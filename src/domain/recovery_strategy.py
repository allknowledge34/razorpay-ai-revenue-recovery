import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

class RecoverySimulator:
    def __init__(self, data_path='data/failed_payments_scored.csv'):
        # We use the already scored dataset to ensure no ML pipeline drift
        self.df = pd.read_csv(data_path)
        
        # Base Simulation Assumptions (Cost Model)
        # Note: These are configurable simulation assumptions, NOT actual Razorpay costs.
        self.costs = {
            "retry_cost": 5.0,
            "reminder_cost": 1.0,
            "manual_review_cost": 100.0,
            "customer_friction_cost": 45.0 # Implicit cost of annoying customer with automatic retry
        }
        
        # Simulation Assumptions (Recovery Multipliers)
        # Note: These values are illustrative assumptions and must be replaced with empirically estimated action-level recovery rates before production use.
        self.recovery_multipliers = {
            "retry": 1.0,
            "reminder": 0.50,
            "manual_review": 0.75
        }
        
    def evaluate_strategy_a_blind_retry(self):
        """Strategy A: Blindly retry every failed payment."""
        total_payments = len(self.df)
        action_cost_per_payment = self.costs['retry_cost'] + self.costs['customer_friction_cost']
        
        # Blind Retry gets full expected recovery modeled by the retry multiplier (1.0)
        expected_recovery = (self.df['payment_amount'] * self.df['recovery_probability'] * self.recovery_multipliers['retry']).sum()
        total_cost = total_payments * action_cost_per_payment
        net_recovery = expected_recovery - total_cost
        
        return {
            "strategy": "Blind Retry",
            "total_payments": total_payments,
            "retry_count": total_payments,
            "retry_percentage": 100.0,
            "expected_recovery": expected_recovery,
            "action_cost": total_cost,
            "expected_net_recovery": net_recovery
        }

    def evaluate_strategy_b_rule_based(self):
        """Strategy B: Use Stage 5 heuristic thresholds."""
        total_payments = len(self.df)
        
        # Action masks
        retry_mask = self.df['recovery_probability'] >= 0.65
        reminder_mask = (self.df['recovery_probability'] >= 0.35) & (self.df['recovery_probability'] < 0.65)
        manual_mask = self.df['recovery_probability'] < 0.35
        
        retry_count = retry_mask.sum()
        reminder_count = reminder_mask.sum()
        manual_count = manual_mask.sum()
        
        # Expected recovery for each segment using the specific multipliers
        expected_retry_recovery = (self.df.loc[retry_mask, 'payment_amount'] * self.df.loc[retry_mask, 'recovery_probability'] * self.recovery_multipliers['retry']).sum()
        expected_reminder_recovery = (self.df.loc[reminder_mask, 'payment_amount'] * self.df.loc[reminder_mask, 'recovery_probability'] * self.recovery_multipliers['reminder']).sum()
        expected_manual_recovery = (self.df.loc[manual_mask, 'payment_amount'] * self.df.loc[manual_mask, 'recovery_probability'] * self.recovery_multipliers['manual_review']).sum()
        
        expected_recovery = expected_retry_recovery + expected_reminder_recovery + expected_manual_recovery
        
        # Costs
        retry_cost_total = retry_count * (self.costs['retry_cost'] + self.costs['customer_friction_cost'])
        reminder_cost_total = reminder_count * self.costs['reminder_cost']
        manual_cost_total = manual_count * self.costs['manual_review_cost']
        
        total_action_cost = retry_cost_total + reminder_cost_total + manual_cost_total
        net_recovery = expected_recovery - total_action_cost
        
        return {
            "strategy": "Current Rule-Based",
            "total_payments": total_payments,
            "retry_count": retry_count,
            "retry_percentage": (retry_count / total_payments) * 100,
            "expected_recovery": expected_recovery,
            "action_cost": total_action_cost,
            "expected_net_recovery": net_recovery
        }

    def evaluate_strategy_c_selective(self, threshold, action_cost):
        """Strategy C: Probability-based selective retry."""
        total_payments = len(self.df)
        
        retry_mask = self.df['recovery_probability'] >= threshold
        retry_count = retry_mask.sum()
        retry_percentage = (retry_count / total_payments) * 100 if total_payments > 0 else 0
        
        # Expected recovery is only for the ones we retry
        expected_recovery = (self.df.loc[retry_mask, 'payment_amount'] * self.df.loc[retry_mask, 'recovery_probability'] * self.recovery_multipliers['retry']).sum()
        
        recovery_cost = retry_count * action_cost
        
        expected_net_recovery = expected_recovery - recovery_cost
        recovery_per_retry = (expected_recovery / retry_count) if retry_count > 0 else 0
        roi = (expected_recovery / recovery_cost) if recovery_cost > 0 else 0
        
        return {
            "threshold": threshold,
            "retry_count": retry_count,
            "retry_percentage": retry_percentage,
            "expected_recovery": expected_recovery,
            "recovery_cost": recovery_cost,
            "expected_net_recovery": expected_net_recovery,
            "recovery_per_retry": recovery_per_retry,
            "roi": roi
        }

    def threshold_sweep(self, action_cost):
        thresholds = np.arange(0.0, 1.0, 0.05)
        results = []
        for t in thresholds:
            results.append(self.evaluate_strategy_c_selective(t, action_cost))
        
        df_results = pd.DataFrame(results)
        optimal_idx = df_results['expected_net_recovery'].idxmax()
        optimal_row = df_results.iloc[optimal_idx]
        return df_results, optimal_row

def run_simulation():
    print("Starting Recovery Strategy Simulation...")
    os.makedirs('reports', exist_ok=True)
    
    sim = RecoverySimulator()
    action_cost_base = sim.costs['retry_cost'] + sim.costs['customer_friction_cost']
    
    assert (sim.df['recovery_probability'] >= 0).all() and (sim.df['recovery_probability'] <= 1).all(), "Probabilities not between 0 and 1"
    assert (sim.df['expected_recovery'] >= 0).all(), "Expected recovery must be non-negative"
    
    # Task 2: Strategy Evaluation
    strat_a = sim.evaluate_strategy_a_blind_retry()
    strat_b = sim.evaluate_strategy_b_rule_based()
    
    assert strat_a['expected_recovery'] > strat_b['expected_recovery'], "Strategy B expected recovery should be lower due to action multipliers"
    
    # Task 4: Threshold Analysis
    df_sweep, opt_row = sim.threshold_sweep(action_cost_base)
    df_sweep.to_csv('reports/threshold_optimization.csv', index=False)
    
    strat_c = {
        "strategy": "Optimized Selective Recovery",
        "total_payments": len(sim.df),
        "retry_count": opt_row['retry_count'],
        "retry_percentage": opt_row['retry_percentage'],
        "expected_recovery": opt_row['expected_recovery'],
        "action_cost": opt_row['recovery_cost'],
        "expected_net_recovery": opt_row['expected_net_recovery']
    }
    
    # Additional validation checks on outcomes
    for s in [strat_a, strat_b, strat_c]:
        assert s['expected_recovery'] >= 0, f"{s['strategy']} expected recovery must be >= 0"
        assert s['action_cost'] >= 0, f"{s['strategy']} action cost must be >= 0"
    
    # Task 5: Strategy Comparison
    df_compare = pd.DataFrame([strat_a, strat_b, strat_c])
    df_compare.to_csv('reports/strategy_comparison.csv', index=False)
    
    # Task 8: Robustness / Sensitivity
    cost_scenarios = {
        "LOW COST": 10.0,
        "BASE COST": 50.0,
        "HIGH COST": 250.0
    }
    
    sensitivity_results = []
    for name, cost in cost_scenarios.items():
        _, best_row = sim.threshold_sweep(cost)
        sensitivity_results.append({
            "cost_scenario": name,
            "retry_cost": cost,
            "optimal_threshold": best_row['threshold'],
            "expected_net_recovery": best_row['expected_net_recovery'],
            "retry_percentage": best_row['retry_percentage']
        })
    df_sens = pd.DataFrame(sensitivity_results)
    df_sens.to_csv('reports/threshold_sensitivity.csv', index=False)
    
    # Visualization Generation
    
    # Plot 1: Net Recovery by Threshold
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_sweep, x='threshold', y='expected_net_recovery', marker='o', lw=2)
    plt.axvline(x=opt_row['threshold'], color='red', linestyle='--', label=f"Optimal: {opt_row['threshold']:.2f}")
    plt.title("Expected Net Recovery by Retry Threshold")
    plt.xlabel("Probability Threshold")
    plt.ylabel("Expected Net Recovery (₹)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/net_recovery_by_threshold.png')
    plt.close()
    
    # Plot 2: Strategy Comparison
    plt.figure(figsize=(12, 6))
    df_melt = df_compare.melt(id_vars=['strategy'], 
                              value_vars=['expected_recovery', 'action_cost', 'expected_net_recovery'],
                              var_name='Metric', value_name='Value (₹)')
    sns.barplot(data=df_melt, x='strategy', y='Value (₹)', hue='Metric')
    plt.title("Recovery Strategies Comparison")
    plt.ylabel("Amount (₹)")
    plt.xlabel("Strategy")
    plt.tight_layout()
    plt.savefig('reports/strategy_comparison.png')
    plt.close()
    
    # Plot 3: Sensitivity Analysis
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_sens, x='cost_scenario', y='optimal_threshold', color='steelblue')
    plt.title("Optimal Threshold Sensitivity to Retry Costs")
    plt.ylabel("Optimal Probability Threshold")
    plt.xlabel("Cost Scenario")
    plt.ylim(0, 1.0)
    for i, row in df_sens.iterrows():
        plt.text(i, row['optimal_threshold'] + 0.02, f"{row['optimal_threshold']:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig('reports/threshold_sensitivity.png')
    plt.close()

    # Markdown Report Generation
    
    report_content = f"""# Recovery Strategy & Cost-Aware Optimization Analysis

## 1. Objective
Upgrade the recovery decision layer from fixed heuristic thresholds to a cost-aware strategy analysis. The goal is to maximize **Expected Net Recovery** by selectively identifying which payments financially justify a recovery attempt.

## 2. Simulation Assumptions
**IMPORTANT:** 
- The monetary costs applied in this simulation are configuration assumptions for testing purposes only. They do not reflect actual Razorpay production costs.
- The recovery probabilities are model estimates based on a synthetic dataset, and expected recovery is not guaranteed actual recovered revenue. This simulation is not a representation of Razorpay's actual economics.
- The simulator models action effectiveness using configurable recovery multipliers. These values are illustrative assumptions and must be replaced with empirically estimated action-level recovery rates before production use.

*Base Simulation Parameters (Costs):*
- **Retry Cost:** ₹{sim.costs['retry_cost']}
- **Customer Friction Cost:** ₹{sim.costs['customer_friction_cost']}
- **Total Effective Retry Action Cost:** ₹{action_cost_base}
- **Reminder Cost:** ₹{sim.costs['reminder_cost']}
- **Manual Review Cost:** ₹{sim.costs['manual_review_cost']}

*Simulation Assumptions (Recovery Multipliers):*
- **Retry Multiplier:** {sim.recovery_multipliers['retry']}
- **Reminder Multiplier:** {sim.recovery_multipliers['reminder']}
- **Manual Review Multiplier:** {sim.recovery_multipliers['manual_review']}

## 3. Strategy Definitions
- **Strategy A (Blind Retry):** Retry every single failed payment regardless of probability. Action Cost applies to every payment.
- **Strategy B (Current Rule-Based):** Stage 5 logic (>=0.65 Retry, 0.35-0.64 Reminder, <0.35 Manual). Action costs and action-specific recovery multipliers apply based on probability bands.
- **Strategy C (Optimized Selective):** Sweep thresholds from 0 to 1 to find the exact probability cutoff that maximizes Expected Net Recovery. Only retries >= threshold incur action cost and generate expected recovery.

## 4. Threshold Sweep & Optimal Threshold
By sweeping thresholds at ₹{action_cost_base} cost per retry, the model identified **{opt_row['threshold']:.2f}** as the optimal threshold.

*At {opt_row['threshold']:.2f}:*
- **Retry Percentage:** {opt_row['retry_percentage']:.1f}%
- **Expected Recovery:** ₹{opt_row['expected_recovery']:,.2f}
- **Action Cost:** ₹{opt_row['recovery_cost']:,.2f}
- **Expected Net Recovery:** ₹{opt_row['expected_net_recovery']:,.2f}

## 5. Strategy Comparison

| Strategy | Total Payments | Retry % | Expected Recovery | Action Cost | Net Recovery |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **A. Blind Retry** | {strat_a['total_payments']:,} | 100.0% | ₹{strat_a['expected_recovery']:,.2f} | ₹{strat_a['action_cost']:,.2f} | ₹{strat_a['expected_net_recovery']:,.2f} |
| **B. Current Rule-Based**| {strat_b['total_payments']:,} | {strat_b['retry_percentage']:.1f}% | ₹{strat_b['expected_recovery']:,.2f} | ₹{strat_b['action_cost']:,.2f} | ₹{strat_b['expected_net_recovery']:,.2f} |
| **C. Optimized Selective**| {strat_c['total_payments']:,} | {strat_c['retry_percentage']:.1f}% | ₹{strat_c['expected_recovery']:,.2f} | ₹{strat_c['action_cost']:,.2f} | ₹{strat_c['expected_net_recovery']:,.2f} |

## 6. Business Interpretation
- The optimized strategy strictly reduces action volume compared to Blind Retry (retrying only {strat_c['retry_percentage']:.1f}% instead of 100%), vastly decreasing operational and customer friction costs.
- The Current Rule-Based strategy suffers significantly in Net Recovery because manual reviews are expensive and the expected recovery generated by a reminder or manual intervention is lower (based on simulation multipliers) than a direct retry.
- By targeting only payments with a mathematically justifiable expected return (prob >= {opt_row['threshold']:.2f}), Strategy C delivers the highest cost efficiency and protects expected revenue.

## 7. Limitations
- Monetary costs are simulation assumptions.
- The simulator models action effectiveness using configurable recovery multipliers. These values are illustrative assumptions and must be replaced with empirically estimated action-level recovery rates before production use.

## 8. Sensitivity Analysis
We tested how the optimal threshold responds to different cost environments:

| Scenario | Effective Retry Cost | Optimal Threshold | Retry Percentage |
| :--- | :--- | :--- | :--- |
"""
    for row in sensitivity_results:
        report_content += f"| {row['cost_scenario']} | ₹{row['retry_cost']} | {row['optimal_threshold']:.2f} | {row['retry_percentage']:.1f}% |\n"
        
    with open('reports/recovery_strategy_analysis.md', 'w') as f:
        f.write(report_content)
        
    print("Simulation Complete. Reports and visuals generated.")
    
    # Test Examples
    print("\n--- Test Example Evaluation ---")
    opt_thresh = opt_row['threshold']
    
    examples = [
        {"name": "A", "amount": 5000, "prob": 0.973},
        {"name": "B", "amount": 2500, "prob": 0.372},
        {"name": "C", "amount": 8000, "prob": 0.001}
    ]
    
    for ex in examples:
        expected = ex['amount'] * ex['prob']
        print(f"\nExample {ex['name']} (₹{ex['amount']}, Prob: {ex['prob']}):")
        print(f"  Calculated Expected Recovery: ₹{expected:.2f}")
        
        # Strategy A
        print(f"  Blind Retry: RETRY")
        
        # Strategy B
        if ex['prob'] >= 0.65:
            sb = "RETRY"
        elif ex['prob'] >= 0.35:
            sb = "REMINDER"
        else:
            sb = "MANUAL REVIEW"
        print(f"  Rule-Based: {sb}")
        
        # Strategy C
        sc = "RETRY" if ex['prob'] >= opt_thresh else "DO NOTHING"
        print(f"  Optimized Selective (Threshold={opt_thresh:.2f}): {sc}")

if __name__ == "__main__":
    run_simulation()
