import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from src.domain.prioritization import RecoveryPrioritizer

def generate_prioritization_benchmark():
    df_scored = pd.read_csv('data/failed_payments_scored.csv')
    
    prioritizer = RecoveryPrioritizer()
    df_prioritized = prioritizer.batch_prioritize(df_scored)
    
    total_payments = len(df_prioritized)
    total_revenue_at_risk = df_prioritized['payment_amount'].sum()
    total_base_expected = df_prioritized['base_expected_recovery'].sum()
    total_business_adj = df_prioritized['business_adjusted_expected_recovery'].sum()
    
    tier_counts = df_prioritized['priority_tier'].value_counts()
    tier_counts_sub = pd.crosstab(df_prioritized['priority_tier'], df_prioritized['is_subscription'])
    
    # Calculate concentrations
    top_1_idx = int(total_payments * 0.01)
    top_5_idx = int(total_payments * 0.05)
    top_10_idx = int(total_payments * 0.10)
    top_20_idx = int(total_payments * 0.20)
    
    top_1_val = df_prioritized.iloc[:top_1_idx]['business_adjusted_expected_recovery'].sum()
    top_5_val = df_prioritized.iloc[:top_5_idx]['business_adjusted_expected_recovery'].sum()
    top_10_val = df_prioritized.iloc[:top_10_idx]['business_adjusted_expected_recovery'].sum()
    top_20_val = df_prioritized.iloc[:top_20_idx]['business_adjusted_expected_recovery'].sum()
    
    md = f"""# Business-Adjusted Recovery Prioritization Benchmark

## 1. Objective
Evaluate the deterministic business-value ranking of recovery opportunities across the synthetic batch, establishing actionable queues without overriding bounded recovery policies.

## 2. Architecture
ML Inference -> Recovery Probability -> Prioritization Layer -> Priority Queue -> Recovery Orchestrator

## 3. Formula
- **Base Expected Recovery**: `recovery_probability * payment_amount`
- **Business-Adjusted Expected Recovery**: `base_expected_recovery * subscription_multiplier`
- **Subscription Multiplier**: `1.5` if subscription, else `1.0`

## 4. Tier Definitions
- **CRITICAL**: >= ₹10,000
- **HIGH**: ₹2,500 - ₹9,999
- **MEDIUM**: ₹500 - ₹2,499
- **LOW**: < ₹500

## 5. Results
- **Total Payments**: {total_payments:,}
- **Total Revenue at Risk**: ₹{total_revenue_at_risk:,.2f}
- **Total Base Expected Recovery**: ₹{total_base_expected:,.2f}
- **Total Business-Adjusted Expected Recovery**: ₹{total_business_adj:,.2f}

### Tier Distribution
| Tier | Count |
|---|---|
| CRITICAL | {tier_counts.get('CRITICAL', 0):,} |
| HIGH | {tier_counts.get('HIGH', 0):,} |
| MEDIUM | {tier_counts.get('MEDIUM', 0):,} |
| LOW | {tier_counts.get('LOW', 0):,} |

### Value Concentration
- **Top 1% of queue**: ₹{top_1_val:,.2f} ({top_1_val/total_business_adj*100:.1f}%)
- **Top 5% of queue**: ₹{top_5_val:,.2f} ({top_5_val/total_business_adj*100:.1f}%)
- **Top 10% of queue**: ₹{top_10_val:,.2f} ({top_10_val/total_business_adj*100:.1f}%)
- **Top 20% of queue**: ₹{top_20_val:,.2f} ({top_20_val/total_business_adj*100:.1f}%)

## 6. Interpretation
The prioritization layer successfully isolates the highest-value opportunities. A small percentage of the queue contains a disproportionate amount of the recoverable business value. 

## 7. Interaction with Bounded Recovery
Prioritization determines queue order; bounded recovery policy still controls whether an action is allowed. Even a CRITICAL priority payment will be halted if it violates maximum attempt constraints or economic viability bounds.

## 8. Double-Counting Considerations
The prioritization layer intentionally avoids manually reapplying model features such as attempts, overdue days, historical success rate, or failure reasons. Those remain pure inputs to the ML probability model.

## 9. Synthetic Assumptions
The priority value is a business-value ranking derived from model-estimated recovery probability and synthetic economic assumptions (like the 1.5x LTV multiplier). 

## 10. Limitations
It is not a causal risk score or a guarantee of recovered revenue. The thresholds and multipliers are synthetic configurations for demonstration.
"""
    
    os.makedirs('reports', exist_ok=True)
    with open('reports/prioritization_benchmark.md', 'w') as f:
        f.write(md)
        
    df_prioritized.to_csv('reports/prioritization_benchmark.csv', index=False)
    
    # Plot tier distribution
    plt.figure(figsize=(10, 6))
    tiers = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
    counts = [tier_counts.get(t, 0) for t in tiers]
    colors = ['#440154', '#31688e', '#35b779', '#fde725']
    plt.bar(tiers, counts, color=colors)
    plt.title('Priority Tier Distribution (Business-Adjusted Value)')
    plt.ylabel('Number of Payments')
    plt.xlabel('Priority Tier')
    plt.savefig('reports/prioritization_tier_distribution.png')
    plt.close()

if __name__ == '__main__':
    generate_prioritization_benchmark()
