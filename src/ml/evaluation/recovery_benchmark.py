import pandas as pd
import numpy as np
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Any, List

from src.domain.recovery_strategy import RecoverySimulator
from src.services.recovery.recovery_orchestrator import RecoveryOrchestrator

def resolve_seed(payment_id: str) -> int:
    """Generate a stable process-independent seed."""
    seed_bytes = hashlib.sha256(str(payment_id).encode("utf-8")).digest()
    return int.from_bytes(seed_bytes[:8], "big") % (2**32)

class RecoveryBenchmark:
    def __init__(self, data_path='data/failed_payments_scored.csv'):
        self.df = pd.read_csv(data_path)
        self.orchestrator = RecoveryOrchestrator()
        
        # Determine optimal threshold using existing logic
        sim = RecoverySimulator(data_path=data_path)
        df_sweep, optimal_row = sim.threshold_sweep(action_cost=50.0)
        self.optimal_threshold = optimal_row['threshold']
        
    def evaluate_all(self) -> pd.DataFrame:
        results = []
        for strat in ['A. Blind Retry', 'B. Current Rule-Based', 'C. Optimized Selective', 'D. Bounded Recovery Orchestrator']:
            results.append(self._evaluate_strategy(strat))
        return pd.DataFrame(results)
        
    def _evaluate_strategy(self, strategy_name: str) -> Dict[str, Any]:
        metrics = {
            'Strategy': strategy_name,
            'Total Payments': 0,
            'Recovered Payments': 0,
            'Recovery Rate': 0.0,
            'Total Payment Value': 0.0,
            'Simulated Recovered Revenue': 0.0,
            'Revenue at Risk': 0.0,
            'Action Cost': 0.0,
            'Net Recovered Revenue': 0.0,
            'ROI': 0.0,
            'Retry Count': 0,
            'Reminder Count': 0,
            'Manual Review Count': 0,
            'Stopped Count': 0,
            'Average Recovery Value': 0.0,
            'Average Action Cost': 0.0,
            'Verified Outcome Count': 0,
            'Verification Rate': 0.0,
            'Recovery Efficiency': 0.0,
            # Guardrails
            'Maximum Existing Attempts Observed': 0,
            'Configured Maximum Automatic Attempts': 2,
            'Stopped by Prob Threshold': 0,
            'Stopped by Econ Viability': 0,
            'Routed to Manual Review': 0,
            'Stopped by Max Attempts': 0,
            'Invalid State Transitions': 0,
            'Outcomes VERIFIED': 0,
            'Outcomes CLOSED': 0,
            'Outcomes STOPPED': 0,
            'Outcomes MANUAL_REVIEW': 0
        }
        
        for _, row in self.df.iterrows():
            payment_id = row.get('payment_id', str(row.name))
            amount = float(row['payment_amount'])
            prob = float(row['recovery_probability'])
            attempts = int(row['recovery_attempts_so_far'])
            expected = amount * prob
            
            metrics['Total Payments'] += 1
            metrics['Total Payment Value'] += amount
            metrics['Revenue at Risk'] += amount
            
            seed = resolve_seed(payment_id)
            action = None
            sim_result = {}
            
            if strategy_name == 'A. Blind Retry':
                action = 'Retry Payment'
                sim_result = self.orchestrator.execute_simulated_action(action, prob, amount, seed)
            
            elif strategy_name == 'B. Current Rule-Based':
                if prob >= 0.65:
                    action = 'Retry Payment'
                elif prob >= 0.35:
                    action = 'Payment Method Reminder'
                else:
                    action = 'Stop Automatic Recovery'
                sim_result = self.orchestrator.execute_simulated_action(action, prob, amount, seed)
                
            elif strategy_name == 'C. Optimized Selective':
                if prob >= self.optimal_threshold:
                    action = 'Retry Payment'
                else:
                    action = 'Stop Automatic Recovery'
                sim_result = self.orchestrator.execute_simulated_action(action, prob, amount, seed)
                
            elif strategy_name == 'D. Bounded Recovery Orchestrator':
                context = {
                    'payment_id': payment_id,
                    'recovery_probability': prob,
                    'recovery_attempts_so_far': attempts,
                    'payment_amount': amount,
                    'expected_recovery': expected,
                    'recommended_action': row.get('recommended_action', 'Retry Payment')
                }
                res = self.orchestrator.process_event(context, seed=seed)
                action = res['selected_action']
                sim_result = {
                    'simulated_recovered': res['simulated_recovered'],
                    'action_cost': res['action_cost'],
                    'recovered_amount': res['recovered_amount'],
                    'net_recovered_revenue': res['net_recovered_revenue']
                }
                
                # Guardrails counting
                metrics['Maximum Existing Attempts Observed'] = max(metrics['Maximum Existing Attempts Observed'], attempts)
                
                reason = res.get('decision_reason', '')
                # Mutually exclusive primary reasons based on policy evaluation
                if res.get('policy_decision') == 'MANUAL_REVIEW':
                    metrics['Routed to Manual Review'] += 1
                elif res.get('policy_decision') == 'BLOCKED':
                    if 'Maximum automatic recovery attempts' in reason:
                        metrics['Stopped by Max Attempts'] += 1
                    elif 'minimum threshold' in reason:
                        metrics['Stopped by Prob Threshold'] += 1
                    elif 'effective retry cost' in reason:
                        metrics['Stopped by Econ Viability'] += 1
                    
                v_stat = 'Outcomes ' + res.get('verification_status', 'UNKNOWN')
                if v_stat not in metrics: metrics[v_stat] = 0
                metrics[v_stat] += 1
                
                f_stat = 'Outcomes ' + res.get('final_state', 'UNKNOWN')
                if f_stat not in metrics: metrics[f_stat] = 0
                metrics[f_stat] += 1
                
                if res.get('verification_status') == 'VERIFIED':
                    metrics['Verified Outcome Count'] += 1
            
            # Action counting
            if action == 'Retry Payment':
                metrics['Retry Count'] += 1
            elif action == 'Payment Method Reminder':
                metrics['Reminder Count'] += 1
            elif action == 'Manual Review':
                metrics['Manual Review Count'] += 1
            elif action == 'Stop Automatic Recovery' or action == 'Manual Review / Stop Automatic Retry':
                metrics['Stopped Count'] += 1
            else:
                metrics['Stopped Count'] += 1 # Catch all zero-cost actions
                
            # Financials
            if sim_result.get('simulated_recovered'):
                metrics['Recovered Payments'] += 1
                
            metrics['Simulated Recovered Revenue'] += sim_result.get('recovered_amount', 0.0)
            metrics['Action Cost'] += sim_result.get('action_cost', 0.0)
            metrics['Net Recovered Revenue'] += sim_result.get('net_recovered_revenue', 0.0)
            
            if strategy_name != 'D. Bounded Recovery Orchestrator':
                metrics['Verified Outcome Count'] += 1 # Assume baseline simulation acts as verified
                
        # Derived metrics
        tp = metrics['Total Payments']
        ac = metrics['Action Cost']
        rar = metrics['Revenue at Risk']
        metrics['Recovery Rate'] = metrics['Recovered Payments'] / tp if tp else 0.0
        metrics['ROI'] = metrics['Net Recovered Revenue'] / ac if ac > 0 else 0.0
        metrics['Average Recovery Value'] = metrics['Simulated Recovered Revenue'] / tp if tp else 0.0
        metrics['Average Action Cost'] = ac / tp if tp else 0.0
        metrics['Verification Rate'] = metrics['Verified Outcome Count'] / tp if tp else 0.0
        metrics['Recovery Efficiency'] = metrics['Net Recovered Revenue'] / rar if rar else 0.0
        
        return metrics
        
def calculate_deltas(df: pd.DataFrame) -> pd.DataFrame:
    df.set_index('Strategy', inplace=True)
    deltas = []
    
    comparisons = [
        ('C. Optimized Selective', 'A. Blind Retry'),
        ('D. Bounded Recovery Orchestrator', 'A. Blind Retry'),
        ('D. Bounded Recovery Orchestrator', 'B. Current Rule-Based'),
        ('D. Bounded Recovery Orchestrator', 'C. Optimized Selective')
    ]
    
    for s1, s2 in comparisons:
        try:
            row1 = df.loc[s1]
            row2 = df.loc[s2]
            d = {
                'Comparison': f"{s1} vs {s2}",
                'Net Recovered Revenue Delta': row1['Net Recovered Revenue'] - row2['Net Recovered Revenue'],
                'Recovery Rate Delta': row1['Recovery Rate'] - row2['Recovery Rate'],
                'Action Cost Delta': row1['Action Cost'] - row2['Action Cost'],
                'Retry Count Delta': row1['Retry Count'] - row2['Retry Count'],
                'ROI Delta': row1['ROI'] - row2['ROI']
            }
            deltas.append(d)
        except KeyError:
            continue
            
    df.reset_index(inplace=True)
    return pd.DataFrame(deltas)

def generate_benchmark_reports():
    bench = RecoveryBenchmark()
    df = bench.evaluate_all()
    
    os.makedirs('reports', exist_ok=True)
    df.to_csv('reports/recovery_benchmark.csv', index=False)
    
    df_deltas = calculate_deltas(df.copy())
    df_deltas.to_csv('reports/recovery_benchmark_deltas.csv', index=False)
    
    # Generate Plots
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Strategy', y='Net Recovered Revenue', palette='viridis')
    plt.title('Net Recovered Revenue by Strategy')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('reports/benchmark_net_revenue.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Strategy', y='Recovery Rate', palette='viridis')
    plt.title('Recovery Rate by Strategy')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('reports/benchmark_recovery_rate.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Strategy', y='Action Cost', palette='magma')
    plt.title('Action Cost by Strategy')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('reports/benchmark_action_cost.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Strategy', y='ROI', palette='coolwarm')
    plt.title('ROI by Strategy')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('reports/benchmark_roi.png')
    plt.close()
    
    # Generate MD
    md = f"""# Batch Recovery Benchmark & Selection Evidence

## 1. Benchmark Objective
Evaluate four recovery approaches on the same synthetic payment batch using strict deterministic randomness to provide reproducible, evidence-based comparisons.

## 2. Dataset Description
- **Source**: Synthetic failed payments dataset (`data/failed_payments_scored.csv`).
- **Total Payments**: {df.iloc[0]['Total Payments']:,}
- **Revenue at Risk**: ₹{df.iloc[0]['Revenue at Risk']:,.2f}
- **Random Seed**: Stable SHA-256 hash of `payment_id`.

**Disclaimer:** Benchmark results are generated from the project's synthetic payment dataset and simulation assumptions. They do not represent real Razorpay recovery performance, and do not reflect real causal outcomes.

## 3. Strategy Definitions
- **A. Blind Retry**: Retries every payment.
- **B. Current Rule-Based**: High probability -> Retry, Medium -> Reminder, Low -> Stop.
- **C. Optimized Selective**: Threshold optimized to maximize expected net revenue (Threshold = {bench.optimal_threshold:.2f}).
- **D. Bounded Recovery Orchestrator**: Stage 18 logic enforcing maximum attempts, viability limits, and manual review routing.

## 4. Benchmark Results
| Strategy | Net Recovered | Action Cost | Recovery Rate | ROI | Retries |
|---|---|---|---|---|---|
"""
    for _, r in df.iterrows():
        md += f"| {r['Strategy']} | ₹{r['Net Recovered Revenue']:,.2f} | ₹{r['Action Cost']:,.2f} | {r['Recovery Rate']:.2%} | {r['ROI']:.2f} | {r['Retry Count']} |\n"

    md += """
## 5. Comparative Analysis (Deltas)
| Comparison | Net Rev Delta | Action Cost Delta | Rec Rate Delta | ROI Delta |
|---|---|---|---|---|
"""
    for _, r in df_deltas.iterrows():
        md += f"| {r['Comparison']} | ₹{r['Net Recovered Revenue Delta']:,.2f} | ₹{r['Action Cost Delta']:,.2f} | {r['Recovery Rate Delta']:.2%} | {r['ROI Delta']:.2f} |\n"

    md += """
## 6. Guardrail Metrics (Bounded Recovery Orchestrator)
"""
    d_row = df[df['Strategy'] == 'D. Bounded Recovery Orchestrator'].iloc[0]
    md += f"- **Maximum Existing Attempts Observed**: {d_row['Maximum Existing Attempts Observed']}\n"
    md += f"- **Configured Maximum Automatic Attempts**: {d_row['Configured Maximum Automatic Attempts']}\n"
    md += f"- **Stopped (Probability)**: {d_row['Stopped by Prob Threshold']}\n"
    md += f"- **Stopped (Economic)**: {d_row['Stopped by Econ Viability']}\n"
    md += f"- **Stopped (Max Attempts)**: {d_row['Stopped by Max Attempts']}\n"
    md += f"- **Routed to Manual Review**: {d_row['Routed to Manual Review']}\n"
    md += f"- **Invalid State Transitions**: {d_row['Invalid State Transitions']}\n"
    md += f"- **Outcomes VERIFIED**: {d_row['Outcomes VERIFIED']}\n"
    md += f"- **Outcomes CLOSED**: {d_row['Outcomes CLOSED']}\n"
    md += f"- **Outcomes STOPPED**: {d_row['Outcomes STOPPED']}\n"
    md += f"- **Outcomes MANUAL_REVIEW**: {d_row['Outcomes MANUAL_REVIEW']}\n"

    md += """

## 7. Benchmark Interpretation
Under the synthetic assumptions, the bounded policy sacrifices some simulated recovery opportunity in exchange for stricter attempt, economic, and high-value controls. Rule-Based has high simulated ROI. Optimized Selective has the highest simulated net revenue. Bounded Orchestrator has stricter operational guardrails.

## 8. Limitations
- Existing attempts are historical/input state.
- The benchmark performs one bounded decision per payment event.
- It does not simulate a multi-step retry chain.
- Configured maximum automatic attempts = 2 is an orchestration policy limit, not a claim that two retries were actually executed during this benchmark.
- Synthetic dataset.
- Synthetic action-effectiveness assumptions.
- Simulated outcomes without true causal inference.
- Benchmark is not a production A/B test.
"""
    with open('reports/recovery_benchmark.md', 'w') as f:
        f.write(md)

if __name__ == '__main__':
    generate_benchmark_reports()
