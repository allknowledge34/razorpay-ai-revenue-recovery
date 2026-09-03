import pandas as pd
import numpy as np
import os

from src.domain.recovery_strategy import RecoverySimulator

class OutcomeSimulator:
    def __init__(self, data_path='data/failed_payments_scored.csv', seed=42):
        self.df = pd.read_csv(data_path)
        self.seed = seed
        
        # Reuse existing cost and multiplier assumptions
        self.strategy_simulator = RecoverySimulator(data_path=data_path)
        self.costs = self.strategy_simulator.costs
        self.multipliers = self.strategy_simulator.recovery_multipliers
        
        # We also need a "do nothing" action for Strategy C dropouts
        self.multipliers['do_nothing'] = 0.0
        
        # Map formal actions to costs
        self.action_costs = {
            "Retry Payment": self.costs['retry_cost'] + self.costs['customer_friction_cost'],
            "Payment Method Reminder": self.costs['reminder_cost'],
            "Manual Review / Stop Automatic Retry": self.costs['manual_review_cost'],
            "Do Nothing": 0.0
        }
        
        self.action_multipliers = {
            "Retry Payment": self.multipliers['retry'],
            "Payment Method Reminder": self.multipliers['reminder'],
            "Manual Review / Stop Automatic Retry": self.multipliers['manual_review'],
            "Do Nothing": self.multipliers['do_nothing']
        }
        
        # Determine the optimal threshold for Strategy C from the RecoverySimulator
        action_cost_base = self.costs['retry_cost'] + self.costs['customer_friction_cost']
        df_sweep, opt_row = self.strategy_simulator.threshold_sweep(action_cost_base)
        self.optimal_threshold = opt_row['threshold']
        
        # Generate the shared random draw for all strategies to ensure fair comparison
        rng = np.random.default_rng(self.seed)
        self.uniform_draws = rng.uniform(0, 1, size=len(self.df))

    def _assign_actions_strategy_a(self, df: pd.DataFrame):
        df['simulated_action'] = "Retry Payment"
        return df
        
    def _assign_actions_strategy_b(self, df: pd.DataFrame):
        conditions = [
            df['recovery_probability'] >= 0.65,
            (df['recovery_probability'] >= 0.35) & (df['recovery_probability'] < 0.65)
        ]
        choices = ["Retry Payment", "Payment Method Reminder"]
        df['simulated_action'] = np.select(conditions, choices, default="Manual Review / Stop Automatic Retry")
        return df
        
    def _assign_actions_strategy_c(self, df: pd.DataFrame):
        df['simulated_action'] = np.where(df['recovery_probability'] >= self.optimal_threshold, 
                                          "Retry Payment", "Do Nothing")
        return df

    def simulate_strategy(self, strategy_name: str) -> pd.DataFrame:
        """
        Simulate a recovery strategy and calculate deterministic outcomes.
        """
        df_sim = self.df.copy()
        
        if strategy_name == "Blind Retry":
            df_sim = self._assign_actions_strategy_a(df_sim)
        elif strategy_name == "Current Rule-Based Strategy":
            df_sim = self._assign_actions_strategy_b(df_sim)
        elif strategy_name == "Optimized Selective Strategy":
            df_sim = self._assign_actions_strategy_c(df_sim)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
            
        # Calculate action effectiveness and cost
        df_sim['action_effectiveness'] = df_sim['simulated_action'].map(self.action_multipliers)
        df_sim['action_cost'] = df_sim['simulated_action'].map(self.action_costs)
        
        # Calculate effective probability and clip to [0,1]
        df_sim['effective_recovery_probability'] = (df_sim['recovery_probability'] * df_sim['action_effectiveness']).clip(0, 1)
        
        # Simulate outcome deterministically using the shared random draw
        df_sim['simulated_recovered'] = (self.uniform_draws < df_sim['effective_recovery_probability']).astype(int)
        
        # Calculate revenue
        df_sim['simulated_recovered_revenue'] = df_sim['simulated_recovered'] * df_sim['payment_amount']
        
        # Calculate net revenue
        df_sim['net_recovered_revenue'] = df_sim['simulated_recovered_revenue'] - df_sim['action_cost']
        
        return df_sim

    def calculate_metrics(self, df_sim: pd.DataFrame, strategy_name: str) -> dict:
        total_payments = len(df_sim)
        retry_count = (df_sim['simulated_action'] == "Retry Payment").sum()
        reminder_count = (df_sim['simulated_action'] == "Payment Method Reminder").sum()
        manual_review_count = (df_sim['simulated_action'] == "Manual Review / Stop Automatic Retry").sum()
        
        retry_rate = retry_count / total_payments if total_payments > 0 else 0
        
        recovery_count = df_sim['simulated_recovered'].sum()
        simulated_recovery_rate = recovery_count / total_payments if total_payments > 0 else 0
        
        gross_simulated_recovered_revenue = df_sim['simulated_recovered_revenue'].sum()
        total_action_cost = df_sim['action_cost'].sum()
        net_recovered_revenue = df_sim['net_recovered_revenue'].sum()
        
        average_recovered_revenue_per_payment = gross_simulated_recovered_revenue / total_payments if total_payments > 0 else 0
        
        recovery_cost_per_recovered_payment = total_action_cost / recovery_count if recovery_count > 0 else 0
        
        if total_action_cost > 0:
            roi = (gross_simulated_recovered_revenue - total_action_cost) / total_action_cost
        else:
            roi = 0.0
            
        total_payment_amount = df_sim['payment_amount'].sum()
        revenue_recovery_rate = gross_simulated_recovered_revenue / total_payment_amount if total_payment_amount > 0 else 0
        
        return {
            "strategy": strategy_name,
            "total_payments": total_payments,
            "retry_count": int(retry_count),
            "reminder_count": int(reminder_count),
            "manual_review_count": int(manual_review_count),
            "retry_rate": float(retry_rate),
            "recovery_count": int(recovery_count),
            "simulated_recovery_rate": float(simulated_recovery_rate),
            "gross_simulated_recovered_revenue": float(gross_simulated_recovered_revenue),
            "total_action_cost": float(total_action_cost),
            "net_recovered_revenue": float(net_recovered_revenue),
            "revenue_recovery_rate": float(revenue_recovery_rate),
            "recovery_cost_per_recovered_payment": float(recovery_cost_per_recovered_payment),
            "roi": float(roi)
        }

if __name__ == "__main__":
    simulator = OutcomeSimulator()
    strategies = ["Blind Retry", "Current Rule-Based Strategy", "Optimized Selective Strategy"]
    
    metrics = []
    # Just to get one dataframe for output
    df_opt = None
    for strat in strategies:
        df_sim = simulator.simulate_strategy(strat)
        if strat == "Optimized Selective Strategy":
            df_opt = df_sim
        m = simulator.calculate_metrics(df_sim, strat)
        metrics.append(m)
        
    df_metrics = pd.DataFrame(metrics)
    os.makedirs('reports', exist_ok=True)
    df_metrics.to_csv("reports/outcome_strategy_comparison.csv", index=False)
    
    if df_opt is not None:
        df_opt.to_csv("reports/recovery_outcome_simulation.csv", index=False)
        
    print(df_metrics.T)
