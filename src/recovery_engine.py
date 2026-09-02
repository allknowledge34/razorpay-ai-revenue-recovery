import pandas as pd
import numpy as np
import joblib
from src.data_validator import DataValidator

import os
import matplotlib.pyplot as plt
import seaborn as sns

class RecoveryEngine:
    def __init__(self, model_path='models/revenue_recovery_model.joblib'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
        
        # Load the pre-trained Logistic Regression pipeline
        self.model = joblib.load(model_path)
        
        # Thresholds derived from the validation probability distribution:
        # P(Recovery) overall is ~43%. 
        # >0.65 represents strong confidence (Precision > 85%) -> HIGH priority
        # 0.35 to 0.65 represents uncertainty -> MEDIUM priority
        # <0.35 represents highly unlikely recovery -> LOW priority
        self.high_threshold = 0.65
        self.medium_threshold = 0.35

    def get_action_and_priority(self, probability):
        """Map probability to business priority and recommended action."""
        if probability >= self.high_threshold:
            return "HIGH", "Retry Payment"
        elif probability >= self.medium_threshold:
            return "MEDIUM", "Payment Method Reminder"
        else:
            return "LOW", "Manual Review / Stop Automatic Retry"

    def predict_recovery(self, record: dict) -> dict:
        """
        Predict recovery for a single payment record.
        Args:
            record (dict): Dictionary representing a single failed payment.
        Returns:
            dict: The prediction results including probabilities, expected revenue, and action.
        """
        validation_result = DataValidator.validate_record(record)
        if not validation_result.is_valid:
            raise ValueError(f"Invalid record: {', '.join(validation_result.errors)}")

        df = pd.DataFrame([record])
        cols_to_drop = [c for c in ['payment_id', 'customer_id', 'recovered'] if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        # Predict probability
        prob = self.model.predict_proba(df)[0, 1]
        
        prob_validation = DataValidator.validate_prediction_probability(prob)
        if not prob_validation.is_valid:
             raise ValueError(f"Invalid model output: {', '.join(prob_validation.errors)}")
        
        # Calculate expected recovery
        expected_recovery = float(record['payment_amount']) * prob
        
        priority, action = self.get_action_and_priority(prob)
        
        return {
            'recovery_probability': prob,
            'expected_recovery': expected_recovery,
            'priority': priority,
            'recommended_action': action
        }

    def predict_batch(self, input_csv: str, output_csv: str):
        """
        Predict recovery for a batch of payments.
        Args:
            input_csv (str): Path to input CSV.
            output_csv (str): Path to save the augmented CSV.
        Returns:
            pd.DataFrame: The resulting dataframe with new columns.
        """
        df = pd.read_csv(input_csv)
        
        cols_to_drop = [c for c in ['payment_id', 'customer_id', 'recovered'] if c in df.columns]
        X = df.drop(columns=cols_to_drop)
        
        # Predict probabilities
        probs = self.model.predict_proba(X)[:, 1]
        probs = np.clip(probs, 0.0, 1.0)
        
        # Calculate new columns
        df['recovery_probability'] = probs
        df['expected_recovery'] = df['payment_amount'] * df['recovery_probability']
        
        # Vectorized mapping for actions and priorities
        conditions = [
            (df['recovery_probability'] >= self.high_threshold),
            (df['recovery_probability'] >= self.medium_threshold) & (df['recovery_probability'] < self.high_threshold)
        ]
        
        priority_choices = ['HIGH', 'MEDIUM']
        action_choices = ['Retry Payment', 'Payment Method Reminder']
        
        df['priority'] = np.select(conditions, priority_choices, default='LOW')
        df['recommended_action'] = np.select(conditions, action_choices, default='Manual Review / Stop Automatic Retry')
        
        # Save to CSV
        df.to_csv(output_csv, index=False)
        print(f"Batch predictions saved to {output_csv}")
        return df

def generate_visuals(df):
    """Generate and save distribution visuals for the reports."""
    os.makedirs('reports', exist_ok=True)
    
    # Visual 1: Recovery Priority Distribution
    plt.figure(figsize=(8, 5))
    priority_order = ['HIGH', 'MEDIUM', 'LOW']
    sns.countplot(data=df, x='priority', order=priority_order, palette='viridis')
    plt.title('Distribution of Recovery Priorities')
    plt.ylabel('Number of Failed Payments')
    plt.xlabel('Priority Level')
    plt.tight_layout()
    plt.savefig('reports/recovery_priority_distribution.png')
    plt.close()
    
    # Visual 2: Expected Recovery Revenue Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='expected_recovery', hue='priority', hue_order=priority_order, 
                 bins=50, multiple='stack', palette='viridis')
    plt.title('Distribution of Expected Recoverable Revenue')
    plt.xlabel('Expected Revenue (₹)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('reports/expected_recovery_distribution.png')
    plt.close()
    print("Visualizations saved to reports/")

def write_markdown_report():
    report_content = """# Revenue Recovery Intelligence

## Objective
To transform raw machine learning probability outputs into actionable, automated business decisions that maximize recovered revenue while minimizing operational cost and customer friction.

## Core Formula
The decision engine ranks and evaluates payments using Expected Recoverable Revenue:
`Expected Recoverable Revenue = Payment Amount × Recovery Probability`

## Threshold Logic
The Logistic Regression model outputs a probability between 0 and 1. Based on the dataset's baseline recovery rate (~43%) and validation distribution, we selected the following calibrated thresholds:

- **High Threshold (>= 0.65):** Represents high statistical confidence in recovery. Payments in this band have a strong historical precedent of succeeding upon retry (e.g., temporary technical glitches).
- **Medium Threshold (0.35 - 0.64):** Represents uncertainty. These often require minor customer intervention to resolve (e.g., updating card details).
- **Low Threshold (< 0.35):** Represents payments highly unlikely to recover (e.g., hard declines, historically inactive customers). Continuing to retry these costs gateway fees and risks chargebacks.

## Action Mapping

| Probability Band | Priority | Recommended Action | Rationale |
| :--- | :--- | :--- | :--- |
| **>= 0.65** | **HIGH** | Retry Payment | Immediate, silent automated retry to capture high-probability revenue seamlessly. |
| **0.35 - 0.64** | **MEDIUM** | Payment Method Reminder | Prompt the user to update their payment method or add funds. |
| **< 0.35** | **LOW** | Manual Review / Stop Automatic Retry | Halt automated retries to save gateway costs. Flag for manual support review or suspend the service. |

## Limitations
- The expected revenue calculation assumes the full amount is recovered. Partial recoveries are not modeled.
- Action mapping currently uses static probability thresholds. In a fully mature system, these thresholds could be dynamically optimized to maximize total net ROI based on changing gateway retry fees.
"""
    with open('reports/recovery_engine.md', 'w') as f:
        f.write(report_content)
    print("Markdown report saved to reports/recovery_engine.md")

if __name__ == "__main__":
    print("Initializing Recovery Engine...")
    engine = RecoveryEngine()
    
    print("\\n--- Testing Single Predictions ---")
    
    # Example A: Technical Error (High historical success, recent success)
    record_a = {
        'payment_amount': 5000,
        'failure_reason': 'technical_error',
        'payment_method': 'upi',
        'is_subscription': 0,
        'customer_tenure_months': 24,
        'past_successful_payments': 10,
        'past_failed_payments': 0,
        'historical_success_rate': 1.0,
        'time_since_last_success_days': 5,
        'days_overdue': 1,
        'recovery_attempts_so_far': 0
    }
    
    # Example B: Insufficient Funds (Average history)
    record_b = {
        'payment_amount': 2500,
        'failure_reason': 'insufficient_funds',
        'payment_method': 'debit_card',
        'is_subscription': 1,
        'customer_tenure_months': 12,
        'past_successful_payments': 3,
        'past_failed_payments': 2,
        'historical_success_rate': 0.6,
        'time_since_last_success_days': 30,
        'days_overdue': 3,
        'recovery_attempts_so_far': 1
    }

    # Example C: Invalid Card (Poor history, overdue)
    record_c = {
        'payment_amount': 8000,
        'failure_reason': 'invalid_card',
        'payment_method': 'credit_card',
        'is_subscription': 0,
        'customer_tenure_months': 2,
        'past_successful_payments': 0,
        'past_failed_payments': 4,
        'historical_success_rate': 0.0,
        'time_since_last_success_days': 999,
        'days_overdue': 15,
        'recovery_attempts_so_far': 3
    }
    
    for name, rec in [("A (Technical Error)", record_a), ("B (Insufficient Funds)", record_b), ("C (Invalid Card)", record_c)]:
        res = engine.predict_recovery(rec)
        print(f"\\nExample {name}:")
        print(f"  Amount: ₹{rec['payment_amount']}")
        print(f"  Prob:   {res['recovery_probability']:.4f}")
        print(f"  Exp:    ₹{res['expected_recovery']:.2f}")
        print(f"  Prior:  {res['priority']}")
        print(f"  Action: {res['recommended_action']}")
        
        assert 0.0 <= res['recovery_probability'] <= 1.0
        assert res['expected_recovery'] >= 0.0

    print("\\n--- Testing Batch Predictions ---")
    input_file = 'data/failed_payments.csv'
    output_file = 'data/failed_payments_scored.csv'
    
    scored_df = engine.predict_batch(input_file, output_file)
    
    # Validate columns exist
    required_cols = ['recovery_probability', 'expected_recovery', 'priority', 'recommended_action']
    for col in required_cols:
        assert col in scored_df.columns, f"Missing {col} in batch output"
    
    # Validate bounds
    assert scored_df['recovery_probability'].min() >= 0.0
    assert scored_df['recovery_probability'].max() <= 1.0
    assert scored_df['expected_recovery'].min() >= 0.0

    print("\\n--- Generating Reports ---")
    generate_visuals(scored_df)
    write_markdown_report()
    
    print("\\nStage 5 processing complete.")
