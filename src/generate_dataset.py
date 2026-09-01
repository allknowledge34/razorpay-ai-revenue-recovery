import pandas as pd
import numpy as np
import os

def generate_failed_payments_data(n_samples=20000, seed=42):
    np.random.seed(seed)
    
    # 1. Base Customer & Payment Features
    payment_amount = np.random.lognormal(mean=np.log(500), sigma=1.0, size=n_samples)
    payment_amount = np.clip(payment_amount, 10, 50000).round(2)
    
    failure_reasons = ['insufficient_funds', 'invalid_card', 'technical_error', 'limit_exceeded']
    failure_reason = np.random.choice(failure_reasons, size=n_samples, p=[0.5, 0.2, 0.2, 0.1])
    
    payment_methods = ['credit_card', 'debit_card', 'upi', 'bank_transfer']
    payment_method = np.random.choice(payment_methods, size=n_samples, p=[0.4, 0.3, 0.2, 0.1])
    
    customer_tenure_months = np.random.randint(1, 60, size=n_samples)
    
    past_successful_payments = np.random.negative_binomial(n=5, p=0.3, size=n_samples)
    past_failed_payments = np.random.poisson(lam=1.5, size=n_samples)
    
    total_past = past_successful_payments + past_failed_payments
    historical_success_rate = np.where(total_past > 0, past_successful_payments / total_past, 0.5)
    
    time_since_last_success_days = np.random.exponential(scale=30, size=n_samples).astype(int)
    time_since_last_success_days = np.clip(time_since_last_success_days, 1, 365)
    
    is_subscription = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
    
    days_overdue = np.random.randint(1, 30, size=n_samples)
    recovery_attempts_so_far = np.random.randint(0, 4, size=n_samples)
    
    # 2. Simulate Target Relationship
    # We use a logistic-style approach to create realistic noisy correlations.
    logits = -0.5
    logits += (historical_success_rate - 0.5) * 3.0
    logits += (customer_tenure_months / 60.0) * 1.5
    logits -= (time_since_last_success_days / 365.0) * 1.0
    logits -= (past_failed_payments * 0.3)
    logits -= (days_overdue / 30.0) * 2.0
    logits -= (recovery_attempts_so_far * 0.5)
    logits += is_subscription * 0.8
    
    reason_impact = {
        'technical_error': 2.0,      # Often recoverable via retry
        'insufficient_funds': 0.0,   # Baseline
        'limit_exceeded': -0.5,      # Harder to recover
        'invalid_card': -2.0         # Very hard, requires customer to update
    }
    logits += np.array([reason_impact[r] for r in failure_reason])
    
    probs = 1 / (1 + np.exp(-logits))
    recovered = np.random.binomial(1, probs)
    
    # 3. Create DataFrame
    df = pd.DataFrame({
        'payment_id': [f"pay_{i:06d}" for i in range(1, n_samples + 1)],
        'customer_id': [f"cust_{np.random.randint(10000, 99999)}" for _ in range(n_samples)],
        'payment_amount': payment_amount,
        'failure_reason': failure_reason,
        'payment_method': payment_method,
        'customer_tenure_months': customer_tenure_months,
        'past_successful_payments': past_successful_payments,
        'past_failed_payments': past_failed_payments,
        'historical_success_rate': np.round(historical_success_rate, 2),
        'time_since_last_success_days': time_since_last_success_days,
        'is_subscription': is_subscription,
        'days_overdue': days_overdue,
        'recovery_attempts_so_far': recovery_attempts_so_far,
        'recovered': recovered
    })
    
    return df

if __name__ == "__main__":
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    df = generate_failed_payments_data(20000, 42)
    output_path = os.path.join(output_dir, "failed_payments.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset generated and saved to {output_path}")
    print("Class distribution:")
    print(df['recovered'].value_counts(normalize=True))
