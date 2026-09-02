# Offline Monitoring and Drift Simulation Report

**Disclaimer:** This is a synthetic/offline monitoring framework demonstrated using synthetic data. It does not represent real-time monitoring, real customer data, real Razorpay metrics, or production model degradation.

## 1. Purpose
The purpose of this report is to outline the methodology for offline monitoring of data quality, prediction stability, and simulated data drift for the AI Revenue Recovery Engine.

## 2. Data Quality Summary
The Monitoring Engine evaluates the structural integrity of incoming batches, checking for missing values, duplicates, out-of-bounds numerical values, and invalid categorical inputs.

## 3. Prediction Monitoring
For the provided scored dataset, the framework aggregates key statistics (Mean, Median, Std Dev) for:
- `recovery_probability`
- `expected_recovery`
It also tracks the distribution of recommended business actions (e.g., HIGH/MEDIUM/LOW priority) to detect potential systemic shifts in recovery strategies.

## 4. Drift Simulation Methodology
To demonstrate drift detection, we apply a controlled, deterministic synthetic drift scenario to a duplicate of the original dataset (`simulate_drift(df, seed=42)`).

### Features Intentionally Shifted:
- **`payment_amount`**: Values are multiplied by a uniform random factor between 1.5 and 3.0, representing a significant upward shift in average transaction values.
- **`historical_success_rate`**: Values are reduced by a uniform random amount between 0.1 and 0.3, simulating a sudden degradation in customer reliability.
- **`failure_reason`**: 80% of `technical_error` instances are reassigned to `insufficient_funds`, simulating a systemic shift in the cause of payment failures.

## 5. Drift Metrics
We use Population Stability Index (PSI) to measure the divergence between the reference distribution and the current (simulated) distribution.

**Monitoring Heuristics:**
- **NORMAL:** PSI < 0.1
- **WARNING:** 0.1 <= PSI < 0.2
- **DRIFT:** PSI >= 0.2

The calculated metrics are saved to `reports/drift_metrics.csv`.
