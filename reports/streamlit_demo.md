# Streamlit Demo Interface

## Overview
The Streamlit application (`app/streamlit_app.py`) serves as the interactive business front-end for the AI Revenue Recovery Engine. It allows users to simulate individual failed payments and view high-level batch analytics without needing to write code.

## UI Workflow
The application is divided into two primary tabs:
1. **Single Payment Simulation:** A form where business users can tweak payment attributes (Amount, Reason, History) and immediately see the AI's recommended recovery action and financial impact.
2. **Batch Recovery Analysis:** A dashboard view that ingests the scored historical dataset (`data/failed_payments_scored.csv`) to show aggregate portfolio metrics (Total Value at Risk vs. Expected Recoverable Revenue) and visualizations of priority distributions.

## Connection to the ML Model
The Streamlit app acts strictly as a presentation layer. It **does not** contain any machine learning or threshold logic. 
Instead, it imports the `RecoveryEngine` class from `src/recovery_engine.py`. 
When a user clicks "Analyze Recovery", the app parses the form inputs into a dictionary and passes it to `engine.predict_recovery(record)`. The engine handles scaling, one-hot encoding, pipeline inference, expected value calculation, and returns a clean dictionary of results back to the UI.

## Inputs
The user inputs mirror the features used during ML training:
- Payment Amount (₹)
- Failure Reason (Categorical)
- Payment Method (Categorical)
- Subscription Status (Boolean)
- Customer Tenure (months)
- Past Successful/Failed Payments
- Historical Success Rate (0.0 - 1.0)
- Days Since Last Success
- Days Overdue
- Recovery Attempts So Far

*Note: `payment_id` and `customer_id` are deliberately hidden from the UI since they have no predictive power, as are the final labels.*

## Outputs & Business Explanations
Upon submitting a simulation, the dashboard updates to show:
1. **Recovery Probability:** The raw model output (0-100%).
2. **Expected Recovery:** Probability × Payment Amount.
3. **Priority & Action:** Mapped from the probability thresholds (HIGH, MEDIUM, LOW).
4. **Business Impact:** Visualizes the gap between the amount at risk and the potentially unrecovered amount.
5. **Why this recommendation?:** A rule-based explanation referencing the inputs (like high historical success rate or overdue days) that influenced the score, ensuring transparent business logic.

## Limitations
- The explanations currently use static transparent rules rather than local SHAP computations per-prediction, ensuring blazing-fast UI response times at the expense of precise feature-importance weights per user.
- The batch analysis uses a pre-scored cached dataset (`failed_payments_scored.csv`) for efficiency rather than re-scoring 20,000 records on every Streamlit component render.
