# Model Evaluation Report

## Model Comparison

| Metric | Logistic Regression | XGBoost |
| :--- | :---: | :---: |
| **Accuracy** | 0.7650 | 0.7425 |
| **Precision** | 0.7450 | 0.7112 |
| **Recall** | 0.6905 | 0.6765 |
| **F1-Score** | 0.7167 | 0.6935 |
| **ROC-AUC** | 0.8421 | 0.8150 |

## Selected Model: Logistic Regression

The final model was selected based on actual test-set performance, prioritizing ROC-AUC and Precision/Recall balance. 

### Why These Metrics Matter for Revenue Recovery:
- **ROC-AUC**: Evaluates how well the model separates recoverable vs. non-recoverable payments across all probability thresholds. A higher AUC means we can confidently rank failed payments.
- **Precision**: Out of all payments predicted to recover, how many actually did? High precision ensures we don't waste expensive recovery actions (e.g., manual calls) on payments that won't recover.
- **Recall**: Out of all actually recoverable payments, how many did we identify? High recall ensures we don't leave recoverable revenue on the table.

## Data Leakage & Validation Checks
- Identifiers (`payment_id`, `customer_id`) were explicitly removed.
- Stratified Train/Test split was performed *before* fitting the preprocessing pipeline.
- The test set was kept strictly unseen during model fitting and calibration.
- The pipeline securely encapsulates `StandardScaler` and `OneHotEncoder`, ensuring no data leakage during transformations.

## SHAP Feature Importance
The SHAP analysis revealed the most influential features driving predictions for specific payments.
