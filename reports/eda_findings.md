# EDA Findings

## Dataset Overview
- **Rows**: 20000
- **Columns**: 14
- **Target Distribution**: Not Recovered (0) = 56.95%, Recovered (1) = 43.05%
- **Missing Values**: 0
- **Duplicates**: 0

## Target Analysis
The dataset has a slight class imbalance but is generally well-balanced (approx 57% negative, 43% positive). Severe class imbalance techniques (like SMOTE) are not required for baseline modeling.

## Feature Analysis
- **Payment Amount**: Highly right-skewed. Most payments are smaller amounts, with a long tail. Log transformation or tree-based models will handle this well.
- **Categorical Features**: 
  - `failure_reason` and `payment_method` have an observable impact on the recovery rate. Technical errors show much higher recovery rates compared to invalid cards.
  - Subscriptions (`is_subscription` = 1) show a noticeably higher recovery rate compared to one-off payments.

## Important Patterns
- **Historical Success Rate**: Shows a very strong positive correlation with recovery. Users who historically paid successfully are observed to be much more likely to recover.
- **Days Overdue & Recovery Attempts**: Higher values of these features are associated with a decreased probability of recovery.
- **Failure Reason**: `technical_error` has the highest recovery rate, followed by `insufficient_funds`. `invalid_card` has the lowest.

## Potentially Useful Features
Based on EDA, the following features appear predictive for the ML model:
- `failure_reason`
- `historical_success_rate`
- `customer_tenure_months`
- `days_overdue`
- `recovery_attempts_so_far`
- `is_subscription`
- `past_failed_payments`
- `time_since_last_success_days`

## Features Excluded From Modeling
- `payment_id`: High cardinality unique identifier, carries no predictive signal.
- `customer_id`: Unique identifier, should be excluded from predictive features to prevent overfitting to specific users.

## Leakage Checks
- No single feature perfectly predicts the target.
- The highest correlation with the target among numerical features is `historical_success_rate` (approx 0.35 - 0.40), which is strong but realistic and does not indicate target leakage.
- All features analyzed are logically available at the time of failure.

## Conclusions for Model Training
1. **Tree-Based Models are suitable**: Given the mix of numerical, categorical, and right-skewed data (e.g. amount), models like LightGBM and XGBoost are ideal choices.
2. **Feature Scaling**: Not strictly necessary for tree models, but may be useful if evaluating Logistic Regression as a baseline.
3. **Categorical Encoding**: `failure_reason` and `payment_method` will need encoding (One-Hot or target encoding) for scikit-learn baselines.
4. **Metrics**: Given the slight imbalance, ROC-AUC and PR-AUC will be more informative than raw accuracy.
