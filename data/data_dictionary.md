# Data Dictionary: Failed Payments Recovery Dataset

This dataset represents a simulated snapshot of failed payments and historical customer behavior, known exactly at the time we attempt to predict if a payment will be recovered.

| Feature Name | Meaning | Data Type | Realistic Range / Categories | Known at Prediction Time? | Why it may help predict recovery |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `payment_id` | Unique identifier for the failed payment | String | `pay_000001` onwards | Yes | N/A (Identifier) |
| `customer_id` | Unique identifier for the customer | String | `cust_10000` to `cust_99999` | Yes | N/A (Identifier, though could be used for grouping) |
| `payment_amount` | The monetary amount of the failed payment | Float | 10.0 to 50000.0 | Yes | Larger amounts might be harder to recover instantly, or might prompt more urgent customer action. |
| `failure_reason` | The reason provided by the payment gateway for the failure | Categorical | `insufficient_funds`, `invalid_card`, `technical_error`, `limit_exceeded` | Yes | Technical errors often auto-recover on retry. Invalid cards require customer intervention (lower recovery). |
| `payment_method` | The method used for the payment attempt | Categorical | `credit_card`, `debit_card`, `upi`, `bank_transfer` | Yes | Different methods have different retry success rates and friction. |
| `customer_tenure_months` | Number of months the customer has been active | Integer | 1 to 60 | Yes | Loyal, long-term customers are usually more likely to update payment details and recover. |
| `past_successful_payments` | Count of previously successful payments from this customer | Integer | 0+ | Yes | Indicates a history of good payment behavior. |
| `past_failed_payments` | Count of previously failed payments from this customer | Integer | 0+ | Yes | Indicates chronic payment issues or bad behavior. |
| `historical_success_rate` | Ratio of successful payments to total past payments | Float | 0.0 to 1.0 | Yes | A strong direct signal of customer reliability. |
| `time_since_last_success_days` | Days elapsed since the customer's last successful payment | Integer | 1 to 365 | Yes | A recent success implies active usage and valid payment methods. |
| `is_subscription` | Whether the failed payment is part of a recurring subscription | Integer (Binary)| 0 (No), 1 (Yes) | Yes | Subscriptions often imply intent to continue service; users may proactively fix failures to avoid service disruption. |
| `days_overdue` | Number of days since the payment originally failed/was due | Integer | 1 to 30 | Yes | Payments overdue for a long time have a drastically lower chance of recovery. |
| `recovery_attempts_so_far` | Number of times we have already tried to retry/recover this specific payment | Integer | 0 to 4 | Yes | Diminishing returns; if it failed 3 times already, the 4th attempt is less likely to work. |
| `recovered` (TARGET) | Whether the payment was eventually recovered | Integer (Binary)| 0 (Not Recovered), 1 (Recovered) | NO (Predicted) | This is the variable we are trying to predict. |

## Leakage Prevention
Features purposefully excluded to prevent target leakage:
- `recovery_timestamp`: Known only after recovery.
- `final_payment_status`: Is basically the target.
- `recovered_amount`: Leaks the target perfectly.
- `future_retry_count`: Known only in the future.
