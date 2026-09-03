# Business-Adjusted Recovery Prioritization Benchmark

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
- **Total Payments**: 20,000
- **Total Revenue at Risk**: ₹16,644,382.52
- **Total Base Expected Recovery**: ₹7,183,345.40
- **Total Business-Adjusted Expected Recovery**: ₹9,889,813.39

### Tier Distribution
| Tier | Count |
|---|---|
| CRITICAL | 16 |
| HIGH | 571 |
| MEDIUM | 4,972 |
| LOW | 14,441 |

### Value Concentration
- **Top 1% of queue**: ₹1,222,991.89 (12.4%)
- **Top 5% of queue**: ₹3,298,431.07 (33.4%)
- **Top 10% of queue**: ₹4,777,574.07 (48.3%)
- **Top 20% of queue**: ₹6,555,260.01 (66.3%)

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
