# Batch Recovery Benchmark & Selection Evidence

## 1. Benchmark Objective
Evaluate four recovery approaches on the same synthetic payment batch using strict deterministic randomness to provide reproducible, evidence-based comparisons.

## 2. Dataset Description
- **Source**: Synthetic failed payments dataset (`data/failed_payments_scored.csv`).
- **Total Payments**: 20,000
- **Revenue at Risk**: ₹16,644,382.52
- **Random Seed**: Stable SHA-256 hash of `payment_id`.

**Disclaimer:** Benchmark results are generated from the project's synthetic payment dataset and simulation assumptions. They do not represent real Razorpay recovery performance, and do not reflect real causal outcomes.

## 3. Strategy Definitions
- **A. Blind Retry**: Retries every payment.
- **B. Current Rule-Based**: High probability -> Retry, Medium -> Reminder, Low -> Stop.
- **C. Optimized Selective**: Threshold optimized to maximize expected net revenue (Threshold = 0.05).
- **D. Bounded Recovery Orchestrator**: Stage 18 logic enforcing maximum attempts, viability limits, and manual review routing.

## 4. Benchmark Results
| Strategy | Net Recovered | Action Cost | Recovery Rate | ROI | Retries |
|---|---|---|---|---|---|
| A. Blind Retry | ₹6,112,278.37 | ₹1,000,000.00 | 42.82% | 6.11 | 20000 |
| B. Current Rule-Based | ₹4,537,942.23 | ₹279,953.00 | 28.73% | 16.21 | 5491 |
| C. Optimized Selective | ₹6,143,459.12 | ₹926,100.00 | 42.59% | 6.63 | 18522 |
| D. Bounded Recovery Orchestrator | ₹2,877,315.10 | ₹181,915.00 | 18.20% | 15.82 | 3582 |

## 5. Comparative Analysis (Deltas)
| Comparison | Net Rev Delta | Action Cost Delta | Rec Rate Delta | ROI Delta |
|---|---|---|---|---|
| C. Optimized Selective vs A. Blind Retry | ₹31,180.75 | ₹-73,900.00 | -0.24% | 0.52 |
| D. Bounded Recovery Orchestrator vs A. Blind Retry | ₹-3,234,963.27 | ₹-818,085.00 | -24.63% | 9.70 |
| D. Bounded Recovery Orchestrator vs B. Current Rule-Based | ₹-1,660,627.13 | ₹-98,038.00 | -10.53% | -0.39 |
| D. Bounded Recovery Orchestrator vs C. Optimized Selective | ₹-3,266,144.02 | ₹-744,185.00 | -24.39% | 9.18 |

## 6. Guardrail Metrics (Bounded Recovery Orchestrator)
- **Maximum Existing Attempts Observed**: 3
- **Configured Maximum Automatic Attempts**: 2
- **Stopped (Probability)**: 347
- **Stopped (Economic)**: 1226
- **Stopped (Max Attempts)**: 9925
- **Routed to Manual Review**: 0
- **Invalid State Transitions**: 0
- **Outcomes VERIFIED**: 8502
- **Outcomes CLOSED**: 3639
- **Outcomes STOPPED**: 16361
- **Outcomes MANUAL_REVIEW**: 0


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
