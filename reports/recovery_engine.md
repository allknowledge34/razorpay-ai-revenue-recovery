# Revenue Recovery Intelligence

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
