# Outcome Simulation Analysis

## 1. Objective
Build a controlled closed-loop simulation that connects:
failed payment -> model recovery probability -> recovery decision/strategy -> simulated recovery outcome -> simulated recovered revenue -> cost -> net recovered revenue -> business impact metrics -> strategy comparison.

## 2. Synthetic Simulation Disclaimer
**IMPORTANT:** This is NOT real payment execution.
- The outcome economics in this stage are explicitly SYNTHETIC SIMULATION ASSUMPTIONS.
- The recovery probabilities are model estimates based on a synthetic dataset, and expected recovery is not guaranteed actual recovered revenue.
- This simulation does not call any real payment API, nor does it represent actual Razorpay or customer metrics.

## 3. Input Dataset
The simulator operates on the already scored synthetic payment dataset (`data/failed_payments_scored.csv`).

## 4. Model Probability Usage
The model produces a `recovery_probability` for each payment, estimating the likelihood of recovery if optimally actioned. 

## 5. Action Effectiveness Assumptions
We apply reasonable, explicit synthetic assumptions for the effectiveness of each recovery action:
- **Retry Payment:** 1.00 (Full estimated probability)
- **Payment Method Reminder:** 0.50 (Half the effectiveness of a direct retry)
- **Manual Review / Stop Automatic Retry:** 0.75 (Lower effectiveness than a direct retry)
- **Do Nothing:** 0.00 (Zero probability of recovery)

`effective_recovery_probability = min(recovery_probability * action_effectiveness, 1.0)`

## 6. Cost Assumptions
We reuse the Stage 8 economic cost assumptions:
- **Retry Payment:** ₹50.0 (₹5 base cost + ₹45 friction cost)
- **Payment Method Reminder:** ₹1.0
- **Manual Review / Stop Automatic Retry:** ₹100.0
- **Do Nothing:** ₹0.0

## 7. Simulation Methodology
- `simulated_recovered` is a Bernoulli trial determined by `effective_recovery_probability`.
- If `simulated_recovered == 1`: `simulated_recovered_revenue = payment_amount`
- Else: `simulated_recovered_revenue = 0`
- `net_recovered_revenue = simulated_recovered_revenue - action_cost`

## 8. Random Seed & Control Methodology
To ensure a controlled, fair strategy comparison:
- We use a deterministic local NumPy random generator: `rng = np.random.default_rng(42)`.
- A single array of uniformly distributed random numbers is generated: `uniform_draws = rng.uniform(0, 1, size=len(df))`.
- Each strategy uses the *same* `uniform_draws` to evaluate the Bernoulli trial (`uniform_draws < effective_recovery_probability`). This isolates the effect of the strategy logic from random sampling noise.

## 9. Strategy Definitions
- **A. Blind Retry:** Every payment receives "Retry Payment".
- **B. Current Rule-Based Strategy:** Based on probability thresholds (Retry >= 0.65, Reminder >= 0.35, else Manual Review).
- **C. Optimized Selective Strategy:** Sweeps thresholds to find the optimum, then applies "Retry Payment" to those above the threshold, and "Do Nothing" otherwise.

## 10. Business Impact Metrics
For each strategy, we evaluate aggregate metrics such as:
- Total Payments, Retry Count, Reminder Count, Manual Review Count
- Retry Rate, Simulated Recovery Rate
- Gross Simulated Recovered Revenue, Total Action Cost, Net Recovered Revenue
- Revenue Recovery Rate, Recovery Cost per Recovered Payment, ROI

## 11. Strategy Comparison
See `reports/outcome_strategy_comparison.csv` for the detailed tabular output of all three strategies across the evaluated business metrics.
Under the synthetic cost and action-effectiveness assumptions, the Optimized Selective Strategy reduces simulated action costs while maintaining similar simulated recovery performance, resulting in higher simulated net recovered revenue and ROI. These are simulated outcomes based on synthetic assumptions, and no actual Razorpay customer metric or recovery is represented.

## 12. Limitations
- The simulated outcomes are entirely based on synthetic Bernoulli trials driven by assumed action effectiveness multipliers.
- The costs are illustrative configurations.

## 13. Production Validation Requirements
In a real production environment, this simulation would need to be replaced (or calibrated) with actual A/B testing data (e.g., Holdout vs. Treatment groups) to measure the true causal uplift of each action, the actual empirical cost of customer friction, and the real-world operational costs.
