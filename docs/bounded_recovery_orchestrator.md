# Bounded Recovery Orchestrator

## Overview
This stage implements a synthetic bounded recovery workflow. It does not execute real payments or represent real Razorpay customer recovery outcomes.

The Bounded Recovery Orchestrator coordinates machine learning intelligence with strict deterministic business rules and synthetic simulation. It prevents unbounded retry loops, respects effective retry costs, routes high-value transactions to manual review, and simulates final transaction resolution.

## Recovery Policy Rules
The `BoundedRecoveryPolicy` enforces the following parameters:
- **Minimum Retry Probability (`0.05`)**: Transactions below this likelihood are automatically stopped to prevent wasted cost.
- **Maximum Recovery Attempts (`2`)**: Strict ceiling on automatic retries per payment.
- **High-Value Threshold (`25000`)**: Payments over this threshold bypass automatic recovery and require manual assessment.
- **Effective Retry Cost (`50`)**: Automatic retries are blocked if the expected recovery value is less than the cost of action.

## State Machine
The workflow relies on a deterministic state machine to ensure valid operational boundaries:
1. `FAILED` (Initial)
2. `ASSESSED` (Policy evaluation)
3. `ACTION_SELECTED` / `STOPPED` / `MANUAL_REVIEW`
4. `ACTION_EXECUTED` (Simulation triggered)
5. `RECOVERED` or `FAILED_RECOVERY`
6. `VERIFIED`
7. `CLOSED` (Success) or `STOPPED` (Failure exhausted)

## Simulated Execution and Outcome Verification
Automatic actions are evaluated against a synthetic effectiveness matrix:
- **Retry Payment**: Cost ₹50, Multiplier 1.0
- **Payment Method Reminder**: Cost ₹1, Multiplier 0.5
- **Manual Review**: Cost ₹100, Multiplier 0.75

A reproducible stochastic function (seeded by a stable SHA-256 hash of `payment_id` for process-level reproducibility) simulates the final state. The `verification_status` is then set to `VERIFIED`.

## PostgreSQL Persistence
When a valid Database URL is configured, outcomes are committed directly to the `recovery_outcomes` table (via `outcome_source = 'SIMULATED'`). The system gracefully degrades to a stateless presentation mode when the database is offline.

## Auditability
The `RecoveryOrchestrator` integrates with the `AuditTrail`, distinguishing between the model's recommendation and the orchestrator's explicit execution limits.
