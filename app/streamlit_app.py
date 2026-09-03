import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.domain.recovery_engine import RecoveryEngine
from src.domain.recovery_strategy import RecoverySimulator
from src.services.explanation.decision_trace import DecisionTracer
from src.services.inference.inference_service import RecoveryInferenceService
from src.domain.prioritization import RecoveryPrioritizer

st.set_page_config(page_title="AI Revenue Recovery Engine", layout="wide")

@st.cache_resource
def load_engine():
    return RecoveryEngine()

@st.cache_resource
def load_simulator():
    return RecoverySimulator()

@st.cache_resource
def load_tracer():
    return DecisionTracer(simulator_cost=50.0, simulator_threshold=0.05)

@st.cache_resource
def load_inference_service():
    return RecoveryInferenceService()

@st.cache_resource
def load_prioritizer():
    return RecoveryPrioritizer()

def main():
    st.title("AI Revenue Recovery Engine")
    st.markdown("**Predicting failed-payment recovery probability and prioritizing economically viable recovery actions.**")
    
    st.markdown("""
Failed Payment → Predict Recovery Probability → Estimate Expected Recovery → Select Recovery Strategy → Apply Guardrails → Prioritize Recovery Queue → Measure Outcome

*(This is a synthetic-data / simulation-based prototype. It does not contain real Razorpay customer data or execute live payments.)*
    """)
    
    st.header("Demo Overview")
    try:
        df_scored = pd.read_csv('data/failed_payments_scored.csv')
        total_payments = len(df_scored)
        total_risk = df_scored['payment_amount'].sum()
        
        df_bench = pd.read_csv('reports/recovery_benchmark.csv')
        best_strat = df_bench[df_bench['Strategy'] == 'Strategy C: Threshold Optimized']
        if not best_strat.empty:
            rec_rate = best_strat['Recovery Rate'].values[0]
            expected_rev = best_strat['Net Recovered Revenue'].values[0]
        else:
            rec_rate = 0.0
            expected_rev = 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Failed Payments", f"{total_payments:,}")
        c2.metric("Revenue at Risk", f"₹{total_risk:,.2f}")
        c3.metric("Simulated Recovery Rate", f"{rec_rate:.2%}")
        c4.metric("Expected Net Revenue", f"₹{expected_rev:,.2f}")
    except Exception as e:
        st.warning("Demo Overview data is currently unavailable. Run the Batch Benchmark first.")

    with st.expander("How It Works"):
        st.markdown("""
1. **DETECT**: Identify failed payments and validate inference inputs.
2. **PREDICT**: Estimate probability of eventual recovery.
3. **DECIDE**: Compare expected recovery with action economics and policy constraints.
4. **PRIORITIZE**: Rank recoverable business value so limited operational attention goes to the highest-value cases. *(Note: Prioritization controls queue order while bounded recovery policy controls whether an action is allowed.)*
5. **VERIFY**: Record decisions/outcomes through the audit and monitoring layers.
        """)
        
    with st.expander("Prototype assumptions & limitations"):
        st.markdown("""
- Dataset is synthetic.
- Recovery outcomes are simulated.
- Action costs and recovery multipliers are simulation assumptions.
- No real Razorpay payment execution occurs.
- PostgreSQL persistence is optional depending on DATABASE_URL.
- Model probabilities are estimates, not guarantees.
- Business prioritization is a queue-ordering layer, not an authorization layer.
        """)
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "Real-Time Inference", 
        "Recovery Prioritization",
        "Recovery Benchmark",
        "Strategy Simulator", 
        "Batch Analysis", 
        "Outcome Simulation",
        "Drift Monitoring",
        "Audit Trail", 
        "Database Status"
    ])
    
    try:
        engine = load_engine()
        simulator = load_simulator()
        tracer = load_tracer()
        service = load_inference_service()
        prioritizer = load_prioritizer()
    except Exception as e:
        st.error(f"Failed to load required services: {e}")
        return

    with tab1:
            st.header("Real-Time Inference")
            st.info("Demonstrate the complete inference pipeline for a single simulated failed payment.")

            try:
                from src.services.inference.inference_service import RecoveryInferenceService
                service = RecoveryInferenceService()

                st.subheader("Simulate Incoming Event")

                c1, c2, c3 = st.columns(3)
                with c1:
                    amt = st.number_input("Payment Amount", min_value=1.0, value=5000.0, key='rt_amt')
                    is_sub = st.selectbox("Is Subscription?", [0, 1], key='rt_is_sub')
                    pm = st.selectbox("Payment Method", ["credit_card", "debit_card", "upi", "bank_transfer"], key='rt_pm')
                    fr = st.selectbox("Failure Reason", ["insufficient_funds", "invalid_card", "technical_error", "limit_exceeded"], key='rt_fr')
                with c2:
                    tenure = st.number_input("Customer Tenure (months)", min_value=0, value=12, key='rt_tenure')
                    successes = st.number_input("Past Successful Payments", min_value=0, value=10, key='rt_succ')
                    fails = st.number_input("Past Failed Payments", min_value=0, value=0, key='rt_fail')
                    rate = st.number_input("Historical Success Rate", min_value=0.0, max_value=1.0, value=1.0, key='rt_rate')
                with c3:
                    last_succ = st.number_input("Days Since Last Success", min_value=0, value=5, key='rt_last_succ')
                    overdue = st.number_input("Days Overdue", min_value=0, value=1, key='rt_overdue')
                    attempts = st.number_input("Recovery Attempts So Far", min_value=0, value=0, key='rt_attempts')

                if st.button("Submit Event"):
                    with st.spinner("Processing event..."):
                        event = {
                            "payment_id": "rt_test_001",
                            "payment_amount": amt,
                            "is_subscription": is_sub,
                            "payment_method": pm,
                            "failure_reason": fr,
                            "customer_tenure_months": tenure,
                            "past_successful_payments": successes,
                            "past_failed_payments": fails,
                            "historical_success_rate": rate,
                            "time_since_last_success_days": last_succ,
                            "days_overdue": overdue,
                            "recovery_attempts_so_far": attempts
                        }

                        res = service.predict_event(event)

                        if res['processing_metadata']['status'] == 'error':
                            st.error(f"Inference failed ({res['processing_metadata']['processing_time_ms']}ms)")
                            st.json(res['validation']['errors'] if res['validation']['errors'] else res['processing_metadata'])
                        else:
                            st.success(f"Inference complete in {res['processing_metadata']['processing_time_ms']}ms")

                            # Get priority value
                            try:
                                from src.domain.prioritization import RecoveryPrioritizer
                                prioritizer = RecoveryPrioritizer()
                                pri_res = prioritizer.calculate_priority(
                                    "rt_test_001",
                                    amt,
                                    res['prediction']['recovery_probability'],
                                    is_sub
                                )
                            except:
                                pri_res = {"priority_value": 0.0, "priority_tier": "LOW"}

                            st.subheader("Why this decision?")
                        
                            col_mdl, col_eco, col_pol, col_pri = st.columns(4)
                        
                            with col_mdl:
                                st.markdown("**MODEL ESTIMATE**")
                                st.write(f"Recovery Probability: **{res['prediction']['recovery_probability']:.2%}**")
                            
                            with col_eco:
                                st.markdown("**ECONOMIC ESTIMATE**")
                                st.write(f"Expected Recovery: **₹{res['economic_estimate']['expected_recovery']:,.2f}**")
                                if 'bounded_recovery' in res:
                                    st.write(f"Effective Retry Cost: **₹{res['bounded_recovery'].get('action_cost', 50.0):,.2f}**")
                                    st.write(f"Expected Net Value: **₹{res['economic_estimate']['expected_recovery'] - res['bounded_recovery'].get('action_cost', 50.0):,.2f}**")
                                else:
                                    st.write("Effective Retry Cost: **N/A**")
                                    st.write("Expected Net Value: **N/A**")
                                
                            with col_pol:
                                st.markdown("**POLICY DECISION**")
                                st.write(f"Recommended Action: **{res['decision']['recommended_action']}**")
                                if 'bounded_recovery' in res:
                                    st.write(f"Policy: **{res['bounded_recovery']['policy_decision']}**")
                            
                            with col_pri:
                                st.markdown("**BUSINESS PRIORITY**")
                                st.write(f"Business-Adjusted EV: **₹{pri_res['priority_value']:,.2f}**")
                                st.write(f"Priority Tier: **{pri_res['priority_tier']}**")
                            
                            st.divider()
                        
                            if 'bounded_recovery' in res:
                                br = res['bounded_recovery']
                                st.subheader("Bounded Recovery Workflow")
                                st.write(f"Policy Decision: **{br['policy_decision']}**")
                                if br['policy_decision'] == 'BLOCKED':
                                    st.write(f"Reason: {br['decision_reason']}")
                                st.write(f"Action: **{br['selected_action']}**")
                                st.write(f"Attempt: **{br['attempt_number']}**")
                                st.write(f"Outcome: **{br.get('simulated_recovered', False) and 'RECOVERED' or 'FAILED_RECOVERY'}**" if br['policy_decision'] == 'ALLOWED' else "")
                                st.write(f"Recovered Amount: **₹{br.get('recovered_amount', 0.0):,.2f}**")
                                st.write(f"Action Cost: **₹{br.get('action_cost', 0.0):,.2f}**")
                                st.write(f"Net Recovered Revenue: **₹{br.get('net_recovered_revenue', 0.0):,.2f}**")
                                st.write(f"Verification: **{br.get('verification_status', 'N/A')}**")
                                st.write(f"Final State: **{br.get('final_state', 'N/A')}**")

                                with st.expander("State History"):
                                    st.write(" → ".join(br.get('state_history', [])))


                            with st.expander("View Full Inference Response Payload"):
                                st.json(res)


            except Exception as e:
                st.error(f"Failed to load Inference Service: {str(e)}")
    with tab2:
            st.header("Recovery Prioritization Queue")
            st.info("Prioritization determines queue order; bounded recovery policy still controls whether an action is allowed. (ML Inference -> Recovery Probability -> Prioritization Layer -> Priority Queue -> Recovery Orchestrator)")
            try:
                from src.domain.prioritization import RecoveryPrioritizer
                prioritizer = RecoveryPrioritizer()
                df_scored = pd.read_csv('data/failed_payments_scored.csv')
                df_prioritized = prioritizer.batch_prioritize(df_scored)
            
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Payments", f"{len(df_prioritized):,}")
                c2.metric("Revenue at Risk", f"₹{df_prioritized['payment_amount'].sum():,.2f}")
                c3.metric("Base Expected", f"₹{df_prioritized['base_expected_recovery'].sum():,.2f}")
                c4.metric("Business Adjusted", f"₹{df_prioritized['business_adjusted_expected_recovery'].sum():,.2f}")
            
                st.subheader("Tier Distribution")
                st.bar_chart(df_prioritized['priority_tier'].value_counts())
            
                st.subheader("Top Prioritized Recovery Opportunities")
                st.dataframe(df_prioritized[[
                    'priority_rank', 'priority_tier', 'payment_id', 'priority_value', 
                    'payment_amount', 'recovery_probability', 'is_subscription', 'priority_explanation'
                ]].head(100))
            
                st.caption("The priority value is a business-value ranking derived from model-estimated recovery probability and synthetic economic assumptions. It is not a causal risk score or a guarantee of recovered revenue.")
            except Exception as e:
                st.error(f"Error loading prioritization logic: {e}")
    with tab3:
            st.header("Batch Recovery Benchmark")
            st.warning("Benchmark results are generated from the project's synthetic payment dataset and simulation assumptions. They do not represent real Razorpay recovery performance.")
        
            if st.button("Run Batch Benchmark", key="run_benchmark_btn"):
                with st.spinner("Evaluating strategies on 20,000 synthetic payments..."):
                    try:
                        from src.ml.evaluation.recovery_benchmark import RecoveryBenchmark
                        bench = RecoveryBenchmark()
                        df_bench = bench.evaluate_all()
                    
                        st.subheader("Strategy Comparison")
                        st.dataframe(
                            df_bench[['Strategy', 'Net Recovered Revenue', 'Recovery Rate', 'Action Cost', 'ROI', 'Recovery Efficiency', 'Retry Count', 'Manual Review Count', 'Stopped Count']].style.format({
                                'Net Recovered Revenue': '₹{:,.2f}',
                                'Action Cost': '₹{:,.2f}',
                                'Recovery Rate': '{:.2%}',
                                'ROI': '{:.2f}',
                                'Recovery Efficiency': '{:.2%}'
                            })
                        )
                    
                        st.info("The Bounded Recovery Orchestrator rigidly enforces business guardrails (e.g. Max Attempts), which safely suppresses action costs and prevents unbounded retries, demonstrating configurable recovery limits in this synthetic simulation.")
                    except Exception as e:
                        st.error(f"Benchmark failed: {str(e)}")
    with tab4:
            st.header("Cost-Aware Strategy Simulator")
            st.markdown("""
            **How the simulator decides:**
            The simulator selects a retry threshold that maximizes expected net recovery under the selected cost assumption.
            *(Recovery Probability → Expected Recovery → Action Cost → Expected Net Recovery → Optimal Retry Threshold)*
            """)
            st.caption("Simulation assumptions — not actual Razorpay production economics.")

            # Sidebar / Controls
            st.sidebar.header("Simulator Controls")

            # Presets
            preset = st.sidebar.selectbox("Cost Scenario Presets", ["Custom", "Low Cost (₹10)", "Base Cost (₹50)", "High Cost (₹250)"], index=2)

            default_cost = 50.0
            if preset == "Low Cost (₹10)":
                default_cost = 10.0
            elif preset == "Base Cost (₹50)":
                default_cost = 50.0
            elif preset == "High Cost (₹250)":
                default_cost = 250.0

            effective_cost = st.sidebar.number_input("Effective Retry Cost (₹)", min_value=0.0, max_value=1000.0, value=default_cost, step=5.0)

            # Calculate optimal threshold for current effective cost
            df_sweep, opt_row = simulator.threshold_sweep(effective_cost)
            optimal_threshold = opt_row['threshold']

            threshold = st.sidebar.slider("Probability Threshold", min_value=0.0, max_value=1.0, value=float(optimal_threshold), step=0.01)

            # Live Selective Strategy Calculation
            strat_c = simulator.evaluate_strategy_c_selective(threshold, effective_cost)
            strat_a = simulator.evaluate_strategy_a_blind_retry()
            # Ensure we pass the updated effective cost to strategy B calculation?
            # Strategy A and B use simulator.costs internally, which hasn't been updated dynamically by the slider.
            # Let's temporarily override simulator costs for live calculation so it perfectly matches.
            orig_retry_cost = simulator.costs['retry_cost']
            orig_friction_cost = simulator.costs['customer_friction_cost']

            # We assign the whole effective_cost to retry_cost and 0 to friction for A and B calculation
            simulator.costs['retry_cost'] = effective_cost
            simulator.costs['customer_friction_cost'] = 0.0

            strat_a = simulator.evaluate_strategy_a_blind_retry()
            strat_b = simulator.evaluate_strategy_b_rule_based()

            # Restore simulator costs
            simulator.costs['retry_cost'] = orig_retry_cost
            simulator.costs['customer_friction_cost'] = orig_friction_cost

            # Key Business Metrics
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Expected Net Recovery", f"₹{strat_c['expected_net_recovery']:,.2f}",
                      f"{(strat_c['expected_net_recovery'] - strat_a['expected_net_recovery']):,.2f} vs Blind")
            k2.metric("Expected Gross Recovery", f"₹{strat_c['expected_recovery']:,.2f}")
            k3.metric("Retry Rate", f"{strat_c['retry_percentage']:.1f}%",
                      f"{strat_c['retry_percentage'] - 100.0:.1f}% vs Blind")
            k4.metric("Action Cost", f"₹{strat_c['recovery_cost']:,.2f}")

            st.divider()

            # Compare Three Strategies
            st.subheader("Strategy Comparison")

            compare_data = [
                {"Strategy": "Blind Retry", "Retry %": f"{strat_a['retry_percentage']:.1f}%",
                 "Expected Recovery": f"₹{strat_a['expected_recovery']:,.2f}",
                 "Action Cost": f"₹{strat_a['action_cost']:,.2f}",
                 "Expected Net Recovery": f"₹{strat_a['expected_net_recovery']:,.2f}"},

                {"Strategy": "Current Rule-Based", "Retry %": f"{strat_b['retry_percentage']:.1f}%",
                 "Expected Recovery": f"₹{strat_b['expected_recovery']:,.2f}",
                 "Action Cost": f"₹{strat_b['action_cost']:,.2f}",
                 "Expected Net Recovery": f"₹{strat_b['expected_net_recovery']:,.2f}"},

                {"Strategy": "Selective Recovery", "Retry %": f"{strat_c['retry_percentage']:.1f}%",
                 "Expected Recovery": f"₹{strat_c['expected_recovery']:,.2f}",
                 "Action Cost": f"₹{strat_c['recovery_cost']:,.2f}",
                 "Expected Net Recovery": f"₹{strat_c['expected_net_recovery']:,.2f}"}
            ]

            st.table(pd.DataFrame(compare_data).set_index("Strategy"))

            st.divider()

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Threshold Optimization Curve")
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(df_sweep['threshold'], df_sweep['expected_net_recovery'], marker='.', lw=2, color='royalblue')
                ax.axvline(x=optimal_threshold, color='green', linestyle='--', label=f"Optimal: {optimal_threshold:.2f}")
                ax.axvline(x=threshold, color='red', linestyle=':', label=f"Selected: {threshold:.2f}")
                ax.set_title("Probability Threshold vs Expected Net Recovery")
                ax.set_xlabel("Probability Threshold")
                ax.set_ylabel("Expected Net Recovery (₹)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

            with c2:
                st.subheader("Cost Sensitivity Presets")
                st.markdown("Optimal thresholds recalculate dynamically based on action costs.")

                # Use threshold_sweep dynamically
                _, low_row = simulator.threshold_sweep(10.0)
                _, base_row = simulator.threshold_sweep(50.0)
                _, high_row = simulator.threshold_sweep(250.0)

                sens_data = [
                    {"Cost Scenario": "Low Cost (₹10)", "Optimal Threshold": f"{low_row['threshold']:.2f}",
                     "Retry %": f"{low_row['retry_percentage']:.1f}%", "Net Recovery": f"₹{low_row['expected_net_recovery']:,.2f}"},
                    {"Cost Scenario": "Base Cost (₹50)", "Optimal Threshold": f"{base_row['threshold']:.2f}",
                     "Retry %": f"{base_row['retry_percentage']:.1f}%", "Net Recovery": f"₹{base_row['expected_net_recovery']:,.2f}"},
                    {"Cost Scenario": "High Cost (₹250)", "Optimal Threshold": f"{high_row['threshold']:.2f}",
                     "Retry %": f"{high_row['retry_percentage']:.1f}%", "Net Recovery": f"₹{high_row['expected_net_recovery']:,.2f}"}
                ]
                st.table(pd.DataFrame(sens_data).set_index("Cost Scenario"))
    with tab5:
            st.header("Batch Recovery Analysis")

            try:
                # Use cached scored dataset
                file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'failed_payments_scored.csv'))
                if not os.path.exists(file_path):
                    st.warning("Scored dataset not found. Please complete Stage 5 to generate batch predictions.")
                    return

                df_scored = pd.read_csv(file_path)

                st.subheader("Overall Portfolio Impact")
                total_failed = len(df_scored)
                total_value = df_scored['payment_amount'].sum()
                total_expected = df_scored['expected_recovery'].sum()

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Failed Payments", f"{total_failed:,}")
                c2.metric("Total Payment Value", f"₹{total_value:,.2f}")
                c3.metric("Expected Recoverable Revenue", f"₹{total_expected:,.2f}")

                st.subheader("Priority Distribution")
                p1, p2, p3 = st.columns(3)
                counts = df_scored['priority'].value_counts()
                p1.metric("HIGH Priority Count", counts.get("HIGH", 0))
                p2.metric("MEDIUM Priority Count", counts.get("MEDIUM", 0))
                p3.metric("LOW Priority Count", counts.get("LOW", 0))

                st.divider()

                st.subheader("Visualizations")
                vcol1, vcol2 = st.columns(2)

                with vcol1:
                    st.markdown("**Priority Distribution**")
                    # Ensure categorical ordering
                    chart_data = pd.DataFrame(counts).reindex(['HIGH', 'MEDIUM', 'LOW']).fillna(0)
                    st.bar_chart(chart_data)

                with vcol2:
                    st.markdown("**Expected Recovery by Priority**")
                    expected_by_priority = df_scored.groupby('priority')['expected_recovery'].sum().reindex(['HIGH', 'MEDIUM', 'LOW']).fillna(0)
                    st.bar_chart(expected_by_priority)

                st.subheader("Scored Payments Preview")
                st.dataframe(df_scored[['payment_amount', 'failure_reason', 'recovery_probability', 'expected_recovery', 'priority', 'recommended_action']].head(100))

            except Exception as e:
                st.error(f"Could not load batch data. Error: {e}")
    with tab6:
            st.header("Synthetic Outcome Simulation — Not Real Payment Execution")
            st.info("Disclaimer: This is a synthetic simulation of recovery outcomes based on assumed action effectiveness and costs. It does not represent actual Razorpay recovery execution, real customer behavior, or real payment processing.")

            try:
                from src.services.outcome.outcome_simulator import OutcomeSimulator

                st.markdown("### Methodology")
                st.write("Model Probability → Action Effectiveness → Synthetic Outcome → Revenue → Cost → Net Impact")
                st.write("We evaluate the recovery strategy on the existing synthetic scored dataset using a fixed deterministic seed (42).")

                if st.button("Run Closed-Loop Simulation"):
                    with st.spinner("Simulating deterministic outcomes..."):
                        outcome_sim = OutcomeSimulator(seed=42)
                        strategies = ["Blind Retry", "Current Rule-Based Strategy", "Optimized Selective Strategy"]

                        metrics_list = []
                        for strat in strategies:
                            df_sim = outcome_sim.simulate_strategy(strat)
                            metrics_list.append(outcome_sim.calculate_metrics(df_sim, strat))

                        df_metrics = pd.DataFrame(metrics_list)

                        st.subheader("Strategy Comparison (Aggregate Business Impact)")

                        # Display the metrics beautifully
                        display_cols = [
                            'strategy', 'simulated_recovery_rate', 'gross_simulated_recovered_revenue',
                            'total_action_cost', 'net_recovered_revenue', 'roi',
                            'recovery_cost_per_recovered_payment', 'retry_count', 'reminder_count', 'manual_review_count'
                        ]

                        df_display = df_metrics[display_cols].copy()
                        # Formatting
                        df_display['simulated_recovery_rate'] = df_display['simulated_recovery_rate'].map("{:.1%}".format)
                        df_display['gross_simulated_recovered_revenue'] = df_display['gross_simulated_recovered_revenue'].map("₹{:,.2f}".format)
                        df_display['total_action_cost'] = df_display['total_action_cost'].map("₹{:,.2f}".format)
                        df_display['net_recovered_revenue'] = df_display['net_recovered_revenue'].map("₹{:,.2f}".format)
                        df_display['roi'] = df_display['roi'].map("{:.2f}x".format)
                        df_display['recovery_cost_per_recovered_payment'] = df_display['recovery_cost_per_recovered_payment'].map("₹{:,.2f}".format)

                        st.dataframe(df_display.set_index('strategy'), use_container_width=True)

                        # Visualizations
                        st.subheader("Simulated Net Recovered Revenue by Strategy")
                        fig, ax = plt.subplots(figsize=(10, 5))

                        strategies = df_metrics['strategy'].tolist()
                        net_rev = df_metrics['net_recovered_revenue'].tolist()

                        ax.bar(strategies, net_rev, color=['#440154', '#21918c', '#fde725'])
                        ax.set_title("Simulated Net Recovered Revenue")
                        ax.set_ylabel("Amount (₹)")
                        st.pyplot(fig)

                        st.subheader("Simulated ROI by Strategy")
                        fig2, ax2 = plt.subplots(figsize=(10, 5))
                        roi_vals = df_metrics['roi'].tolist()
                        ax2.bar(strategies, roi_vals, color=['#0d0887', '#cc4678', '#f0f921'])
                        ax2.set_title("Simulated Return on Investment (ROI)")
                        ax2.set_ylabel("ROI Multiple")
                        st.pyplot(fig2)

            except Exception as e:
                st.error(f"Failed to load Outcome Simulator: {str(e)}")
    with tab7:
            st.header("Synthetic / Offline Monitoring")
            st.info("Disclaimer: This is a synthetic/offline monitoring framework demonstrated using synthetic data. It does not represent interactive monitoring, real customer data, real Razorpay metrics, or production model degradation.")

            try:
                from src.services.monitoring.monitoring import MonitoringEngine
                monitor = MonitoringEngine()

                df_raw = pd.read_csv("data/failed_payments.csv")
                df_scored = pd.read_csv("data/failed_payments_scored.csv")

                st.subheader("1. Data Quality Summary")
                dq = monitor.check_data_quality(df_raw)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Records", dq["total_records"])
                col2.metric("Missing Values", dq["missing_value_count"])
                col3.metric("Validation Failures", dq["validation_failure_count"])
                col4.metric("Failure Rate", f"{dq['validation_failure_rate']*100:.1f}%")

                st.subheader("2. Prediction Summary (Reference)")
                pred_summary = monitor.summarize_predictions(df_scored)

                if "recovery_probability" in pred_summary:
                    rp = pred_summary["recovery_probability"]
                    col_rp1, col_rp2, col_rp3 = st.columns(3)
                    col_rp1.metric("Mean Recovery Prob", f"{rp['mean']:.3f}")
                    col_rp2.metric("Median Recovery Prob", f"{rp['median']:.3f}")
                    col_rp3.metric("Max Recovery Prob", f"{rp['max']:.3f}")

                st.subheader("3. Controlled Drift Simulation")
                st.write("Click below to apply a controlled synthetic shift to the raw input dataset (e.g., inflating payment amounts, degrading success rates) and compute PSI metrics.")

                if st.button("Run Drift Simulation"):
                    with st.spinner("Simulating data drift..."):
                        df_drift = monitor.simulate_drift(df_raw)
                        drift_metrics = monitor.calculate_drift_metrics(df_raw, df_drift)

                        st.write("**Drift Metrics (PSI)**")

                        # Apply color formatting
                        def color_status(val):
                            color = 'green' if val == 'NORMAL' else 'orange' if val == 'WARNING' else 'red'
                            return f'color: {color}'

                        st.dataframe(drift_metrics.style.map(color_status, subset=['status']), use_container_width=True)

            except Exception as e:
                st.error(f"Failed to load monitoring module: {str(e)}")
    with tab8:
            st.header("Recovery Decision Audit Trail")
            st.info("Disclaimer: Audit timestamps represent audit record generation time, not payment execution time. This is an AUDITABILITY layer, not real payment execution. The displayed audit history is loaded from the locally generated audit artifact. All model predictions and simulated outcomes are based on synthetic data and do not represent real Razorpay transactions.")

            try:
                from src.services.audit.audit_trail import AuditTrail
                import os

                audit_file = 'reports/recovery_audit_trail.csv'

                if os.path.exists(audit_file):
                    df_audit = pd.read_csv(audit_file)
                    st.success(f"Loaded {len(df_audit)} audit records.")

                    # Filters
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        actions = ["All"] + list(df_audit['recommended_action'].unique())
                        filter_action = st.selectbox("Filter by Recommended Action", actions)
                    with col2:
                        priorities = ["All"] + list(df_audit['recovery_priority'].unique())
                        filter_priority = st.selectbox("Filter by Priority", priorities)
                    with col3:
                        strategies = ["All"] + list(df_audit['strategy_name'].unique())
                        filter_strategy = st.selectbox("Filter by Strategy", strategies)

                    df_filtered = df_audit.copy()
                    if filter_action != "All":
                        df_filtered = df_filtered[df_filtered['recommended_action'] == filter_action]
                    if filter_priority != "All":
                        df_filtered = df_filtered[df_filtered['recovery_priority'] == filter_priority]
                    if filter_strategy != "All":
                        df_filtered = df_filtered[df_filtered['strategy_name'] == filter_strategy]

                    st.write(f"Showing {len(df_filtered)} records.")
                    st.dataframe(df_filtered)

                    # Summary metrics
                    auditor = AuditTrail()
                    summary = auditor.summarize_audit_history(df_filtered)

                    st.subheader("Audit Summary")
                    met1, met2, met3 = st.columns(3)
                    met1.metric("Total Decisions", summary['total_audit_records'])
                    met2.metric("Average Model Probability", f"{summary['average_model_probability']:.1%}")
                    met3.metric("Total Expected Recovery", f"₹{summary['total_expected_recovery']:,.2f}")

                    if summary.get('simulated_recovery_count') is not None:
                        st.subheader("Simulated Outcome Summary")
                        met4, met5, met6 = st.columns(3)
                        met4.metric("Simulated Recoveries", summary['simulated_recovery_count'])
                        met5.metric("Simulated Recovered Revenue", f"₹{summary['simulated_recovered_revenue']:,.2f}")
                        met6.metric("Simulated Net Recovered Revenue", f"₹{summary['simulated_net_recovered_revenue']:,.2f}")

                else:
                    st.warning("No audit trail found. The displayed audit history is loaded from the locally generated audit artifact. Please run the Outcome Simulation to generate audit records.")
                    st.info("Note: Generating a default audit trail creates a new local synthetic audit artifact from the deterministic outcome simulation.")
                    if st.button("Generate Default Audit Trail"):
                        with st.spinner("Generating..."):
                            from src.services.outcome.outcome_simulator import OutcomeSimulator
                            sim = OutcomeSimulator(seed=42)
                            df_sim = sim.simulate_strategy("Optimized Selective Strategy")
                            auditor = AuditTrail()
                            df_audit = auditor.create_from_dataframe(df_sim, strategy_name="Optimized Selective Strategy")
                            auditor.export_audit_records(df_audit, audit_file)
                            st.success("Generated default audit trail. Refreshing...")
                            st.rerun()
            except Exception as e:
                st.error(f"Failed to load Audit Trail: {str(e)}")
    with tab9:
            st.header("PostgreSQL Persistence Status")
            st.caption(
                "Local demo infrastructure only. This database stores synthetic inference records. "
                "It does NOT represent real Razorpay payment execution or real customer data."
            )

            try:
                from src.infrastructure.database.database import is_database_available, get_table_counts, get_database_url

                db_url = get_database_url()
                if not db_url:
                    st.warning("DATABASE_URL is not configured. Copy `.env.example` to `.env` and set your credentials.")
                else:
                    if is_database_available():
                        st.success("PostgreSQL: Connected")
                        counts = get_table_counts()
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Payment Events", counts.get('payment_events', 0))
                        c2.metric("Recovery Decisions", counts.get('recovery_decisions', 0))
                        c3.metric("Audit Records", counts.get('audit_records', 0))
                        c4.metric("Recovery Outcomes", counts.get('recovery_outcomes', 0))

                        with st.expander("Model Versions Registered"):
                            st.write(f"Model Versions: **{counts.get('model_versions', 0)}**")

                        st.info(
                            "Persistence is active. Recovery decisions from the Real-Time Inference Demo "
                            "tab are stored in PostgreSQL when DATABASE_URL is configured."
                        )
                    else:
                        st.error("PostgreSQL: Unavailable")
                        st.info(
                            "The inference pipeline continues to function in stateless mode. "
                            "Start PostgreSQL with Docker to enable persistence:\n\n"
                            "```bash\ndocker compose up -d\n```\n\n"
                            "Then ensure DATABASE_URL is set in your `.env` file."
                        )
            except Exception as e:
                st.error(f"PostgreSQL status check failed: {str(e)}")

if __name__ == "__main__":
    main()
