import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Add root directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.recovery_engine import RecoveryEngine
from src.recovery_strategy import RecoverySimulator
from src.decision_trace import DecisionTracer

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

def main():
    st.title("AI Revenue Recovery Engine")
    st.markdown("### Predict failed-payment recovery and prioritize the next best recovery action.")
    
    try:
        engine = load_engine()
        simulator = load_simulator()
        tracer = load_tracer()
    except Exception as e:
        st.error(f"Failed to load the Recovery Engine or Simulator. Error: {e}")
        return

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Single Payment Simulation", "Batch Recovery Analysis", "Strategy Simulator", "Monitoring", "Outcome Simulation", "Audit Trail"])

    with tab1:
        st.header("Payment Simulation")
        
        with st.form("payment_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                payment_amount = st.number_input("Payment Amount (₹)", min_value=1.0, value=5000.0, step=100.0)
                failure_reason = st.selectbox("Failure Reason", ["technical_error", "insufficient_funds", "invalid_card", "limit_exceeded"])
                payment_method = st.selectbox("Payment Method", ["credit_card", "debit_card", "upi", "bank_transfer"])
                is_subscription = st.checkbox("Subscription Payment?")
            with col2:
                customer_tenure = st.number_input("Customer Tenure (months)", min_value=0, value=12)
                past_success = st.number_input("Past Successful Payments", min_value=0, value=5)
                past_failed = st.number_input("Past Failed Payments", min_value=0, value=1)
                hist_success_rate = st.slider("Historical Success Rate", 0.0, 1.0, value=0.8, step=0.01)
            with col3:
                days_since_success = st.number_input("Days Since Last Success", min_value=0, value=15)
                days_overdue = st.number_input("Days Overdue", min_value=0, value=2)
                recovery_attempts = st.number_input("Recovery Attempts So Far", min_value=0, value=0)
            
            submit = st.form_submit_button("Analyze Recovery")

        if submit:
            record = {
                "payment_amount": payment_amount,
                "failure_reason": failure_reason,
                "payment_method": payment_method,
                "is_subscription": int(is_subscription),
                "customer_tenure_months": customer_tenure,
                "past_successful_payments": past_success,
                "past_failed_payments": past_failed,
                "historical_success_rate": hist_success_rate,
                "time_since_last_success_days": days_since_success,
                "days_overdue": days_overdue,
                "recovery_attempts_so_far": recovery_attempts
            }

            try:
                # Validate the record explicitly so we can show user-friendly errors
                from src.data_validator import DataValidator
                val_result = DataValidator.validate_record(record)
                if not val_result.is_valid:
                    st.error("Validation Error:")
                    for err in val_result.errors:
                        st.warning(f"- {err}")
                    return

                res = engine.predict_recovery(record)
                
                st.divider()
                st.subheader("Prediction Results")
                
                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Recovery Probability", f"{res['recovery_probability']*100:.1f}%")
                m2.metric("Expected Recovery", f"₹{res['expected_recovery']:,.2f}")
                m3.metric("Priority", res['priority'])
                m4.metric("Recommended Action", res['recommended_action'])
                st.divider()
                st.subheader("Decision Trace")
                trace = tracer.generate_trace(record, res)
                
                dt1, dt2, dt3, dt4 = st.columns(4)
                dt1.markdown("**Model Estimate**")
                dt1.write(f"Recovery Prob: {trace['recovery_probability']*100:.1f}%")
                dt1.write(f"Expected: ₹{trace['expected_recovery']:,.2f}")
                
                dt2.markdown("**Strategy Simulator Boundary**")
                dt2.write(f"Hypothetical Threshold: {trace['selected_threshold']*100:.1f}%")
                exceeds = "YES" if trace['recovery_probability'] >= trace['selected_threshold'] else "NO"
                dt2.write(f"Exceeds Threshold: {exceeds}")
                
                dt3.markdown("**Hypothetical Retry Economics**")
                dt3.write(f"Amount: ₹{trace['payment_amount']:,.2f}")
                dt3.write(f"Retry Cost: ₹{trace['effective_retry_cost']:,.2f}")
                dt3.write(f"Net Retry Value: ₹{trace['expected_retry_net_value']:,.2f}")
                
                dt4.markdown("**Stage 5 Decision**")
                dt4.write(f"**{trace['recommended_action']}**")
                
                st.markdown("**Why?**")
                st.info(trace['decision_reason'])
                
                with st.expander("Key Input Factors Considered"):
                    for factor in trace['key_input_factors']:
                        st.write(f"- {factor}")
                        
                st.caption("Decision explanations describe model estimates and configured business rules. They are not causal explanations and do not guarantee successful recovery. Cost and action-effectiveness values are simulation assumptions, not actual Razorpay production economics.")



                    
            except Exception as e:
                st.error(f"Error making prediction: {e}")

    with tab2:
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

    with tab3:
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


    with tab4:
        st.header("Synthetic / Offline Monitoring")
        st.info("Disclaimer: This is a synthetic/offline monitoring framework demonstrated using synthetic data. It does not represent interactive monitoring, real customer data, real Razorpay metrics, or production model degradation.")
        
        try:
            from src.monitoring import MonitoringEngine
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



    with tab5:
        st.header("Synthetic Outcome Simulation — Not Real Payment Execution")
        st.info("Disclaimer: This is a synthetic simulation of recovery outcomes based on assumed action effectiveness and costs. It does not represent actual Razorpay recovery execution, real customer behavior, or real payment processing.")
        
        try:
            from src.outcome_simulator import OutcomeSimulator
            
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


    with tab6:
        st.header("Recovery Decision Audit Trail")
        st.info("Disclaimer: Audit timestamps represent audit record generation time, not payment execution time. This is an AUDITABILITY layer, not real payment execution. The displayed audit history is loaded from the locally generated audit artifact. All model predictions and simulated outcomes are based on synthetic data and do not represent real Razorpay transactions.")
        
        try:
            from src.audit_trail import AuditTrail
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
                        from src.outcome_simulator import OutcomeSimulator
                        sim = OutcomeSimulator(seed=42)
                        df_sim = sim.simulate_strategy("Optimized Selective Strategy")
                        auditor = AuditTrail()
                        df_audit = auditor.create_from_dataframe(df_sim, strategy_name="Optimized Selective Strategy")
                        auditor.export_audit_records(df_audit, audit_file)
                        st.success("Generated default audit trail. Refreshing...")
                        st.rerun()
        except Exception as e:
            st.error(f"Failed to load Audit Trail: {str(e)}")

if __name__ == "__main__":
    main()
