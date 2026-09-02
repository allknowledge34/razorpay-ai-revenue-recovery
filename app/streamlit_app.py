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
    tab1, tab2, tab3 = st.tabs(["Single Payment Simulation", "Batch Recovery Analysis", "Strategy Simulator"])

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

if __name__ == "__main__":
    main()
