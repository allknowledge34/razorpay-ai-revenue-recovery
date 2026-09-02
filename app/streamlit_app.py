import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add root directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.recovery_engine import RecoveryEngine

st.set_page_config(page_title="AI Revenue Recovery Engine", layout="wide")

@st.cache_resource
def load_engine():
    return RecoveryEngine()

def main():
    st.title("AI Revenue Recovery Engine")
    st.markdown("### Predict failed-payment recovery and prioritize the next best recovery action.")
    
    try:
        engine = load_engine()
    except Exception as e:
        st.error(f"Failed to load the Recovery Engine. Make sure the model exists. Error: {e}")
        return

    # Create tabs for Single Payment Simulation and Batch Analysis
    tab1, tab2 = st.tabs(["Single Payment Simulation", "Batch Recovery Analysis"])

    with tab1:
        st.header("Payment Simulation")
        
        with st.form("payment_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                payment_amount = st.number_input("Payment Amount (₹)", min_value=1.0, value=5000.0, step=100.0)
                failure_reason = st.selectbox("Failure Reason", ["technical_error", "insufficient_funds", "invalid_card", "other"])
                payment_method = st.selectbox("Payment Method", ["credit_card", "debit_card", "upi", "net_banking"])
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
                
                col_exp, col_biz = st.columns(2)
                
                with col_exp:
                    st.subheader("Why this recommendation?")
                    st.markdown(f"- **Historical Success:** A rate of {hist_success_rate*100:.0f}% heavily influences the recovery likelihood.")
                    st.markdown(f"- **Failure Reason:** `{failure_reason}` failures typically exhibit specific operational recovery patterns.")
                    st.markdown(f"- **Age:** The payment is {days_overdue} days overdue, and {recovery_attempts} attempts have been made, which strongly dictates the urgency.")
                    st.markdown("- **Formula:** `Expected Recovery = Payment Amount × Recovery Probability`")

                with col_biz:
                    st.subheader("Business Impact")
                    unrecovered = payment_amount - res['expected_recovery']
                    b1, b2, b3 = st.columns(3)
                    b1.metric("Payment At Risk", f"₹{payment_amount:,.2f}")
                    b2.metric("Expected Recoverable Revenue", f"₹{res['expected_recovery']:,.2f}")
                    b3.metric("Potentially Unrecovered Amount", f"₹{unrecovered:,.2f}")
                    
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

if __name__ == "__main__":
    main()
