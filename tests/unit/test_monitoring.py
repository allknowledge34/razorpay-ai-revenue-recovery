import pytest
import pandas as pd
import numpy as np
from src.services.monitoring.monitoring import MonitoringEngine

def get_valid_record():
    return {
        'payment_amount': 5000.0,
        'failure_reason': 'technical_error',
        'payment_method': 'upi',
        'is_subscription': 0,
        'customer_tenure_months': 12,
        'past_successful_payments': 5,
        'past_failed_payments': 1,
        'historical_success_rate': 0.8,
        'time_since_last_success_days': 15,
        'days_overdue': 2,
        'recovery_attempts_so_far': 0
    }

@pytest.fixture
def sample_df():
    data = []
    for i in range(100):
        rec = get_valid_record()
        # vary some things
        rec['payment_amount'] = 100.0 * (i % 5 + 1)
        rec['historical_success_rate'] = [0.1, 0.5, 0.9, 1.0, 0.0][i % 5]
        rec['failure_reason'] = ['technical_error', 'insufficient_funds', 'invalid_card', 'limit_exceeded', 'technical_error'][i % 5]
        rec['payment_method'] = ['upi', 'credit_card', 'debit_card', 'bank_transfer', 'upi'][i % 5]
        rec['is_subscription'] = [0, 1, 0, 1, 0][i % 5]
        data.append(rec)
    return pd.DataFrame(data)

def test_data_quality(sample_df):
    engine = MonitoringEngine()
    dq = engine.check_data_quality(sample_df)
    assert dq['total_records'] == 100
    assert dq['missing_value_count'] == 0
    assert dq['validation_failure_count'] == 0

def test_missing_values(sample_df):
    engine = MonitoringEngine()
    df_missing = sample_df.copy()
    df_missing.loc[0, 'payment_amount'] = np.nan
    dq = engine.check_data_quality(df_missing)
    assert dq['missing_value_count'] == 1
    assert dq['missing_by_column']['payment_amount'] == 1

def test_categorical_distribution(sample_df):
    engine = MonitoringEngine()
    psi = engine.calculate_categorical_psi(sample_df['failure_reason'], sample_df['failure_reason'])
    assert psi == 0.0

def test_numerical_drift(sample_df):
    engine = MonitoringEngine()
    # Identical dataset should have 0 PSI
    psi = engine.calculate_psi(sample_df['payment_amount'].values, sample_df['payment_amount'].values)
    assert psi == 0.0
    
    # Strongly shifted dataset should have non-zero PSI
    shifted = sample_df['payment_amount'].values * 10
    psi_shifted = engine.calculate_psi(sample_df['payment_amount'].values, shifted)
    assert psi_shifted > 0.1

def test_categorical_drift(sample_df):
    engine = MonitoringEngine()
    shifted_reason = sample_df['failure_reason'].replace('technical_error', 'insufficient_funds')
    psi = engine.calculate_categorical_psi(sample_df['failure_reason'], shifted_reason)
    assert psi > 0.1

def test_deterministic_drift_simulation(sample_df):
    engine = MonitoringEngine()
    drift1 = engine.simulate_drift(sample_df, seed=42)
    drift2 = engine.simulate_drift(sample_df, seed=42)
    
    pd.testing.assert_frame_equal(drift1, drift2)

def test_drift_simulation_does_not_mutate(sample_df):
    engine = MonitoringEngine()
    original_sum = sample_df['payment_amount'].sum()
    _ = engine.simulate_drift(sample_df, seed=42)
    assert sample_df['payment_amount'].sum() == original_sum

def test_no_nan_inf_metric(sample_df):
    engine = MonitoringEngine()
    drift_df = engine.simulate_drift(sample_df, seed=42)
    metrics = engine.calculate_drift_metrics(sample_df, drift_df)
    
    assert not metrics['drift_metric_psi'].isna().any()
    assert not np.isinf(metrics['drift_metric_psi']).any()

def test_monitoring_status(sample_df):
    engine = MonitoringEngine()
    drift_df = engine.simulate_drift(sample_df, seed=42)
    metrics = engine.calculate_drift_metrics(sample_df, drift_df)
    
    statuses = metrics['status'].unique()
    for status in statuses:
        assert status in ['NORMAL', 'WARNING', 'DRIFT']


def test_validation_counted_by_record(sample_df):
    engine = MonitoringEngine()
    df_invalid = sample_df.copy()
    # Create 3 errors in the first record
    df_invalid.loc[0, 'payment_amount'] = -100
    df_invalid.loc[0, 'historical_success_rate'] = 1.5
    df_invalid.loc[0, 'is_subscription'] = 2
    
    dq = engine.check_data_quality(df_invalid)
    # Total failure count should be 1, because it's 1 invalid record, even with 3 errors
    assert dq['validation_failure_count'] == 1

def test_monitoring_handles_missing_columns_safely(sample_df):
    engine = MonitoringEngine()
    df_missing = sample_df.drop(columns=['payment_amount'])
    dq = engine.check_data_quality(df_missing)
    assert dq['validation_failure_count'] == 100

def test_simulate_drift_does_not_mutate_global_rng(sample_df):
    engine = MonitoringEngine()
    np.random.seed(999)
    val1 = np.random.rand()
    
    np.random.seed(999)
    engine.simulate_drift(sample_df, seed=42)
    val2 = np.random.rand()
    
    # If simulate_drift called np.random.seed(42), val2 would differ from val1.
    assert val1 == val2

def test_invalid_is_subscription_detected(sample_df):
    engine = MonitoringEngine()
    df_invalid = sample_df.copy()
    df_invalid.loc[0, 'is_subscription'] = 2
    dq = engine.check_data_quality(df_invalid)
    assert dq['validation_failure_count'] == 1

def test_fractional_count_detected(sample_df):
    engine = MonitoringEngine()
    df_invalid = sample_df.copy()
    df_invalid['past_successful_payments'] = df_invalid['past_successful_payments'].astype(float)
    df_invalid.loc[0, 'past_successful_payments'] = 2.5
    dq = engine.check_data_quality(df_invalid)
    assert dq['validation_failure_count'] == 1

def test_nan_inf_input_detected(sample_df):
    engine = MonitoringEngine()
    df_invalid = sample_df.copy()
    df_invalid['payment_amount'] = df_invalid['payment_amount'].astype(float)
    df_invalid.loc[0, 'payment_amount'] = float('nan')
    df_invalid.loc[1, 'payment_amount'] = float('inf')
    dq = engine.check_data_quality(df_invalid)
    assert dq['validation_failure_count'] == 2
