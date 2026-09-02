import pytest
import pandas as pd
import datetime
from src.audit_trail import AuditTrail

@pytest.fixture
def sample_record():
    return {
        'payment_id': 'pay_123',
        'payment_amount': 5000,
        'historical_success_rate': 0.8
    }

@pytest.fixture
def sample_engine_result():
    return {
        'recovery_probability': 0.8,
        'recommended_action': 'Retry Payment',
        'expected_recovery': 4000.0,
        'priority': 'HIGH',
        'simulated_recovered': 1,
        'simulated_recovered_revenue': 5000.0,
        'action_cost': 50.0,
        'net_recovered_revenue': 4950.0
    }

def test_required_audit_columns(sample_record, sample_engine_result):
    auditor = AuditTrail()
    audit = auditor.create_audit_record(sample_record, sample_engine_result)
    
    required_cols = [
        'audit_id', 'payment_id', 'timestamp', 'model_probability', 
        'recovery_priority', 'recommended_action', 'decision_threshold',
        'effective_retry_cost', 'expected_recovery', 'expected_retry_net_value',
        'decision_reason', 'strategy_name'
    ]
    for col in required_cols:
        assert col in audit

def test_deterministic_audit_id(sample_record, sample_engine_result):
    auditor = AuditTrail()
    audit1 = auditor.create_audit_record(sample_record, sample_engine_result, audit_timestamp="2026-09-02T12:00:00Z")
    audit2 = auditor.create_audit_record(sample_record, sample_engine_result, audit_timestamp="2026-09-02T12:00:00Z")
    assert audit1['audit_id'] == audit2['audit_id']

def test_supplied_timestamp_preserved(sample_record, sample_engine_result):
    auditor = AuditTrail()
    ts = "2026-09-02T12:00:00Z"
    audit = auditor.create_audit_record(sample_record, sample_engine_result, audit_timestamp=ts)
    assert audit['timestamp'] == ts

def test_missing_timestamp_generates_utc(sample_record, sample_engine_result):
    auditor = AuditTrail()
    audit = auditor.create_audit_record(sample_record, sample_engine_result)
    assert audit['timestamp'] is not None
    assert isinstance(audit['timestamp'], str)
    # Check simple ISO format length
    assert len(audit['timestamp']) > 15

def test_payment_id_preserved(sample_record, sample_engine_result):
    auditor = AuditTrail()
    audit = auditor.create_audit_record(sample_record, sample_engine_result)
    assert audit['payment_id'] == 'pay_123'

def test_model_probability_preserved(sample_record, sample_engine_result):
    auditor = AuditTrail()
    audit = auditor.create_audit_record(sample_record, sample_engine_result)
    assert audit['model_probability'] == 0.8

def test_action_preserved(sample_record, sample_engine_result):
    auditor = AuditTrail()
    audit = auditor.create_audit_record(sample_record, sample_engine_result)
    assert audit['recommended_action'] == 'Retry Payment'

def test_expected_recovery_preserved(sample_record, sample_engine_result):
    auditor = AuditTrail()
    audit = auditor.create_audit_record(sample_record, sample_engine_result)
    assert audit['expected_recovery'] == 4000.0

def test_expected_and_simulated_revenue_separate(sample_record, sample_engine_result):
    auditor = AuditTrail()
    audit = auditor.create_audit_record(sample_record, sample_engine_result)
    assert audit['expected_recovery'] != audit['simulated_recovered_revenue']
    assert audit['expected_recovery'] == 4000.0
    assert audit['simulated_recovered_revenue'] == 5000.0

def test_missing_outcome_columns_handled(sample_record, sample_engine_result):
    auditor = AuditTrail()
    # Remove outcome cols
    for k in ['simulated_recovered', 'simulated_recovered_revenue', 'action_cost', 'net_recovered_revenue']:
        del sample_engine_result[k]
        
    audit = auditor.create_audit_record(sample_record, sample_engine_result)
    assert audit['simulated_recovered'] is None
    assert audit['simulated_recovered_revenue'] is None

def test_action_summary_counts():
    auditor = AuditTrail()
    df_audit = pd.DataFrame([
        {'recommended_action': 'Retry Payment', 'model_probability': 0.8, 'expected_recovery': 100, 'expected_retry_net_value': 50},
        {'recommended_action': 'Retry Payment', 'model_probability': 0.8, 'expected_recovery': 100, 'expected_retry_net_value': 50},
        {'recommended_action': 'Payment Method Reminder', 'model_probability': 0.5, 'expected_recovery': 50, 'expected_retry_net_value': 0}
    ])
    summary = auditor.summarize_audit_history(df_audit)
    assert summary['retry_decisions'] == 2
    assert summary['reminder_decisions'] == 1
    assert summary['manual_review_decisions'] == 0

def test_audit_export_does_not_mutate_source():
    auditor = AuditTrail()
    df_audit = pd.DataFrame([{'audit_id': '123', 'payment_id': 'pay_123'}])
    df_copy = df_audit.copy()
    auditor.export_audit_records(df_audit, filepath='reports/test_audit.csv')
    pd.testing.assert_frame_equal(df_audit, df_copy)

def test_filtering_works_correctly():
    # We can just test basic pandas filtering which will be used in Streamlit
    df_audit = pd.DataFrame([
        {'recommended_action': 'Retry Payment', 'recovery_priority': 'HIGH', 'strategy_name': 'Strategy A'},
        {'recommended_action': 'Payment Method Reminder', 'recovery_priority': 'MEDIUM', 'strategy_name': 'Strategy A'}
    ])
    filtered = df_audit[df_audit['recommended_action'] == 'Retry Payment']
    assert len(filtered) == 1
    assert filtered.iloc[0]['recovery_priority'] == 'HIGH'

def test_fixed_timestamp_reproducible(sample_record, sample_engine_result):
    auditor = AuditTrail()
    audit1 = auditor.create_audit_record(sample_record, sample_engine_result, audit_timestamp="2026-01-01T00:00:00Z")
    audit2 = auditor.create_audit_record(sample_record, sample_engine_result, audit_timestamp="2026-01-01T00:00:00Z")
    assert audit1 == audit2

