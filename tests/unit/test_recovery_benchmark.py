import pytest
import pandas as pd
import numpy as np
from src.ml.evaluation.recovery_benchmark import RecoveryBenchmark, calculate_deltas, resolve_seed

@pytest.fixture
def sample_data(tmp_path):
    df = pd.DataFrame({
        'payment_id': ['p1', 'p2', 'p3'],
        'payment_amount': [1000.0, 5000.0, 30000.0],
        'recovery_probability': [0.9, 0.4, 0.01],
        'recovery_attempts_so_far': [0, 1, 3],
        'recommended_action': ['Retry Payment', 'Payment Method Reminder', 'Stop Automatic Recovery']
    })
    path = tmp_path / "mock_scored.csv"
    df.to_csv(path, index=False)
    
    return str(path)

def test_resolve_seed():
    seed1 = resolve_seed("pay_123")
    seed2 = resolve_seed("pay_123")
    seed3 = resolve_seed("pay_456")
    assert seed1 == seed2
    assert seed1 != seed3
    assert isinstance(seed1, int)

def test_benchmark_metrics(sample_data, monkeypatch):
    from src.domain.recovery_strategy import RecoverySimulator
    def mock_sweep(self, action_cost):
        return pd.DataFrame({'threshold': [0.5], 'expected_net_recovery': [5000]}), pd.Series({'threshold': 0.5})
    monkeypatch.setattr(RecoverySimulator, "threshold_sweep", mock_sweep)
    
    bench = RecoveryBenchmark(data_path=sample_data)
    df = bench.evaluate_all()
    
    assert len(df) == 4
    strats = df['Strategy'].tolist()
    assert 'A. Blind Retry' in strats
    assert 'B. Current Rule-Based' in strats
    assert 'C. Optimized Selective' in strats
    assert 'D. Bounded Recovery Orchestrator' in strats
    
    for _, r in df.iterrows():
        assert r['Total Payments'] == 3
        assert r['Revenue at Risk'] == 36000.0
        assert 0.0 <= r['Recovery Rate'] <= 1.0
        assert r['Action Cost'] >= 0.0
        assert r['Simulated Recovered Revenue'] >= 0.0
        
def test_calculate_deltas():
    df = pd.DataFrame({
        'Strategy': ['A. Blind Retry', 'C. Optimized Selective'],
        'Net Recovered Revenue': [1000.0, 1200.0],
        'Recovery Rate': [0.5, 0.4],
        'Action Cost': [100.0, 50.0],
        'Retry Count': [10, 5],
        'ROI': [10.0, 24.0]
    })
    deltas = calculate_deltas(df)
    assert len(deltas) == 1
    
    row = deltas.iloc[0]
    assert row['Comparison'] == 'C. Optimized Selective vs A. Blind Retry'
    assert row['Net Recovered Revenue Delta'] == 200.0
    assert row['Action Cost Delta'] == -50.0

def test_reproducibility(sample_data, monkeypatch):
    from src.domain.recovery_strategy import RecoverySimulator
    def mock_sweep(self, action_cost):
        return pd.DataFrame({'threshold': [0.5], 'expected_net_recovery': [5000]}), pd.Series({'threshold': 0.5})
    monkeypatch.setattr(RecoverySimulator, "threshold_sweep", mock_sweep)
    
    bench1 = RecoveryBenchmark(data_path=sample_data)
    df1 = bench1.evaluate_all()
    
    bench2 = RecoveryBenchmark(data_path=sample_data)
    df2 = bench2.evaluate_all()
    
    pd.testing.assert_frame_equal(df1, df2)

def test_strategy_d_guardrails_and_recommendations(sample_data, monkeypatch):
    from src.domain.recovery_strategy import RecoverySimulator
    def mock_sweep(self, action_cost):
        return pd.DataFrame({'threshold': [0.5], 'expected_net_recovery': [5000]}), pd.Series({'threshold': 0.5})
    monkeypatch.setattr(RecoverySimulator, "threshold_sweep", mock_sweep)
    
    bench = RecoveryBenchmark(data_path=sample_data)
    df = bench.evaluate_all()
    strat_d = df[df['Strategy'] == 'D. Bounded Recovery Orchestrator'].iloc[0]
    
    assert strat_d['Maximum Existing Attempts Observed'] == 3
    assert strat_d['Configured Maximum Automatic Attempts'] == 2
    assert 'Maximum Automatic Attempt Number' not in strat_d.index
