import pytest
import pandas as pd
import numpy as np
from src.services.outcome.outcome_simulator import OutcomeSimulator

def test_deterministic_simulation_same_seed():
    sim1 = OutcomeSimulator(seed=42)
    sim2 = OutcomeSimulator(seed=42)
    df_sim1 = sim1.simulate_strategy("Optimized Selective Strategy")
    df_sim2 = sim2.simulate_strategy("Optimized Selective Strategy")
    pd.testing.assert_frame_equal(df_sim1, df_sim2)

def test_different_seed_different_outcomes():
    sim1 = OutcomeSimulator(seed=42)
    sim2 = OutcomeSimulator(seed=999)
    df_sim1 = sim1.simulate_strategy("Optimized Selective Strategy")
    df_sim2 = sim2.simulate_strategy("Optimized Selective Strategy")
    # Due to different random uniforms, simulated_recovered should differ on average
    assert not df_sim1['simulated_recovered'].equals(df_sim2['simulated_recovered'])

def test_effective_probability_bounds():
    sim = OutcomeSimulator()
    df_sim = sim.simulate_strategy("Optimized Selective Strategy")
    assert df_sim['effective_recovery_probability'].min() >= 0.0
    assert df_sim['effective_recovery_probability'].max() <= 1.0

def test_simulated_recovered_binary():
    sim = OutcomeSimulator()
    df_sim = sim.simulate_strategy("Blind Retry")
    unique_vals = df_sim['simulated_recovered'].unique()
    for val in unique_vals:
        assert val in [0, 1]

def test_recovered_revenue_logic():
    sim = OutcomeSimulator()
    df_sim = sim.simulate_strategy("Current Rule-Based Strategy")
    
    # If simulated_recovered == 1, revenue == payment_amount
    mask1 = df_sim['simulated_recovered'] == 1
    assert (df_sim.loc[mask1, 'simulated_recovered_revenue'] == df_sim.loc[mask1, 'payment_amount']).all()
    
    # If simulated_recovered == 0, revenue == 0
    mask0 = df_sim['simulated_recovered'] == 0
    assert (df_sim.loc[mask0, 'simulated_recovered_revenue'] == 0.0).all()

def test_net_revenue_formula():
    sim = OutcomeSimulator()
    df_sim = sim.simulate_strategy("Blind Retry")
    expected_net = df_sim['simulated_recovered_revenue'] - df_sim['action_cost']
    assert np.allclose(df_sim['net_recovered_revenue'], expected_net)

def test_zero_cost_roi_handling():
    sim = OutcomeSimulator()
    df_sim = sim.simulate_strategy("Optimized Selective Strategy")
    # Artificially set total action cost to 0
    df_sim['action_cost'] = 0.0
    metrics = sim.calculate_metrics(df_sim, "Zero Cost Test")
    assert metrics['roi'] == 0.0

def test_zero_payment_value_edge_case():
    sim = OutcomeSimulator()
    df_sim = sim.simulate_strategy("Optimized Selective Strategy")
    # Artificially set all payment amounts to 0
    df_sim['payment_amount'] = 0.0
    df_sim['simulated_recovered_revenue'] = 0.0 # because amount is 0
    metrics = sim.calculate_metrics(df_sim, "Zero Amount Test")
    assert metrics['revenue_recovery_rate'] == 0.0
    assert metrics['gross_simulated_recovered_revenue'] == 0.0

def test_strategy_comparison_same_population():
    sim = OutcomeSimulator()
    df_a = sim.simulate_strategy("Blind Retry")
    df_b = sim.simulate_strategy("Current Rule-Based Strategy")
    df_c = sim.simulate_strategy("Optimized Selective Strategy")
    assert len(df_a) == len(df_b) == len(df_c) == len(sim.df)
    assert (df_a['payment_id'] == df_b['payment_id']).all()
    assert (df_b['payment_id'] == df_c['payment_id']).all()

def test_required_output_columns_exist():
    sim = OutcomeSimulator()
    df_sim = sim.simulate_strategy("Blind Retry")
    required_cols = [
        'payment_id', 'payment_amount', 'recovery_probability', 
        'simulated_action', 'action_effectiveness', 'effective_recovery_probability',
        'simulated_recovered', 'simulated_recovered_revenue', 'action_cost', 'net_recovered_revenue'
    ]
    for col in required_cols:
        assert col in df_sim.columns

def test_original_dataset_unchanged():
    df_original = pd.read_csv("data/failed_payments_scored.csv")
    sim = OutcomeSimulator()
    df_sim = sim.simulate_strategy("Optimized Selective Strategy")
    pd.testing.assert_frame_equal(sim.df, df_original)

