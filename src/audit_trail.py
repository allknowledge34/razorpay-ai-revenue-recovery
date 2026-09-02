import pandas as pd
import numpy as np
import datetime
import hashlib
from typing import Dict, Any, List, Optional
import os

from src.decision_trace import DecisionTracer
from src.recovery_strategy import RecoverySimulator

class AuditTrail:
    def __init__(self, simulator_cost=50.0, simulator_threshold=0.05):
        self.tracer = DecisionTracer(simulator_cost=simulator_cost, simulator_threshold=simulator_threshold)
        self.simulator_cost = simulator_cost
        
    def _generate_audit_id(self, payment_id: str, strategy_name: str, timestamp_str: str) -> str:
        """
        Generate a deterministic, reproducible Audit ID.
        Uses payment_id, strategy_name, and timestamp to ensure reproducible hashes for tests.
        """
        seed_string = f"{payment_id}_{strategy_name}_{timestamp_str}"
        return hashlib.sha256(seed_string.encode('utf-8')).hexdigest()[:16]
        
    def create_audit_record(self, record: dict, engine_result: dict, strategy_name: str = "Rule-Based", audit_timestamp: str = None) -> dict:
        """
        Create a single audit record.
        Combines model prediction, decision trace, and simulated outcomes (if available).
        """
        timestamp = audit_timestamp if audit_timestamp else datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # Generate Trace
        trace = self.tracer.generate_trace(record, engine_result)
        
        # Determine priority safely (it might be in engine_result)
        priority = engine_result.get('priority', 'UNKNOWN')
        if priority == 'UNKNOWN':
            # Approximate priority band if not present
            prob = engine_result.get('recovery_probability', 0)
            if prob >= 0.65:
                priority = 'HIGH'
            elif prob >= 0.35:
                priority = 'MEDIUM'
            else:
                priority = 'LOW'
                
        payment_id = str(record.get('payment_id', 'UNKNOWN'))
        audit_id = self._generate_audit_id(payment_id, strategy_name, timestamp)
        
        audit_record = {
            'audit_id': audit_id,
            'payment_id': payment_id,
            'timestamp': timestamp,
            'model_probability': engine_result.get('recovery_probability', 0.0),
            'recovery_priority': priority,
            'recommended_action': engine_result.get('recommended_action', trace.get('recommended_action', 'Unknown')),
            'decision_threshold': trace.get('selected_threshold', self.tracer.simulator_threshold),
            'effective_retry_cost': trace.get('effective_retry_cost', self.simulator_cost),
            'expected_recovery': engine_result.get('expected_recovery', trace.get('expected_recovery', 0.0)),
            'expected_retry_net_value': trace.get('expected_retry_net_value', 0.0),
            'decision_reason': trace.get('decision_reason', ''),
            'key_input_factors': " | ".join(trace.get('key_input_factors', [])),
            'strategy_name': strategy_name,
            # Outcomes (None if missing)
            'simulated_recovered': engine_result.get('simulated_recovered', None),
            'simulated_recovered_revenue': engine_result.get('simulated_recovered_revenue', None),
            'action_cost': engine_result.get('action_cost', None),
            'net_recovered_revenue': engine_result.get('net_recovered_revenue', None)
        }
        
        return audit_record
        
    def create_from_dataframe(self, df: pd.DataFrame, strategy_name: str, audit_timestamp: str = None) -> pd.DataFrame:
        """
        Generates an audit trail DataFrame from a scoring/simulation DataFrame.
        """
        records = df.to_dict('records')
        audit_records = []
        timestamp = audit_timestamp if audit_timestamp else datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        for row in records:
            engine_result = {
                'recovery_probability': row.get('recovery_probability', 0.0),
                # Map simulated_action back to recommended_action so DecisionTracer can use it
                'recommended_action': row.get('simulated_action', row.get('recommended_action', 'Unknown')),
                'expected_recovery': row.get('payment_amount', 0.0) * row.get('recovery_probability', 0.0),
                'priority': row.get('priority', 'UNKNOWN')
            }
            
            # Outcome injection
            for key in ['simulated_recovered', 'simulated_recovered_revenue', 'action_cost', 'net_recovered_revenue']:
                if key in row:
                    engine_result[key] = row[key]
                    
            audit_record = self.create_audit_record(row, engine_result, strategy_name=strategy_name, audit_timestamp=timestamp)
            audit_records.append(audit_record)
            
        return pd.DataFrame(audit_records)
        
    def summarize_audit_history(self, audit_df: pd.DataFrame) -> dict:
        total = len(audit_df)
        actions = audit_df['recommended_action'].value_counts().to_dict()
        
        summary = {
            'total_audit_records': total,
            'retry_decisions': actions.get('Retry Payment', 0),
            'reminder_decisions': actions.get('Payment Method Reminder', 0),
            'manual_review_decisions': actions.get('Manual Review / Stop Automatic Retry', 0),
            'do_nothing_decisions': actions.get('Do Nothing', 0),
            'average_model_probability': audit_df['model_probability'].mean() if total > 0 else 0.0,
            'total_expected_recovery': audit_df['expected_recovery'].sum() if total > 0 else 0.0,
            'total_expected_retry_net_value': audit_df['expected_retry_net_value'].sum() if total > 0 else 0.0
        }
        
        if 'simulated_recovered' in audit_df.columns and audit_df['simulated_recovered'].notna().any():
            summary['simulated_recovery_count'] = int(audit_df['simulated_recovered'].sum())
            summary['simulated_recovered_revenue'] = audit_df['simulated_recovered_revenue'].sum()
            summary['simulated_net_recovered_revenue'] = audit_df['net_recovered_revenue'].sum()
        else:
            summary['simulated_recovery_count'] = None
            summary['simulated_recovered_revenue'] = None
            summary['simulated_net_recovered_revenue'] = None
            
        return summary
        
    def export_audit_records(self, audit_df: pd.DataFrame, filepath: str = 'reports/recovery_audit_trail.csv'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        audit_df.to_csv(filepath, index=False)
        return filepath

if __name__ == "__main__":
    from src.outcome_simulator import OutcomeSimulator
    
    sim = OutcomeSimulator(seed=42)
    df_sim = sim.simulate_strategy("Optimized Selective Strategy")
    
    auditor = AuditTrail()
    df_audit = auditor.create_from_dataframe(df_sim, strategy_name="Optimized Selective Strategy", audit_timestamp="2026-09-02T12:00:00Z")
    
    summary = auditor.summarize_audit_history(df_audit)
    
    print("Audit Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
        
    auditor.export_audit_records(df_audit)
    print("\nAudit trail exported to reports/recovery_audit_trail.csv")
