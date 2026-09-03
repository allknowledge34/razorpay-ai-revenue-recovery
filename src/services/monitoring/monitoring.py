import pandas as pd
import numpy as np
import json
import os
from src.validation.data_validator import DataValidator

class MonitoringEngine:
    def __init__(self):
        pass

    def check_data_quality(self, df: pd.DataFrame) -> dict:
        total = len(df)
        missing_counts = df.isnull().sum().to_dict()
        duplicates = int(df.duplicated().sum())

        validation_failure_count = 0
        
        # Determine if required features are present
        missing_features = [f for f in DataValidator.REQUIRED_FEATURES if f not in df.columns]
        
        if missing_features:
            # If a required column is missing, all records are technically invalid
            validation_failure_count = total
        else:
            # Iterate through rows and validate
            # We can use apply or iterate
            for _, row in df.iterrows():
                record = row.to_dict()
                res = DataValidator.validate_record(record)
                if not res.is_valid:
                    validation_failure_count += 1

        return {
            "total_records": total,
            "missing_value_count": sum(missing_counts.values()),
            "missing_by_column": missing_counts,
            "duplicate_count": duplicates,
            "validation_failure_count": validation_failure_count,
            "validation_failure_rate": round(validation_failure_count / total, 4) if total > 0 else 0
        }

    def summarize_predictions(self, df: pd.DataFrame) -> dict:
        summary = {}
        if 'recovery_probability' in df.columns:
            summary['recovery_probability'] = {
                "mean": float(df['recovery_probability'].mean()),
                "median": float(df['recovery_probability'].median()),
                "min": float(df['recovery_probability'].min()),
                "max": float(df['recovery_probability'].max()),
                "std": float(df['recovery_probability'].std())
            }
        
        if 'expected_recovery' in df.columns:
            summary['expected_recovery'] = {
                "mean": float(df['expected_recovery'].mean()),
                "median": float(df['expected_recovery'].median()),
                "min": float(df['expected_recovery'].min()),
                "max": float(df['expected_recovery'].max()),
                "std": float(df['expected_recovery'].std())
            }

        if 'priority' in df.columns:
            counts = df['priority'].value_counts().to_dict()
            summary['priority_distribution'] = {
                "HIGH": counts.get("HIGH", 0),
                "MEDIUM": counts.get("MEDIUM", 0),
                "LOW": counts.get("LOW", 0)
            }
            
        if 'recommended_action' in df.columns:
            summary['action_distribution'] = df['recommended_action'].value_counts().to_dict()

        return summary

    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).
        For continuous data, we use quantiles of expected data to create bins.
        """
        def replace_zero(arr):
            # Replace 0 with a very small number to avoid division by zero or log(0)
            return np.where(arr == 0, 0.0001, arr)
            
        breakpoints = np.unique(np.percentile(expected, np.linspace(0, 100, buckets + 1)))
        
        # If unique breakpoints are too few (e.g. constant value), just return 0
        if len(breakpoints) < 2:
            return 0.0
            
        # Ensure the boundaries encompass all data
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
        
        expected_percents = replace_zero(expected_percents)
        actual_percents = replace_zero(actual_percents)
        
        psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return float(psi_value)

    def calculate_categorical_psi(self, expected: pd.Series, actual: pd.Series) -> float:
        """Calculate PSI for categorical data."""
        categories = list(set(expected.unique()) | set(actual.unique()))
        
        expected_counts = expected.value_counts(normalize=True).to_dict()
        actual_counts = actual.value_counts(normalize=True).to_dict()
        
        def replace_zero(val):
            return 0.0001 if val == 0 else val
            
        psi_value = 0.0
        for cat in categories:
            e_pct = replace_zero(expected_counts.get(cat, 0))
            a_pct = replace_zero(actual_counts.get(cat, 0))
            psi_value += (a_pct - e_pct) * np.log(a_pct / e_pct)
            
        return float(psi_value)

    def calculate_drift_metrics(self, df_ref: pd.DataFrame, df_curr: pd.DataFrame) -> pd.DataFrame:
        metrics = []
        
        # Define the thresholds
        # < 0.1 : NORMAL, 0.1 - 0.2 : WARNING, > 0.2 : DRIFT
        def get_status(psi):
            if psi < 0.1:
                return "NORMAL"
            elif psi < 0.2:
                return "WARNING"
            else:
                return "DRIFT"
        
        # Features to monitor
        numeric_features = ['payment_amount', 'customer_tenure_months', 'historical_success_rate']
        categorical_features = ['failure_reason', 'payment_method', 'is_subscription']
        
        for feature in numeric_features:
            if feature in df_ref.columns and feature in df_curr.columns:
                psi = self.calculate_psi(df_ref[feature].dropna().values, df_curr[feature].dropna().values)
                metrics.append({
                    "feature": feature,
                    "type": "numeric",
                    "drift_metric_psi": round(psi, 4),
                    "status": get_status(psi)
                })
                
        for feature in categorical_features:
            if feature in df_ref.columns and feature in df_curr.columns:
                psi = self.calculate_categorical_psi(df_ref[feature].dropna(), df_curr[feature].dropna())
                metrics.append({
                    "feature": feature,
                    "type": "categorical",
                    "drift_metric_psi": round(psi, 4),
                    "status": get_status(psi)
                })

        return pd.DataFrame(metrics)

    def simulate_drift(self, df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
        """
        Create a controlled synthetic drift scenario.
        Modifies a copy of the dataframe using a local random generator.
        """
        rng = np.random.default_rng(seed)
        df_drift = df.copy()
        
        if 'payment_amount' in df_drift.columns:
            # Shift payment_amount up by 30% with some noise
            shift = df_drift['payment_amount'] * rng.uniform(1.5, 3.0, size=len(df_drift))
            df_drift['payment_amount'] = shift.clip(lower=10.0) # ensure valid
            
        if 'historical_success_rate' in df_drift.columns:
            # Degrade historical success rate
            degrade = df_drift['historical_success_rate'] - rng.uniform(0.1, 0.3, size=len(df_drift))
            df_drift['historical_success_rate'] = degrade.clip(lower=0.0, upper=1.0)
            
        if 'failure_reason' in df_drift.columns:
            # Shift categorical distribution towards 'insufficient_funds'
            # Select 80% of technical_error and change them
            mask = (df_drift['failure_reason'] == 'technical_error') & (rng.random(len(df_drift)) < 0.8)
            df_drift.loc[mask, 'failure_reason'] = 'insufficient_funds'
            
        return df_drift

if __name__ == "__main__":
    # Smoke test
    engine = MonitoringEngine()
    df_raw = pd.read_csv("data/failed_payments.csv")
    df_scored = pd.read_csv("data/failed_payments_scored.csv")
    
    df_curr = engine.simulate_drift(df_raw)
    
    dq = engine.check_data_quality(df_curr)
    print("Data Quality:", json.dumps(dq, indent=2))
    
    pred_summary = engine.summarize_predictions(df_scored)
    print("Pred Summary Ref:", json.dumps(pred_summary, indent=2))
    
    drift_metrics = engine.calculate_drift_metrics(df_raw, df_curr)
    print("Drift Metrics:")
    print(drift_metrics)

    os.makedirs('reports', exist_ok=True)
    drift_metrics.to_csv("reports/drift_metrics.csv", index=False)