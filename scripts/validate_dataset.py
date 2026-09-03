import pandas as pd
import numpy as np

def validate_dataset(filepath):
    print(f"Validating dataset: {filepath}")
    df = pd.read_csv(filepath)
    
    print("\n--- Basic Checks ---")
    print(f"Row count: {len(df)}")
    print(f"Column count: {len(df.columns)}")
    
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values.")
    
    print("\n--- Duplicates ---")
    dupes = df.duplicated().sum()
    print(f"Duplicate rows: {dupes}")
    
    print("\n--- Data Types ---")
    print(df.dtypes)
    
    print("\n--- Class Distribution ---")
    print(df['recovered'].value_counts(normalize=True))
    
    print("\n--- Impossible Values Check ---")
    issues = []
    if (df['payment_amount'] < 0).any(): issues.append("Negative payment amounts found.")
    if (df['historical_success_rate'] < 0).any() or (df['historical_success_rate'] > 1).any():
        issues.append("historical_success_rate out of bounds [0, 1].")
    if (df['is_subscription'].isin([0, 1]) == False).any():
        issues.append("is_subscription contains invalid values.")
    if (df['recovered'].isin([0, 1]) == False).any():
        issues.append("Target 'recovered' contains invalid values.")
        
    if issues:
        for i in issues: print(f"ERROR: {i}")
    else:
        print("No impossible values detected.")
        
    print("\n--- Suspicious Target Leakage Check ---")
    num_cols = df.select_dtypes(include=[np.number]).columns.drop('recovered')
    found_leakage = False
    for col in num_cols:
        corr = df[col].corr(df['recovered'])
        if abs(corr) > 0.8:
            print(f"WARNING: High correlation ({corr:.2f}) between {col} and target. Possible leakage.")
            found_leakage = True
            
    if not found_leakage:
        print("No suspiciously high correlation found with numerical features.")
        
    print("\nValidation Complete.")

if __name__ == "__main__":
    validate_dataset("data/failed_payments.csv")
