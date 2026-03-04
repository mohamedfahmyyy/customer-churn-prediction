"""
Data Validation Script
Validates the ML dataset exported from BigQuery
Checks shape, target distribution, missing values, and data quality
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw"

def validate_data(filepath):
    """Load and validate the dataset"""
    
    print("="*80)
    print("DATA VALIDATION REPORT")
    print("="*80)
    
    
    df = pd.read_csv(filepath)
    
    
    print(f"\nShape: {df.shape}")
    print(f"Columns: {df.shape[1]}")
    print(f"Rows: {df.shape[0]}")
    
    
    print(f"\nTarget Distribution:")
    print(df['target'].value_counts(normalize=True))
    
    expected_repeat_rate = 0.655
    actual_repeat_rate = df['target'].mean()
    
    if abs(actual_repeat_rate - expected_repeat_rate) < 0.01:
        print(f"Target distribution looks good (approximately 65% repeat buyers)")
    else:
        print(f"WARNING: Expected around {expected_repeat_rate:.1%} repeat rate, but got {actual_repeat_rate:.1%}")
    
    
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
        
        if 'days_to_second_purchase' in missing.index:
            nulls = missing['days_to_second_purchase']
            expected_nulls = (df['target'] == 0).sum()
            if nulls == expected_nulls:
                print(f"Note: {nulls} nulls in days_to_second_purchase matches the {expected_nulls} one-time buyers")
            else:
                print(f"WARNING: Expected {expected_nulls} nulls in days_to_second_purchase, but got {nulls}")
    else:
        print("No missing values found")
    
    
    print(f"\nChecking Key Features:")
    required_features = [
        'basket_repeat_score', 
        'best_product_repeat_score',
        'order_value',
        'month',
        'Country'
    ]
    
    missing_features = []
    for feature in required_features:
        if feature in df.columns:
            print(f"  {feature}: present")
        else:
            print(f"  {feature}: MISSING")
            missing_features.append(feature)
    
    if missing_features:
        print(f"\nERROR: Missing required features: {missing_features}")
        sys.exit(1)
    
    
    print(f"\nData Quality Checks:")
    
    
    if (df['order_value'] <= 0).any():
        print(f"  WARNING: Found {(df['order_value'] <= 0).sum()} negative or zero order values")
    else:
        print(f"  All order values are positive (minimum: ${df['order_value'].min():.2f})")
    
    
    if (df['basket_repeat_score'] < 0).any() or (df['basket_repeat_score'] > 1).any():
        print("  WARNING: basket_repeat_score has values outside the 0-1 range")
    else:
        print(f"  basket_repeat_score values are within valid range (0 to 1)")
    
    print("\n" + "="*80)
    print("Validation complete - data looks ready for modeling")
    print("="*80)
    
    return df

if __name__ == "__main__":
   
    csv_files = list(DATA_PATH.glob("*.csv"))
    
    if len(csv_files) == 0:
        print("ERROR: No CSV file found in data/raw/")
        sys.exit(1)
    elif len(csv_files) > 1:
        print("WARNING: Multiple CSV files found. Using the most recent one.")
        csv_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    else:
        csv_file = csv_files[0]
    
    print(f"Loading: {csv_file.name}\n")
    df = validate_data(csv_file)
    
    print(f"\nDataset is ready for modeling")
    print(f"Next step: python src/02_eda_correlations.py")