"""
Simple Preprocessing Demo
"""
import pandas as pd
from pathlib import Path

# Load data
print("Loading data...")
df = pd.read_csv("data/raw/BankChurners.csv")
print(f"Original shape: {df.shape}")

# Show columns to be dropped
print("\n" + "="*60)
print("COLUMNS TO BE DROPPED")
print("="*60)

cols_to_drop = []

# CLIENTNUM (identifier)
if 'CLIENTNUM' in df.columns:
    cols_to_drop.append('CLIENTNUM')
    print("\n1. CLIENTNUM - Identifier (no predictive value)")

# Naive Bayes columns (target leakage)
naive_bayes_cols = [col for col in df.columns if col.startswith('Naive_Bayes')]
for col in naive_bayes_cols:
    cols_to_drop.append(col)
print(f"\n2. Naive Bayes columns ({len(naive_bayes_cols)}):")
for col in naive_bayes_cols:
    print(f"   - {col}")

# Check Credit_Limit correlation
print("\n3. Checking Credit_Limit correlation...")
if 'Credit_Limit' in df.columns and 'Avg_Open_To_Buy' in df.columns:
    corr = df[['Credit_Limit', 'Avg_Open_To_Buy']].corr().iloc[0, 1]
    print(f"   Correlation: {corr:.4f}")
    if corr > 0.95:
        cols_to_drop.append('Credit_Limit')
        print(f"   → Credit_Limit will be dropped (correlation > 0.95)")
    else:
        print(f"   → Credit_Limit will be kept (correlation < 0.95)")

# Drop columns
print("\n" + "="*60)
print("DROPPING COLUMNS")
print("="*60)
print(f"\nTotal columns to drop: {len(cols_to_drop)}")
df_cleaned = df.drop(columns=cols_to_drop)

print(f"\nShape before: {df.shape}")
print(f"Shape after:  {df_cleaned.shape}")
print(f"Columns removed: {df.shape[1] - df_cleaned.shape[1]}")

# Check missing values
print("\n" + "="*60)
print("MISSING VALUES CHECK")
print("="*60)
missing_total = df_cleaned.isnull().sum().sum()
print(f"Total missing values: {missing_total}")

if missing_total == 0:
    print("✓ No missing values! No rows need to be dropped or imputed.")
    rows_dropped = 0
    values_imputed = 0
else:
    print("Missing values found:")
    missing = df_cleaned.isnull().sum()
    for col in missing[missing > 0].index:
        print(f"  - {col}: {missing[col]} ({missing[col]/len(df_cleaned)*100:.2f}%)")
    rows_dropped = len(df_cleaned[df_cleaned.isnull().any(axis=1)])
    values_imputed = 0

# Final summary
print("\n" + "="*60)
print("PREPROCESSING SUMMARY")
print("="*60)
print(f"Initial shape:        {df.shape}")
print(f"Final shape:          {df_cleaned.shape}")
print(f"Rows removed:         {df.shape[0] - df_cleaned.shape[0]}")
print(f"Columns removed:      {df.shape[1] - df_cleaned.shape[1]}")
print(f"Rows dropped (missing): {rows_dropped}")
print(f"Values imputed:       {values_imputed}")
print(f"Remaining null counts: {df_cleaned.isnull().sum().sum()}")

# Save
output_path = Path("data/processed/processed_data.csv")
df_cleaned.to_csv(output_path, index=False)
print(f"\n✓ Saved to: {output_path}")

print("\n" + "="*60)
print("REMAINING COLUMNS")
print("="*60)
print(f"Total: {len(df_cleaned.columns)}")
for i, col in enumerate(df_cleaned.columns, 1):
    print(f"{i:2d}. {col}")
