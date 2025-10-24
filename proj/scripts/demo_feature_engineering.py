"""
Simple Feature Engineering Demo
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path

print("="*70)
print("FEATURE ENGINEERING DEMO")
print("="*70)

# Load data
print("\n1. Loading processed data...")
df = pd.read_csv("data/processed/processed_data.csv")
print(f"   Shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")

# Store original for comparison
df_original = df.copy()

# ============================================================
# STEP 1: Label Encode Binary Columns
# ============================================================
print("\n" + "="*70)
print("STEP 1: LABEL ENCODING BINARY COLUMNS")
print("="*70)

binary_cols = []
for col in df.select_dtypes(include=['object']).columns:
    if df[col].nunique() == 2:
        binary_cols.append(col)

print(f"\nFound {len(binary_cols)} binary columns: {binary_cols}")

label_encoders = {}
for col in binary_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    
    # Print mapping
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"\n{col}:")
    for original, encoded in mapping.items():
        print(f"  '{original}' -> {encoded}")

# ============================================================
# STEP 2: One-Hot Encode Multi-Category Columns
# ============================================================
print("\n" + "="*70)
print("STEP 2: ONE-HOT ENCODING MULTI-CATEGORY COLUMNS")
print("="*70)

multi_category_cols = []
for col in df.select_dtypes(include=['object']).columns:
    if df[col].nunique() > 2:
        multi_category_cols.append(col)

print(f"\nFound {len(multi_category_cols)} multi-category columns:")

new_columns = []
for col in multi_category_cols:
    n_unique = df[col].nunique()
    unique_vals = df[col].unique()
    print(f"\n{col}: {n_unique} unique values")
    print(f"  Values: {list(unique_vals)}")
    
    # One-hot encode
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_cols = dummies.columns.tolist()
    new_columns.extend(new_cols)
    
    print(f"  Created {len(new_cols)} binary columns:")
    for new_col in new_cols:
        print(f"    * {new_col}")
    
    # Add to dataframe
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=[col])

print(f"\n✓ Total new features created: {len(new_columns)}")

# ============================================================
# STEP 3: Standardize Numerical Columns
# ============================================================
print("\n" + "="*70)
print("STEP 3: STANDARDIZING NUMERICAL FEATURES")
print("="*70)

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Remove already encoded binary columns
numerical_cols = [col for col in numerical_cols if col not in binary_cols]

print(f"\nScaling {len(numerical_cols)} numerical features")
print(f"Method: StandardScaler (Z-score normalization)")
print(f"Formula: (X - mean) / std")

print(f"\nNumerical features to scale:")
for col in numerical_cols:
    print(f"  * {col}")

# Show before stats
print(f"\n{'Before Scaling (First 5 features)':^70}")
print("-"*70)
for col in numerical_cols[:5]:
    mean_val = df[col].mean()
    std_val = df[col].std()
    print(f"{col:35s} | μ={mean_val:10.2f}  σ={std_val:10.2f}")

# Scale
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Show after stats
print(f"\n{'After Scaling (First 5 features)':^70}")
print("-"*70)
for col in numerical_cols[:5]:
    mean_val = df[col].mean()
    std_val = df[col].std()
    print(f"{col:35s} | μ={mean_val:10.4f}  σ={std_val:10.4f}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("TRANSFORMATION SUMMARY")
print("="*70)

print(f"\nOriginal shape:  {df_original.shape}")
print(f"Final shape:     {df.shape}")
print(f"Features added:  {df.shape[1] - df_original.shape[1]}")

print(f"\nTransformations applied:")
print(f"  * Binary columns label-encoded:        {len(binary_cols)}")
print(f"  * Multi-category columns one-hot encoded: {len(multi_category_cols)}")
print(f"  * New binary features created:         {len(new_columns)}")
print(f"  * Numerical features standardized:     {len(numerical_cols)}")

print(f"\n{'TRANSFORMED FEATURE NAMES':^70}")
print("="*70)
print(f"\nTotal features: {len(df.columns)}")

print(f"\nLabel-encoded (Binary):")
for col in binary_cols:
    if col in df.columns:
        print(f"  * {col}")

print(f"\nOne-hot encoded:")
for col in new_columns:
    print(f"  * {col}")

print(f"\nNumerical (Scaled):")
for col in numerical_cols:
    print(f"  * {col}")

print(f"\n{'SAMPLE OF SCALED VALUES (First 5 rows × First 5 features)':^70}")
print("="*70)
sample_cols = numerical_cols[:5]
print(df[sample_cols].head().to_string())

print(f"\n{'BEFORE vs AFTER (First 3 rows × First 3 features)':^70}")
print("="*70)
comparison_cols = numerical_cols[:3]
print("\nBefore (Original):")
print(df_original[comparison_cols].head(3).to_string())
print("\nAfter (Scaled):")
print(df[comparison_cols].head(3).to_string())

# Save
output_path = Path("data/processed/engineered_data.csv")
df.to_csv(output_path, index=False)
print(f"\n✓ Saved engineered data to: {output_path}")
print(f"  Shape: {df.shape}")

print("\n" + "="*70)
print("✓ FEATURE ENGINEERING COMPLETED")
print("="*70)
