import pandas as pd
import numpy as np

print("="*70)
print("VERIFYING DATA SPLITS")
print("="*70)

# Load splits
print("\nLoading train/test splits...")
X_train = pd.read_csv("data/processed/splits/X_train.csv")
X_test = pd.read_csv("data/processed/splits/X_test.csv")
y_train = pd.read_csv("data/processed/splits/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/splits/y_test.csv").squeeze()

print("[OK] Splits loaded successfully")

# Display shapes
print("\n" + "="*70)
print("DATA SHAPES")
print("="*70)
print(f"\nX_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test:  {y_test.shape}")

# Number of samples
total = len(X_train) + len(X_test)
train_pct = len(X_train) / total * 100
test_pct = len(X_test) / total * 100

print("\n" + "="*70)
print("NUMBER OF SAMPLES")
print("="*70)
print(f"\nTotal samples: {total:,}")
print(f"Training samples: {len(X_train):,} ({train_pct:.2f}%)")
print(f"Test samples:     {len(X_test):,} ({test_pct:.2f}%)")

# Class ratios
print("\n" + "="*70)
print("CLASS DISTRIBUTION & RATIOS")
print("="*70)

# Training set
train_dist = y_train.value_counts().sort_index()
print(f"\nTraining Set:")
print(f"  Total: {len(y_train):,}")
for class_val, count in train_dist.items():
    pct = count / len(y_train) * 100
    bar = '#' * int(pct / 2)
    print(f"  Class {class_val}: {count:,} ({pct:.2f}%) {bar}")

train_ratio = train_dist[1] / len(y_train)
print(f"  Class 1 Ratio: {train_ratio:.4f} ({train_ratio*100:.2f}%)")

# Test set
test_dist = y_test.value_counts().sort_index()
print(f"\nTest Set:")
print(f"  Total: {len(y_test):,}")
for class_val, count in test_dist.items():
    pct = count / len(y_test) * 100
    bar = '#' * int(pct / 2)
    print(f"  Class {class_val}: {count:,} ({pct:.2f}%) {bar}")

test_ratio = test_dist[1] / len(y_test)
print(f"  Class 1 Ratio: {test_ratio:.4f} ({test_ratio*100:.2f}%)")

# Verify stratification
print("\n" + "="*70)
print("STRATIFICATION VERIFICATION")
print("="*70)

original_df = pd.read_csv("data/processed/engineered_data.csv")
original_ratio = original_df['Attrition_Flag'].value_counts()[1] / len(original_df)

print(f"\nClass 1 Ratios:")
print(f"  Original Dataset: {original_ratio:.4f} ({original_ratio*100:.2f}%)")
print(f"  Training Set:     {train_ratio:.4f} ({train_ratio*100:.2f}%)")
print(f"  Test Set:         {test_ratio:.4f} ({test_ratio*100:.2f}%)")

train_diff = abs(original_ratio - train_ratio)
test_diff = abs(original_ratio - test_ratio)

print(f"\nDifference from Original:")
print(f"  Training: {train_diff:.4f} ({train_diff*100:.2f}%)")
print(f"  Test:     {test_diff:.4f} ({test_diff*100:.2f}%)")

if train_diff < 0.01 and test_diff < 0.01:
    print(f"\n[OK] Stratification successful! Class ratios well-balanced.")
else:
    print(f"\n[WARNING] Stratification may have slight imbalance.")

# K-Fold demonstration
print("\n" + "="*70)
print("K-FOLD CROSS-VALIDATION SETUP")
print("="*70)

from sklearn.model_selection import StratifiedKFold

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

print(f"\nConfiguration:")
print(f"  Number of folds: {n_folds}")
print(f"  Shuffle: True")
print(f"  Random seed: 42")
print(f"  Stratification: Enabled")

print(f"\nFold distribution (on training data):")

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    y_val_fold = y_train.iloc[val_idx]
    val_class_dist = y_val_fold.value_counts().sort_index()
    val_ratio = val_class_dist[1] / len(y_val_fold)
    
    print(f"\n  Fold {fold_idx}:")
    print(f"    Train: {len(train_idx):,} samples")
    print(f"    Val:   {len(val_idx):,} samples ({len(val_idx)/len(X_train)*100:.2f}%)")
    print(f"    Val Class 0: {val_class_dist[0]:,} ({val_class_dist[0]/len(y_val_fold)*100:.2f}%)")
    print(f"    Val Class 1: {val_class_dist[1]:,} ({val_class_dist[1]/len(y_val_fold)*100:.2f}%)")
    print(f"    Val Ratio: {val_ratio:.4f}")

print("\n" + "="*70)
print("[OK] DATA SPLITTING COMPLETED SUCCESSFULLY")
print("="*70)

print("\nSummary:")
print(f"  [OK] Train/test split: {train_pct:.1f}% / {test_pct:.1f}%")
print(f"  [OK] Stratification: Maintained")
print(f"  [OK] K-Fold CV: {n_folds} folds configured")
print(f"  [OK] Ready for model training")
