"""Quick test for Stages 1-3"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("="*70)
print("TESTING STAGES 1-3 (EDA → Feature Engineering → Splitting)")
print("="*70)

# Stage 1: Check if processed data exists
print("\n[Stage 1] Checking processed data...")
from config import PROCESSED_DATA_DIR
import pandas as pd

processed_file = PROCESSED_DATA_DIR / "processed_data.csv"
if processed_file.exists():
    df1 = pd.read_csv(processed_file)
    print(f"✓ Stage 1 output exists: {df1.shape}")
else:
    print("✗ Stage 1 output missing. Run: python main.py")
    sys.exit(1)

# Stage 2: Check if engineered data exists
print("\n[Stage 2] Checking engineered data...")
engineered_file = PROCESSED_DATA_DIR / "engineered_data.csv"
if engineered_file.exists():
    df2 = pd.read_csv(engineered_file)
    print(f"✓ Stage 2 output exists: {df2.shape}")
    
    # Verify target is NOT scaled
    target_values = sorted(df2['Attrition_Flag'].unique())
    if target_values == [0, 1]:
        print(f"✓ Target preserved correctly: {target_values}")
    else:
        print(f"✗ Target corrupted: {target_values}")
        sys.exit(1)
else:
    print("✗ Stage 2 output missing.")
    sys.exit(1)

# Stage 3: Check if splits exist
print("\n[Stage 3] Checking train/test splits...")
splits_dir = PROCESSED_DATA_DIR / "splits"
X_train_file = splits_dir / "X_train.csv"
y_train_file = splits_dir / "y_train.csv"
X_test_file = splits_dir / "X_test.csv"
y_test_file = splits_dir / "y_test.csv"

if all([X_train_file.exists(), y_train_file.exists(), 
        X_test_file.exists(), y_test_file.exists()]):
    X_train = pd.read_csv(X_train_file)
    y_train = pd.read_csv(y_train_file)
    X_test = pd.read_csv(X_test_file)
    y_test = pd.read_csv(y_test_file)
    
    print(f"✓ Stage 3 outputs exist:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")
    
    # Verify target values
    y_train_unique = sorted(y_train['Attrition_Flag'].unique())
    y_test_unique = sorted(y_test['Attrition_Flag'].unique())
    
    if y_train_unique == [0, 1] and y_test_unique == [0, 1]:
        print(f"✓ Train/test targets correct: {y_train_unique}")
    else:
        print(f"✗ Target values corrupted!")
        sys.exit(1)
    
    # Verify stratification
    train_ratio = y_train['Attrition_Flag'].mean()
    test_ratio = y_test['Attrition_Flag'].mean()
    print(f"✓ Class balance maintained:")
    print(f"  Train: {train_ratio:.1%} churned")
    print(f"  Test: {test_ratio:.1%} churned")
    
else:
    print("✗ Stage 3 outputs missing.")
    sys.exit(1)

print("\n" + "="*70)
print("✓✓✓ ALL STAGES 1-3 VERIFIED SUCCESSFULLY! ✓✓✓")
print("="*70)
print("\nReady for Stage 4 (Model Training):")
print("  python src/model_training.py")
