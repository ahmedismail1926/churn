"""Demo script for resampling methods
Tests SMOTE-ENN and other resampling techniques on the training data
"""
from config import PROCESSED_DATA_DIR, RANDOM_SEED
from resampling import (
    resample_data_smoteenn,
    resample_data_smote,
    resample_data_smotetomek,
    resample_data_adasyn,
    compare_resampling_methods
)
import pandas as pd
import os
import sys

# Force UTF-8 encoding for console output to handle special characters
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def main():
    """Main demonstration of resampling methods"""
    
    print("\n" + "="*70)
    print("RESAMPLING METHODS DEMONSTRATION")
    print("="*70)
    print("\nThis script demonstrates various resampling techniques")
    print("to handle class imbalance in churn prediction.")
    
    # Load the training data
    splits_dir = PROCESSED_DATA_DIR / "splits"
    
    if not os.path.exists(splits_dir / "X_train.csv"):
        print("\n[ERROR] Training data not found!")
        print("Please run the following first:")
        print("  1. python main.py (preprocessing)")
        print("  2. python demo_feature_engineering.py (feature engineering)")
        print("  3. python demo_splitting.py (data splitting)")
        exit(1)
    
    print(f"\nLoading training data from: {splits_dir}")
    X_train = pd.read_csv(splits_dir / "X_train.csv")
    y_train = pd.read_csv(splits_dir / "y_train.csv").squeeze()
    
    print(f"[OK] Loaded X_train: {X_train.shape}")
    print(f"[OK] Loaded y_train: {y_train.shape}")
    
    # Demonstrate each resampling method individually
    print("\n" + "="*70)
    print("TESTING INDIVIDUAL METHODS")
    print("="*70)
    
    # 1. SMOTE-ENN (Recommended)
    print("\n[1/4] Testing SMOTE-ENN (Recommended Method)...")
    X_smoteenn, y_smoteenn, stats_smoteenn = resample_data_smoteenn(
        X_train, y_train, 
        random_state=RANDOM_SEED,
        verbose=True
    )
    
    # 2. SMOTE
    print("\n[2/4] Testing SMOTE...")
    X_smote, y_smote, stats_smote = resample_data_smote(
        X_train, y_train, 
        random_state=RANDOM_SEED,
        verbose=True
    )
    
    # 3. SMOTE-Tomek
    print("\n[3/4] Testing SMOTE-Tomek...")
    X_smotetomek, y_smotetomek, stats_smotetomek = resample_data_smotetomek(
        X_train, y_train, 
        random_state=RANDOM_SEED,
        verbose=True
    )
    
    # 4. ADASYN
    print("\n[4/4] Testing ADASYN...")
    X_adasyn, y_adasyn, stats_adasyn = resample_data_adasyn(
        X_train, y_train, 
        random_state=RANDOM_SEED,
        verbose=True
    )
    
    # Compare all methods
    print("\n" + "="*70)
    print("FINAL COMPARISON OF ALL METHODS")
    print("="*70)
    
    print(f"\n{'Method':<20} {'Original':<12} {'Resampled':<12} {'Change':<15} {'Runtime (s)':<12}")
    print("-"*70)
    
    all_stats = [stats_smoteenn, stats_smote, stats_smotetomek, stats_adasyn]
    
    for stats in all_stats:
        method = stats['method']
        orig = stats['original_samples']
        resamp = stats['resampled_samples']
        change = resamp - orig
        change_pct = (resamp / orig - 1) * 100
        runtime = stats['runtime']
        
        print(f"{method:<20} {orig:<12,} {resamp:<12,} {change:+8,} ({change_pct:+5.1f}%) {runtime:<12.3f}")
    
    # Print recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION: SMOTE-ENN")
    print("="*70)
    print("\nWhy SMOTE-ENN is recommended for churn prediction:")
    print("\n1. Balanced Approach:")
    print("   - Oversampling (SMOTE) + Cleaning (ENN)")
    print("   - Creates synthetic samples AND removes noise")
    print("\n2. Better Generalization:")
    print("   - Cleaner decision boundaries")
    print("   - Removes ambiguous/borderline samples")
    print("\n3. Research-Backed:")
    print("   - Proven effective in multiple churn prediction studies")
    print("   - Handles overlap between classes better than pure oversampling")
    print("\n4. Computational Efficiency:")
    print(f"   - Runtime: {stats_smoteenn['runtime']:.3f}s (reasonable for dataset size)")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\n[OK] All {len(all_stats)} resampling methods tested")
    print(f"[OK] Recommended method: SMOTE-ENN")
    print(f"[OK] Resampled data ready for model training")
    print("\nNext steps:")
    print("  - Use resampled data (X_smoteenn, y_smoteenn) for model training")
    print("  - Compare model performance with/without resampling")
    print("  - Evaluate on test set (DO NOT resample test set!)")


if __name__ == "__main__":
    main()