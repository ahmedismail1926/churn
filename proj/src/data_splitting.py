"""
Data Splitting Module
Handles stratified train-test split and K-fold cross-validation setup
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from config import RANDOM_SEED, TEST_SIZE, VALIDATION_SIZE, PROCESSED_DATA_DIR
import pickle


def stratified_train_test_split(X, y, test_size=0.2, random_state=42, print_summary=True):
    """
    Perform stratified train-test split to maintain class balance
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion of test set (default 0.2)
        random_state: Random seed for reproducibility
        print_summary: Whether to print split summary
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    if print_summary:
        print("\n" + "="*60)
        print("STRATIFIED TRAIN-TEST SPLIT")
        print("="*60)
    
    # Perform stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )
    
    if print_summary:
        print(f"\nSplit Configuration:")
        print(f"  Test size: {test_size*100:.1f}%")
        print(f"  Random seed: {random_state}")
        print(f"  Stratification: Enabled (maintains class ratio)")
        
        print(f"\n{'Dataset Statistics':^60}")
        print("-"*60)
        
        # Overall statistics
        total_samples = len(X)
        train_samples = len(X_train)
        test_samples = len(X_test)
        
        print(f"\nTotal samples: {total_samples:,}")
        print(f"  Training samples: {train_samples:,} ({train_samples/total_samples*100:.2f}%)")
        print(f"  Test samples:     {test_samples:,} ({test_samples/total_samples*100:.2f}%)")
        
        # Class distribution analysis
        print(f"\n{'Class Distribution':^60}")
        print("-"*60)
        
        # Original distribution
        original_dist = y.value_counts().sort_index()
        print(f"\nOriginal Dataset:")
        for class_val, count in original_dist.items():
            pct = count / len(y) * 100
            bar = '#' * int(pct / 2)
            print(f"  Class {class_val}: {count:,} ({pct:.2f}%) {bar}")
        
        # Training set distribution
        train_dist = y_train.value_counts().sort_index()
        print(f"\nTraining Set:")
        for class_val, count in train_dist.items():
            pct = count / len(y_train) * 100
            bar = '#' * int(pct / 2)
            print(f"  Class {class_val}: {count:,} ({pct:.2f}%) {bar}")
        
        # Test set distribution
        test_dist = y_test.value_counts().sort_index()
        print(f"\nTest Set:")
        for class_val, count in test_dist.items():
            pct = count / len(y_test) * 100
            bar = '#' * int(pct / 2)
            print(f"  Class {class_val}: {count:,} ({pct:.2f}%) {bar}")
        
        # Verify stratification maintained balance
        print(f"\n{'Stratification Verification':^60}")
        print("-"*60)
        print("\nClass Ratios (Class 1 / Total):")
        original_ratio = original_dist[1] / len(y)
        train_ratio = train_dist[1] / len(y_train)
        test_ratio = test_dist[1] / len(y_test)
        
        print(f"  Original: {original_ratio:.4f} ({original_ratio*100:.2f}%)")
        print(f"  Training: {train_ratio:.4f} ({train_ratio*100:.2f}%)")
        print(f"  Test:     {test_ratio:.4f} ({test_ratio*100:.2f}%)")
        
        difference_train = abs(original_ratio - train_ratio)
        difference_test = abs(original_ratio - test_ratio)
        
        print(f"\nDifference from Original:")
        print(f"  Training: {difference_train:.4f} ({difference_train*100:.2f}%)")
        print(f"  Test:     {difference_test:.4f} ({difference_test*100:.2f}%)")
        
        if difference_train < 0.01 and difference_test < 0.01:
            print(f"\n[OK] Stratification successful! Class ratios are well-balanced.")
        else:
            print(f"\n[WARNING] Warning: Stratification may have slight imbalance.")
    
    return X_train, X_test, y_train, y_test


def create_stratified_kfold(n_splits=5, shuffle=True, random_state=42):
    """
    Create a stratified K-fold cross-validation generator
    
    Args:
        n_splits: Number of folds (default 5)
        shuffle: Whether to shuffle data before splitting
        random_state: Random seed for reproducibility
    
    Returns:
        StratifiedKFold object
    """
    print("\n" + "="*60)
    print("STRATIFIED K-FOLD CROSS-VALIDATION SETUP")
    print("="*60)
    
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )
    
    print(f"\nConfiguration:")
    print(f"  Number of folds: {n_splits}")
    print(f"  Shuffle: {shuffle}")
    print(f"  Random seed: {random_state}")
    print(f"  Stratification: Enabled")
    
    print(f"\nHow it works:")
    print(f"  1. Data is split into {n_splits} folds")
    print(f"  2. Each fold maintains the same class distribution")
    print(f"  3. Each fold is used once as validation set")
    print(f"  4. Remaining {n_splits-1} folds are used for training")
    print(f"  5. Process repeats {n_splits} times")
    
    train_pct = ((n_splits - 1) / n_splits) * 100
    val_pct = (1 / n_splits) * 100
    
    print(f"\nEach iteration:")
    print(f"  Training set: ~{train_pct:.1f}% of data")
    print(f"  Validation set: ~{val_pct:.1f}% of data")
    
    print(f"\n[OK] K-Fold generator created successfully!")
    
    return skf


def demonstrate_kfold_splits(X, y, skf, max_folds_to_show=5):
    """
    Demonstrate how K-fold splits the data
    
    Args:
        X: Features
        y: Target
        skf: StratifiedKFold object
        max_folds_to_show: Maximum number of folds to display details
    """
    print("\n" + "="*60)
    print("K-FOLD SPLIT DEMONSTRATION")
    print("="*60)
    
    total_samples = len(X)
    
    print(f"\nTotal samples: {total_samples:,}")
    print(f"\nSplit details for each fold:")
    print("-"*60)
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        if fold_idx <= max_folds_to_show:
            print(f"\nFold {fold_idx}:")
            
            # Training set info
            train_size = len(train_idx)
            train_pct = train_size / total_samples * 100
            y_train_fold = y.iloc[train_idx]
            train_class_dist = y_train_fold.value_counts().sort_index()
            train_class_ratio = train_class_dist[1] / len(y_train_fold) if 1 in train_class_dist.index else 0
            
            print(f"  Training: {train_size:,} samples ({train_pct:.2f}%)")
            print(f"    Class 0: {train_class_dist[0]:,} ({train_class_dist[0]/train_size*100:.2f}%)")
            print(f"    Class 1: {train_class_dist[1]:,} ({train_class_dist[1]/train_size*100:.2f}%)")
            print(f"    Ratio (Class 1): {train_class_ratio:.4f}")
            
            # Validation set info
            val_size = len(val_idx)
            val_pct = val_size / total_samples * 100
            y_val_fold = y.iloc[val_idx]
            val_class_dist = y_val_fold.value_counts().sort_index()
            val_class_ratio = val_class_dist[1] / len(y_val_fold) if 1 in val_class_dist.index else 0
            
            print(f"  Validation: {val_size:,} samples ({val_pct:.2f}%)")
            print(f"    Class 0: {val_class_dist[0]:,} ({val_class_dist[0]/val_size*100:.2f}%)")
            print(f"    Class 1: {val_class_dist[1]:,} ({val_class_dist[1]/val_size*100:.2f}%)")
            print(f"    Ratio (Class 1): {val_class_ratio:.4f}")
        
        if fold_idx == max_folds_to_show and skf.n_splits > max_folds_to_show:
            remaining = skf.n_splits - max_folds_to_show
            print(f"\n  ... and {remaining} more fold(s) with similar distribution")
            break
    
    # Summary statistics across all folds
    print(f"\n{'Summary Across All Folds':^60}")
    print("-"*60)
    
    class_ratios = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        y_val_fold = y.iloc[val_idx]
        val_class_dist = y_val_fold.value_counts().sort_index()
        ratio = val_class_dist[1] / len(y_val_fold) if 1 in val_class_dist.index else 0
        class_ratios.append(ratio)
    
    mean_ratio = np.mean(class_ratios)
    std_ratio = np.std(class_ratios)
    
    print(f"\nClass 1 ratio across folds:")
    print(f"  Mean: {mean_ratio:.4f} ({mean_ratio*100:.2f}%)")
    print(f"  Std:  {std_ratio:.6f}")
    print(f"  Min:  {min(class_ratios):.4f}")
    print(f"  Max:  {max(class_ratios):.4f}")
    
    print(f"\n[OK] All folds maintain consistent class distribution!")


def prepare_data_for_modeling(filepath=None, target_column='Attrition_Flag', 
                              test_size=0.2, n_folds=5, random_state=42,
                              save_splits=True):
    """
    Complete data preparation pipeline for modeling
    
    Args:
        filepath: Path to engineered data CSV
        target_column: Name of target column
        test_size: Test set proportion
        n_folds: Number of folds for cross-validation
        random_state: Random seed
        save_splits: Whether to save train/test splits
    
    Returns:
        Dictionary with X_train, X_test, y_train, y_test, skf
    """
    print("\n" + "="*60)
    print("DATA PREPARATION FOR MODELING")
    print("="*60)
    
    # Load data
    if filepath is None:
        filepath = PROCESSED_DATA_DIR / "engineered_data.csv"
    
    print(f"\nLoading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[OK] Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    # Separate features and target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data!")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"\nFeatures (X): {X.shape[1]} columns")
    print(f"Target (y): {target_column}")
    print(f"\nFeature names:")
    for i, col in enumerate(X.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Stratified train-test split
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        print_summary=True
    )
    
    # Create K-fold generator
    skf = create_stratified_kfold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state
    )
    
    # Demonstrate K-fold splits on training data
    demonstrate_kfold_splits(X_train, y_train, skf, max_folds_to_show=3)
    
    # Save splits if requested
    if save_splits:
        splits_dir = PROCESSED_DATA_DIR / "splits"
        splits_dir.mkdir(exist_ok=True)
        
        # Save as CSV
        X_train.to_csv(splits_dir / "X_train.csv", index=False)
        X_test.to_csv(splits_dir / "X_test.csv", index=False)
        y_train.to_csv(splits_dir / "y_train.csv", index=False)
        y_test.to_csv(splits_dir / "y_test.csv", index=False)
        
        # Save as pickle for faster loading
        splits_dict = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': X.columns.tolist(),
            'target_name': target_column
        }
        
        with open(splits_dir / "train_test_splits.pkl", 'wb') as f:
            pickle.dump(splits_dict, f)
        
        print(f"\n[OK] Saved train/test splits to: {splits_dir}")
        print(f"  Files: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
        print(f"  Pickle: train_test_splits.pkl")
    
    # Final summary
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"\n[OK] Train set: {len(X_train):,} samples")
    print(f"[OK] Test set: {len(X_test):,} samples")
    print(f"[OK] Features: {X.shape[1]}")
    print(f"[OK] K-Fold CV: {n_folds} folds")
    print(f"[OK] Stratification: Enabled")
    print(f"[OK] Random seed: {random_state}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'skf': skf,
        'feature_names': X.columns.tolist(),
        'target_name': target_column
    }
