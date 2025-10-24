"""
Resampling Module
Implements various resampling techniques to handle class imbalance
SMOTE-ENN is recommended based on research for churn prediction
"""
import pandas as pd
import numpy as np
from collections import Counter
import time
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from config import RANDOM_SEED


def print_class_distribution(y, dataset_name="Dataset", indent=""):
    """
    Print class distribution in a formatted way
    
    Args:
        y: Target variable (Series or array)
        dataset_name: Name of the dataset for display
        indent: Indentation for nested display
    """
    if isinstance(y, pd.Series):
        counts = y.value_counts().sort_index().to_dict()
    else:
        counts = dict(Counter(y))
        counts = dict(sorted(counts.items()))
    
    total = sum(counts.values())
    
    print(f"{indent}{dataset_name}:")
    for class_label, count in counts.items():
        pct = count / total * 100
        bar = '#' * int(pct / 2)
        class_name = "Existing" if class_label == 0 else "Churned"
        print(f"{indent}  Class {class_label} ({class_name}): {count:,} ({pct:.2f}%) {bar}")
    
    # Calculate imbalance ratio
    if len(counts) == 2:
        minority_count = min(counts.values())
        majority_count = max(counts.values())
        imbalance_ratio = majority_count / minority_count
        print(f"{indent}  Imbalance Ratio: {imbalance_ratio:.2f}:1")
    
    return counts


def resample_data_smoteenn(X_train, y_train, random_state=RANDOM_SEED, verbose=True):
    """
    Apply SMOTE-ENN (SMOTE + Edited Nearest Neighbours) resampling
    Best method according to research papers for churn prediction
    
    SMOTE-ENN combines:
    1. SMOTE: Generates synthetic minority class samples
    2. ENN: Cleans borderline/noisy samples from both classes
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed for reproducibility
        verbose: Whether to print detailed information
    
    Returns:
        X_resampled, y_resampled, resampling_stats
    """
    if verbose:
        print("\n" + "="*70)
        print("RESAMPLING: SMOTE-ENN (Recommended Method)")
        print("="*70)
        print("\nMethod: SMOTE + Edited Nearest Neighbours")
        print("  • SMOTE: Creates synthetic minority samples using k-nearest neighbors")
        print("  • ENN: Removes samples whose class differs from majority of neighbors")
        print("  • Result: Balanced classes with cleaner decision boundaries")
    
    # Print original distribution
    if verbose:
        print("\n" + "-"*70)
        print("BEFORE RESAMPLING:")
        print("-"*70)
        original_counts = print_class_distribution(y_train, "Training Set", indent="")
    
    # Apply SMOTE-ENN
    start_time = time.time()
    
    smoteenn = SMOTEENN(
        random_state=random_state,
        sampling_strategy='auto',  # Balance to 1:1 ratio
        smote=SMOTE(
            random_state=random_state,
            k_neighbors=5
        ),
        enn=EditedNearestNeighbours(
            n_neighbors=3,
            kind_sel='all'
        )
    )
    
    X_resampled, y_resampled = smoteenn.fit_resample(X_train, y_train)
    
    runtime = time.time() - start_time
    
    # Print resampled distribution
    if verbose:
        print("\n" + "-"*70)
        print("AFTER RESAMPLING:")
        print("-"*70)
        resampled_counts = print_class_distribution(y_resampled, "Resampled Training Set", indent="")
        
        print("\n" + "-"*70)
        print("RESAMPLING SUMMARY:")
        print("-"*70)
        
        original_total = len(y_train)
        resampled_total = len(y_resampled)
        
        print(f"\nTotal samples:")
        print(f"  Before: {original_total:,}")
        print(f"  After:  {resampled_total:,}")
        print(f"  Change: {resampled_total - original_total:+,} ({(resampled_total/original_total - 1)*100:+.2f}%)")
        
        print(f"\nRuntime: {runtime:.3f} seconds")
        
        print(f"\n[OK] SMOTE-ENN resampling completed successfully!")
    
    stats = {
        'method': 'SMOTE-ENN',
        'original_samples': len(y_train),
        'resampled_samples': len(y_resampled),
        'original_distribution': dict(Counter(y_train)),
        'resampled_distribution': dict(Counter(y_resampled)),
        'runtime': runtime
    }
    
    return X_resampled, y_resampled, stats


def resample_data_smote(X_train, y_train, random_state=RANDOM_SEED, verbose=True):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) resampling
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed for reproducibility
        verbose: Whether to print detailed information
    
    Returns:
        X_resampled, y_resampled, resampling_stats
    """
    if verbose:
        print("\n" + "="*70)
        print("RESAMPLING: SMOTE")
        print("="*70)
        print("\nMethod: Synthetic Minority Over-sampling Technique")
        print("  • Creates synthetic samples by interpolating between minority samples")
    
    # Print original distribution
    if verbose:
        print("\n" + "-"*70)
        print("BEFORE RESAMPLING:")
        print("-"*70)
        print_class_distribution(y_train, "Training Set", indent="")
    
    # Apply SMOTE
    start_time = time.time()
    
    smote = SMOTE(
        random_state=random_state,
        sampling_strategy='auto',
        k_neighbors=5
    )
    
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    runtime = time.time() - start_time
    
    # Print resampled distribution
    if verbose:
        print("\n" + "-"*70)
        print("AFTER RESAMPLING:")
        print("-"*70)
        print_class_distribution(y_resampled, "Resampled Training Set", indent="")
        
        print("\n" + "-"*70)
        print("RESAMPLING SUMMARY:")
        print("-"*70)
        print(f"  Total samples: {len(y_train):,} -> {len(y_resampled):,}")
        print(f"  Runtime: {runtime:.3f} seconds")
        print(f"\n[OK] SMOTE resampling completed!")
    
    stats = {
        'method': 'SMOTE',
        'original_samples': len(y_train),
        'resampled_samples': len(y_resampled),
        'original_distribution': dict(Counter(y_train)),
        'resampled_distribution': dict(Counter(y_resampled)),
        'runtime': runtime
    }
    
    return X_resampled, y_resampled, stats


def resample_data_smotetomek(X_train, y_train, random_state=RANDOM_SEED, verbose=True):
    """
    Apply SMOTE-Tomek resampling
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed for reproducibility
        verbose: Whether to print detailed information
    
    Returns:
        X_resampled, y_resampled, resampling_stats
    """
    if verbose:
        print("\n" + "="*70)
        print("RESAMPLING: SMOTE-Tomek")
        print("="*70)
        print("\nMethod: SMOTE + Tomek Links removal")
        print("  • SMOTE: Creates synthetic minority samples")
        print("  • Tomek: Removes Tomek links (pairs of opposite classes that are nearest neighbors)")
    
    # Print original distribution
    if verbose:
        print("\n" + "-"*70)
        print("BEFORE RESAMPLING:")
        print("-"*70)
        print_class_distribution(y_train, "Training Set", indent="")
    
    # Apply SMOTE-Tomek
    start_time = time.time()
    
    smotetomek = SMOTETomek(
        random_state=random_state,
        sampling_strategy='auto',
        smote=SMOTE(random_state=random_state, k_neighbors=5),
        tomek=TomekLinks(sampling_strategy='all')
    )
    
    X_resampled, y_resampled = smotetomek.fit_resample(X_train, y_train)
    
    runtime = time.time() - start_time
    
    # Print resampled distribution
    if verbose:
        print("\n" + "-"*70)
        print("AFTER RESAMPLING:")
        print("-"*70)
        print_class_distribution(y_resampled, "Resampled Training Set", indent="")
        
        print("\n" + "-"*70)
        print("RESAMPLING SUMMARY:")
        print("-"*70)
        print(f"  Total samples: {len(y_train):,} -> {len(y_resampled):,}")
        print(f"  Runtime: {runtime:.3f} seconds")
        print(f"\n[OK] SMOTE-Tomek resampling completed!")
    
    stats = {
        'method': 'SMOTE-Tomek',
        'original_samples': len(y_train),
        'resampled_samples': len(y_resampled),
        'original_distribution': dict(Counter(y_train)),
        'resampled_distribution': dict(Counter(y_resampled)),
        'runtime': runtime
    }
    
    return X_resampled, y_resampled, stats


def resample_data_adasyn(X_train, y_train, random_state=RANDOM_SEED, verbose=True):
    """
    Apply ADASYN (Adaptive Synthetic Sampling) resampling
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed for reproducibility
        verbose: Whether to print detailed information
    
    Returns:
        X_resampled, y_resampled, resampling_stats
    """
    if verbose:
        print("\n" + "="*70)
        print("RESAMPLING: ADASYN")
        print("="*70)
        print("\nMethod: Adaptive Synthetic Sampling")
        print("  • Focuses on generating samples near decision boundary")
    
    # Print original distribution
    if verbose:
        print("\n" + "-"*70)
        print("BEFORE RESAMPLING:")
        print("-"*70)
        print_class_distribution(y_train, "Training Set", indent="")
    
    # Apply ADASYN
    start_time = time.time()
    
    adasyn = ADASYN(
        random_state=random_state,
        sampling_strategy='auto',
        n_neighbors=5
    )
    
    X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
    
    runtime = time.time() - start_time
    
    # Print resampled distribution
    if verbose:
        print("\n" + "-"*70)
        print("AFTER RESAMPLING:")
        print("-"*70)
        print_class_distribution(y_resampled, "Resampled Training Set", indent="")
        
        print("\n" + "-"*70)
        print("RESAMPLING SUMMARY:")
        print("-"*70)
        print(f"  Total samples: {len(y_train):,} -> {len(y_resampled):,}")
        print(f"  Runtime: {runtime:.3f} seconds")
        print(f"\n[OK] ADASYN resampling completed!")
    
    stats = {
        'method': 'ADASYN',
        'original_samples': len(y_train),
        'resampled_samples': len(y_resampled),
        'original_distribution': dict(Counter(y_train)),
        'resampled_distribution': dict(Counter(y_resampled)),
        'runtime': runtime
    }
    
    return X_resampled, y_resampled, stats


def compare_resampling_methods(X_train, y_train, random_state=RANDOM_SEED):
    """
    Compare different resampling methods
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with results from all methods
    """
    print("\n" + "="*70)
    print("COMPARING RESAMPLING METHODS")
    print("="*70)
    
    methods = {
        'SMOTE-ENN': resample_data_smoteenn,
        'SMOTE': resample_data_smote,
        'SMOTE-Tomek': resample_data_smotetomek,
        'ADASYN': resample_data_adasyn,
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        try:
            X_res, y_res, stats = method_func(X_train, y_train, random_state=random_state, verbose=True)
            results[method_name] = {
                'X_resampled': X_res,
                'y_resampled': y_res,
                'stats': stats
            }
        except Exception as e:
            print(f"\n[WARNING] {method_name} failed: {str(e)}")
            continue
    
    # Print comparison summary
    print("\n" + "="*70)
    print("RESAMPLING METHODS COMPARISON")
    print("="*70)
    
    print(f"\n{'Method':<15} {'Original':<12} {'Resampled':<12} {'Change':<12} {'Runtime (s)':<12}")
    print("-"*70)
    
    for method_name, result in results.items():
        stats = result['stats']
        orig = stats['original_samples']
        resamp = stats['resampled_samples']
        change = resamp - orig
        runtime = stats['runtime']
        
        print(f"{method_name:<15} {orig:<12,} {resamp:<12,} {change:<+12,} {runtime:<12.3f}")
    
    print("\n" + "="*70)
    print("RECOMMENDATION: SMOTE-ENN")
    print("="*70)
    print("\nBased on research literature for churn prediction:")
    print("  • SMOTE-ENN provides best balance between oversampling and cleaning")
    print("  • Removes noisy samples while balancing classes")
    print("  • Generally yields better model performance on imbalanced datasets")
    print("="*70)
    
    return results


def apply_best_resampling(X_train, y_train, method='smoteenn', random_state=RANDOM_SEED, verbose=True):
    """
    Apply the best resampling method (wrapper function)
    
    Args:
        X_train: Training features
        y_train: Training target
        method: Resampling method ('smoteenn', 'smote', 'smotetomek', 'adasyn')
        random_state: Random seed for reproducibility
        verbose: Whether to print detailed information
    
    Returns:
        X_resampled, y_resampled, resampling_stats
    """
    method = method.lower()
    
    if method == 'smoteenn':
        return resample_data_smoteenn(X_train, y_train, random_state, verbose)
    elif method == 'smote':
        return resample_data_smote(X_train, y_train, random_state, verbose)
    elif method == 'smotetomek':
        return resample_data_smotetomek(X_train, y_train, random_state, verbose)
    elif method == 'adasyn':
        return resample_data_adasyn(X_train, y_train, random_state, verbose)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: smoteenn, smote, smotetomek, adasyn")


if __name__ == "__main__":
    """
    Demo script to test resampling methods
    """
    print("\n" + "="*70)
    print("RESAMPLING MODULE - DEMONSTRATION")
    print("="*70)
    
    # Load the training data
    from config import PROCESSED_DATA_DIR
    import os
    
    splits_dir = PROCESSED_DATA_DIR / "splits"
    
    if not os.path.exists(splits_dir / "X_train.csv"):
        print("\n[ERROR] Training data not found!")
        print("Please run data splitting first to generate train/test splits.")
        exit(1)
    
    print(f"\nLoading training data from: {splits_dir}")
    X_train = pd.read_csv(splits_dir / "X_train.csv")
    y_train = pd.read_csv(splits_dir / "y_train.csv").squeeze()
    
    print(f"[OK] Loaded X_train: {X_train.shape}")
    print(f"[OK] Loaded y_train: {y_train.shape}")
    
    # Apply SMOTE-ENN (recommended method)
    X_resampled, y_resampled, stats = resample_data_smoteenn(
        X_train, y_train, 
        random_state=RANDOM_SEED,
        verbose=True
    )
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETED")
    print("="*70)
    print(f"\n[OK] Resampled data shape: {X_resampled.shape}")
    print(f"[OK] Ready for model training!")
