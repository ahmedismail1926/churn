"""
Data Splitting Demonstration
Shows stratified train-test split and K-fold cross-validation
"""
from data_splitting import prepare_data_for_modeling
from config import RANDOM_SEED, TEST_SIZE


def main():
    """Run data splitting demonstration"""
    
    print("="*70)
    print("DATA SPLITTING & CROSS-VALIDATION DEMONSTRATION")
    print("="*70)
    
    # Prepare data with stratified split and K-fold CV
    data_dict = prepare_data_for_modeling(
        filepath=None,  # Uses default: data/processed/engineered_data.csv
        target_column='Attrition_Flag',
        test_size=TEST_SIZE,  # From config (0.2)
        n_folds=5,
        random_state=RANDOM_SEED,  # From config (42)
        save_splits=True
    )
    
    # Extract components
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    skf = data_dict['skf']
    
    # Additional verification
    print("\n" + "="*70)
    print("FINAL VERIFICATION")
    print("="*70)
    
    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test:  {y_test.shape}")
    
    print(f"\nData types:")
    print(f"  X_train types:")
    print(f"    {X_train.dtypes.value_counts().to_dict()}")
    
    print(f"\nTarget distribution:")
    print(f"  Training set:")
    print(f"    {y_train.value_counts().to_dict()}")
    print(f"  Test set:")
    print(f"    {y_test.value_counts().to_dict()}")
    
    print(f"\nK-Fold Cross-Validation:")
    print(f"  Configured with {skf.n_splits} folds")
    print(f"  Ready for model training and evaluation")
    
    print("\n" + "="*70)
    print("SUCCESS - Data is ready for model training!")
    print("="*70)
    
    print("\nNext steps:")
    print("  1. Train machine learning models")
    print("  2. Use K-fold CV for hyperparameter tuning")
    print("  3. Evaluate on test set for final performance")
    print("  4. Compare multiple algorithms")


if __name__ == "__main__":
    main()
