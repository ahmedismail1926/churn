"""
Test the model training module with a small sample
"""
import pandas as pd
import numpy as np
from model_training import ModelTrainer, load_training_data

def test_loading():
    """Test data loading"""
    print("Testing data loading...")
    X_train, X_test, y_train, y_test = load_training_data()
    print(f"✓ Data loaded successfully")
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def test_single_model():
    """Test training a single baseline model"""
    print("\nTesting single model training...")
    X_train, X_test, y_train, y_test = load_training_data()
    
    # Sample data for quick test (10% of data)
    sample_size = len(X_train) // 10
    X_train_sample = X_train.iloc[:sample_size]
    y_train_sample = y_train.iloc[:sample_size]
    
    trainer = ModelTrainer(X_train_sample, y_train_sample, X_test, y_test, n_folds=2)
    
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    result = trainer.train_with_cv(model, "Logistic Regression (Test)", X_train_sample, y_train_sample)
    
    print("\n✓ Single model test completed")
    print(f"  CV F1-Score: {np.mean(result['cv_results']['f1']):.4f}")
    print(f"  Test F1-Score: {result['test_metrics']['f1']:.4f}")

if __name__ == "__main__":
    print("="*80)
    print("MODEL TRAINING TEST SCRIPT")
    print("="*80)
    
    try:
        # Test loading
        test_loading()
        
        # Test single model
        test_single_model()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nYou can now run the full training:")
        print("  python model_training.py         (Full training)")
        print("  python demo_model_training.py    (Quick demo)")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
