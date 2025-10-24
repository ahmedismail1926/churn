"""
Quick Demo of Stacking Ensemble
Uses fewer folds and models for faster execution
"""
from ensemble_stacking import StackingEnsemble, load_trained_models, compare_ensemble_vs_best_model
from model_training import load_training_data
from sklearn.linear_model import LogisticRegression
from config import RANDOM_SEED

def main():
    print("\n" + "="*80)
    print("QUICK STACKING ENSEMBLE DEMO")
    print("="*80)
    print("Faster version with 3 folds and top 3 models")
    print("="*80)
    
    # Load data
    X_train, X_test, y_train, y_test = load_training_data()
    
    # Load trained models
    all_models = load_trained_models()
    
    if len(all_models) == 0:
        print("\n✗ No trained models found!")
        print("Please run model_training.py first.")
        return
    
    # Use only top 3 models for speed
    top_models = {
        'Random Forest': all_models.get('Random Forest'),
        'XGBoost (Tuned)': all_models.get('XGBoost (Tuned)'),
        'LightGBM (Tuned)': all_models.get('LightGBM (Tuned)')
    }
    
    # Remove None values
    top_models = {k: v for k, v in top_models.items() if v is not None}
    
    print(f"\nUsing {len(top_models)} base models for demo:")
    for name in top_models.keys():
        print(f"  • {name}")
    
    # Create ensemble with fewer folds
    ensemble = StackingEnsemble(
        base_models=top_models,
        meta_classifier=LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
        n_folds=3,  # Reduced from 5
        use_proba=True,
        random_state=RANDOM_SEED
    )
    
    # Train ensemble
    ensemble.fit(X_train, y_train, apply_smote=True, verbose=True)
    
    # Evaluate
    ensemble_metrics = ensemble.evaluate(X_test, y_test, verbose=True)
    
    # Get base model predictions
    base_metrics = ensemble.get_base_model_predictions(X_test, y_test)
    
    # Compare
    compare_ensemble_vs_best_model(ensemble_metrics, base_metrics, 'XGBoost (Tuned)')
    
    print("\n" + "="*80)
    print("QUICK DEMO COMPLETE")
    print("="*80)
    print("\nFor full ensemble with all models and 5 folds:")
    print("  python ensemble_stacking.py")

if __name__ == "__main__":
    main()
