"""
Project Status Checker
Verifies all training outputs and provides summary
"""
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from config import MODELS_DIR, PROCESSED_DATA_DIR, ARTIFACTS_DIR
import pickle

def check_file_exists(filepath, description):
    """Check if file exists and print status"""
    if filepath.exists():
        size = filepath.stat().st_size / 1024  # KB
        print(f"  ✅ {description:<40} ({size:.1f} KB)")
        return True
    else:
        print(f"  ❌ {description:<40} (NOT FOUND)")
        return False

def main():
    print("\n" + "="*80)
    print("PROJECT STATUS CHECK")
    print("="*80)
    
    # Check data files
    print("\n📊 DATA FILES:")
    data_dir = PROCESSED_DATA_DIR / "splits"
    check_file_exists(data_dir / "X_train.csv", "Training features")
    check_file_exists(data_dir / "X_test.csv", "Test features")
    check_file_exists(data_dir / "y_train.csv", "Training labels")
    check_file_exists(data_dir / "y_test.csv", "Test labels")
    
    # Check models
    print("\n🤖 TRAINED MODELS:")
    models_dir = Path(MODELS_DIR)
    
    baseline_count = 0
    if check_file_exists(models_dir / "baseline_Logistic Regression.pkl", "Logistic Regression"):
        baseline_count += 1
    if check_file_exists(models_dir / "baseline_Naive Bayes.pkl", "Naive Bayes"):
        baseline_count += 1
    if check_file_exists(models_dir / "baseline_Random Forest.pkl", "Random Forest"):
        baseline_count += 1
    
    print("\n🚀 MAIN MODELS (with SMOTE-ENN):")
    main_count = 0
    if check_file_exists(models_dir / "main_XGBoost.pkl", "XGBoost"):
        main_count += 1
    if check_file_exists(models_dir / "main_LightGBM.pkl", "LightGBM"):
        main_count += 1
    
    print("\n🎯 TUNED MODELS (Optuna):")
    tuned_count = 0
    if check_file_exists(models_dir / "xgboost_tuned.pkl", "XGBoost (Tuned)"):
        tuned_count += 1
    if check_file_exists(models_dir / "lightgbm_tuned.pkl", "LightGBM (Tuned)"):
        tuned_count += 1
    
    print("\n🏆 ENSEMBLE:")
    ensemble_exists = check_file_exists(models_dir / "stacking_ensemble.pkl", "Stacking Ensemble")
    
    # Check results
    print("\n📈 RESULTS:")
    training_results = check_file_exists(models_dir / "training_results.pkl", "Training Results")
    ensemble_results = check_file_exists(models_dir / "ensemble_results.pkl", "Ensemble Results")
    
    # Check output logs
    print("\n📝 OUTPUT LOGS:")
    proj_dir = Path(__file__).parent.parent
    outputs_dir = proj_dir / "outputs"
    check_file_exists(outputs_dir / "model_training_output.txt", "Training Output Log")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_models = baseline_count + main_count + tuned_count
    print(f"\n✓ Baseline models: {baseline_count}/3")
    print(f"✓ Main models: {main_count}/2")
    print(f"✓ Tuned models: {tuned_count}/2")
    print(f"✓ Total trained models: {total_models}/7")
    
    if ensemble_exists:
        print(f"✓ Stacking ensemble: Ready ✅")
    else:
        print(f"✓ Stacking ensemble: Not created yet ⏳")
    
    # Load and display key metrics
    if training_results:
        print("\n" + "="*80)
        print("BEST MODEL PERFORMANCE")
        print("="*80)
        
        try:
            with open(models_dir / "training_results.pkl", 'rb') as f:
                results = pickle.load(f)
            
            # Find best baseline
            best_baseline = None
            best_baseline_f1 = 0
            for name, result in results.get('baseline_results', {}).items():
                f1 = result['test_metrics']['f1']
                if f1 > best_baseline_f1:
                    best_baseline_f1 = f1
                    best_baseline = (name, result['test_metrics'])
            
            if best_baseline:
                name, metrics = best_baseline
                print(f"\nBest Baseline: {name}")
                print(f"  Accuracy:  {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")
                print(f"  F1-Score:  {metrics['f1']:.4f}")
                print(f"  ROC-AUC:   {metrics.get('roc_auc', 0):.4f}")
            
            # Find best overall
            best_overall = None
            best_overall_f1 = 0
            for name, result in results.get('main_results', {}).items():
                f1 = result['test_metrics']['f1']
                if f1 > best_overall_f1:
                    best_overall_f1 = f1
                    best_overall = (name, result['test_metrics'])
            
            if best_overall:
                name, metrics = best_overall
                print(f"\nBest Overall: {name}")
                print(f"  Accuracy:  {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")
                print(f"  F1-Score:  {metrics['f1']:.4f}")
                print(f"  ROC-AUC:   {metrics.get('roc_auc', 0):.4f}")
        
        except Exception as e:
            print(f"\n⚠️  Could not load training results: {e}")
    
    # Next steps
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    
    if total_models < 7:
        print("\n⏳ Training incomplete. Run:")
        print("   D:\\GP\\churn\\.venv\\Scripts\\python.exe model_training.py")
    elif not ensemble_exists:
        print("\n🎯 Training complete! Create ensemble:")
        print("   D:\\GP\\churn\\.venv\\Scripts\\python.exe ensemble_stacking.py")
    else:
        print("\n✅ Everything complete! Generate visualizations:")
        print("   D:\\GP\\churn\\.venv\\Scripts\\python.exe visualize_results.py")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
