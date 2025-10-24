"""
Quick Results Display
Shows all model performances in a clean format
"""
import pickle
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from config import MODELS_DIR

def main():
    print("\n" + "="*80)
    print("COMPLETE TRAINING RESULTS SUMMARY")
    print("="*80)
    
    # Load training results
    results_path = Path(MODELS_DIR) / 'training_results.pkl'
    
    if not results_path.exists():
        print("❌ No training results found!")
        return
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    # Display baseline models
    print("\n📊 BASELINE MODELS (Original Imbalanced Data)")
    print("-" * 80)
    print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
    print("-" * 80)
    
    for name, result in results.get('baseline_results', {}).items():
        m = result['test_metrics']
        print(f"{name:<25} {m['accuracy']:<10.4f} {m['precision']:<10.4f} "
              f"{m['recall']:<10.4f} {m['f1']:<10.4f} {m.get('roc_auc', 0):<10.4f}")
    
    # Display main models
    print("\n🚀 MAIN & TUNED MODELS (with SMOTE-ENN)")
    print("-" * 80)
    print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
    print("-" * 80)
    
    for name, result in results.get('main_results', {}).items():
        m = result['test_metrics']
        display_name = name.replace('_tuned', ' (Tuned)')
        print(f"{display_name:<25} {m['accuracy']:<10.4f} {m['precision']:<10.4f} "
              f"{m['recall']:<10.4f} {m['f1']:<10.4f} {m.get('roc_auc', 0):<10.4f}")
    
    # Load ensemble results
    ensemble_path = Path(MODELS_DIR) / 'ensemble_results.pkl'
    if ensemble_path.exists():
        with open(ensemble_path, 'rb') as f:
            ensemble_results = pickle.load(f)
        
        print("\n🏆 STACKING ENSEMBLE")
        print("-" * 80)
        print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
        print("-" * 80)
        
        m = ensemble_results['ensemble_metrics']
        print(f"{'Stacking Ensemble':<25} {m['accuracy']:<10.4f} {m['precision']:<10.4f} "
              f"{m['recall']:<10.4f} {m['f1']:<10.4f} {m.get('roc_auc', 0):<10.4f}")
    
    # Find best performers
    print("\n" + "="*80)
    print("🏅 TOP PERFORMERS")
    print("="*80)
    
    all_models = {}
    for name, result in results.get('baseline_results', {}).items():
        all_models[f"[B] {name}"] = result['test_metrics']
    for name, result in results.get('main_results', {}).items():
        display_name = name.replace('_tuned', ' (Tuned)')
        all_models[f"[M] {display_name}"] = result['test_metrics']
    
    if ensemble_path.exists():
        all_models["[E] Stacking Ensemble"] = ensemble_results['ensemble_metrics']
    
    # Best F1
    best_f1 = max(all_models.items(), key=lambda x: x[1]['f1'])
    print(f"\n🥇 Best F1-Score: {best_f1[0]}")
    print(f"   F1 = {best_f1[1]['f1']:.4f}")
    
    # Best Recall
    best_recall = max(all_models.items(), key=lambda x: x[1]['recall'])
    print(f"\n🥇 Best Recall: {best_recall[0]}")
    print(f"   Recall = {best_recall[1]['recall']:.4f}")
    
    # Best Precision
    best_precision = max(all_models.items(), key=lambda x: x[1]['precision'])
    print(f"\n🥇 Best Precision: {best_precision[0]}")
    print(f"   Precision = {best_precision[1]['precision']:.4f}")
    
    print("\n" + "="*80)
    print("✅ ALL RESULTS DISPLAYED")
    print("="*80)

if __name__ == "__main__":
    main()
