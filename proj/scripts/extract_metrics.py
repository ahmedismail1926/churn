import pickle
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from config import MODELS_DIR

# Load training results
results_file = Path(MODELS_DIR) / 'training_results.pkl'
if results_file.exists():
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    print("\n" + "="*80)
    print("ACTUAL TEST SET RESULTS")
    print("="*80)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
else:
    print("Results file not found!")
