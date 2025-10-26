"""Test Stage 3 - Data Splitting"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def stage_3_data_splitting():
    """Stage 3: Train-Test Split"""
    from data_splitting import prepare_data_for_modeling
    from config import PROCESSED_DATA_DIR, RANDOM_SEED
    
    print("\n[1/1] Splitting data into train/test sets...")
    splits = prepare_data_for_modeling(
        filepath=PROCESSED_DATA_DIR / "engineered_data.csv",
        target_column='Attrition_Flag',
        test_size=0.2,
        n_folds=5,
        random_state=RANDOM_SEED,
        save_splits=True
    )
    
    print(f"\n✓ Data split complete:")
    print(f"  Training samples: {len(splits['X_train']):,}")
    print(f"  Test samples: {len(splits['X_test']):,}")
    print(f"  Features: {len(splits['feature_names'])}")

if __name__ == "__main__":
    try:
        stage_3_data_splitting()
        print("\n✓✓✓ STAGE 3 PASSED ✓✓✓")
    except Exception as e:
        print(f"\n✗✗✗ STAGE 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
