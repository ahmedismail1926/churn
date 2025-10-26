"""Test Stage 2 only"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def stage_2_feature_engineering():
    """Stage 2: Feature Engineering"""
    from feature_engineering import (
        label_encode_binary_columns,
        one_hot_encode_categorical_columns,
        scale_numerical_features
    )
    from config import PROCESSED_DATA_DIR
    import pandas as pd
    
    print("\n[1/3] Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_DIR / "processed_data.csv")
    print(f"Loaded data shape: {df.shape}")
    
    print("\n[2/3] Encoding categorical features...")
    # Label encode binary columns
    df, label_encoders = label_encode_binary_columns(df)
    
    # One-hot encode multi-category columns
    df, new_columns = one_hot_encode_categorical_columns(df, drop_first=True)
    
    # Scale numerical features
    df, scaler = scale_numerical_features(df)
    
    print("\n[3/3] Saving engineered data...")
    output_path = PROCESSED_DATA_DIR / "engineered_data.csv"
    df.to_csv(output_path, index=False)
    print(f"✓ Engineered data saved to: {output_path}")
    print(f"  Final shape: {df.shape}")

if __name__ == "__main__":
    try:
        stage_2_feature_engineering()
        print("\n✓✓✓ STAGE 2 PASSED ✓✓✓")
    except Exception as e:
        print(f"\n✗✗✗ STAGE 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
