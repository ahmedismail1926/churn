import sys
from pathlib import Path
sys.path.insert(0, 'src')

from config import PROCESSED_DATA_DIR
from feature_engineering import label_encode_binary_columns, one_hot_encode_categorical_columns, scale_numerical_features
import pandas as pd

try:
    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_DIR / "processed_data.csv")
    print(f"✓ Loaded: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}")
    
    print("\n" + "="*60)
    print("Testing label encoding...")
    df, encoders = label_encode_binary_columns(df)
    print(f"✓ After label encoding: {df.shape}")
    
    print("\n" + "="*60)
    print("Testing one-hot encoding...")
    df, new_cols = one_hot_encode_categorical_columns(df, drop_first=True)
    print(f"✓ After one-hot encoding: {df.shape}")
    
    print("\n" + "="*60)
    print("Testing scaling...")
    df, scaler = scale_numerical_features(df)
    print(f"✓ After scaling: {df.shape}")
    
    print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
    
except Exception as e:
    print(f"\n✗✗✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
