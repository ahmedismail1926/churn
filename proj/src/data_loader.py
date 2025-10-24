"""
Data Loading Module
Handles loading and basic validation of the dataset
"""
import pandas as pd
from config import RAW_DATA_FILE


def load_data():
    """Load the bank churners dataset"""
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    
    df = pd.read_csv(RAW_DATA_FILE)
    print(f"✓ Data loaded successfully from {RAW_DATA_FILE}")
    print(f"  Rows: {df.shape[0]:,}")
    print(f"  Columns: {df.shape[1]}")
    
    return df


def get_dataset_info(df):
    """Display basic information about the dataset"""
    print("\n" + "="*60)
    print("DATASET SHAPE")
    print("="*60)
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    print(f"Shape: {df.shape}")
    
    print("\n" + "="*60)
    print("FIRST 5 ROWS OF DATASET")
    print("="*60)
    print(df.head(5))
    
    print("\n" + "="*60)
    print("COLUMN NAMES AND DATA TYPES")
    print("="*60)
    print(df.dtypes)
    print(f"\nTotal columns: {len(df.columns)}")
