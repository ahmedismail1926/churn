"""
Feature Engineering Demo
Demonstrates encoding and scaling transformations
"""
import pandas as pd
from config import PROCESSED_DATA_DIR
from feature_engineering import (
    engineer_features,
    print_transformation_summary
)


def main():
    """Run feature engineering pipeline"""
    
    print("="*60)
    print("FEATURE ENGINEERING DEMONSTRATION")
    print("="*60)
    
    # Load processed data
    print("\nLoading processed data...")
    processed_file = PROCESSED_DATA_DIR / "processed_data.csv"
    df = pd.read_csv(processed_file)
    
    print(f"✓ Loaded: {processed_file}")
    print(f"  Shape: {df.shape}")
    
    # Store original for comparison
    df_original = df.copy()
    
    # Run feature engineering pipeline
    print("\nRunning feature engineering pipeline...")
    df_transformed, artifacts = engineer_features(
        df,
        scale_method='standardize',  # Options: 'standardize' or 'normalize'
        drop_first=True,  # Drop first dummy to avoid multicollinearity
        save_artifacts=True
    )
    
    # Print detailed summary
    print_transformation_summary(df_original, df_transformed, artifacts)
    
    # Save transformed data
    output_file = PROCESSED_DATA_DIR / "engineered_data.csv"
    df_transformed.to_csv(output_file, index=False)
    print(f"\n✓ Saved engineered data to: {output_file}")
    
    # Additional statistics
    print("\n" + "="*60)
    print("FINAL DATASET STATISTICS")
    print("="*60)
    print(f"\nDataset shape: {df_transformed.shape}")
    print(f"Total features: {df_transformed.shape[1]}")
    print(f"Total samples: {df_transformed.shape[0]}")
    print(f"\nData types:")
    print(df_transformed.dtypes.value_counts())
    
    print("\n✓ Feature engineering completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
