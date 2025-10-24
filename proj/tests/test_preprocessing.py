"""
Test Preprocessing Module
Demonstrates data preprocessing functionality
"""
from config import create_directories, print_folder_structure, RAW_DATA_FILE, PROCESSED_DATA_DIR
from data_loader import load_data
from correlation_analysis import analyze_correlations
from preprocessing import preprocess_data
import pandas as pd


def main():
    """Test preprocessing pipeline"""
    
    print("="*60)
    print("TESTING DATA PREPROCESSING MODULE")
    print("="*60)
    
    # Load data
    print("\nStep 1: Loading data...")
    df = load_data()
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {list(df.columns)[:5]}... (showing first 5)")
    
    # Analyze correlations
    print("\nStep 2: Analyzing correlations...")
    # Get numerical features for correlation (excluding identifiers and naive bayes)
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_features = [col for col in numerical_features 
                         if not col.startswith('Naive_Bayes') and col != 'CLIENTNUM']
    corr_matrix = df[numerical_features].corr()
    print(f"Correlation matrix calculated for {len(numerical_features)} features")
    
    # Preprocess data
    print("\nStep 3: Running preprocessing pipeline...")
    df_processed, stats = preprocess_data(
        df, 
        corr_matrix=corr_matrix,
        drop_corr_threshold=0.95,
        missing_value_strategy='drop'  # Since we have no missing values
    )
    
    # Display results
    print("\n" + "="*60)
    print("PREPROCESSING RESULTS")
    print("="*60)
    print(f"\nOriginal dataset:")
    print(f"  Shape: {stats['initial_shape']}")
    
    print(f"\nProcessed dataset:")
    print(f"  Shape: {stats['final_shape']}")
    
    print(f"\nChanges:")
    print(f"  Rows removed: {stats['rows_removed']} ({stats['rows_removed'] / stats['initial_shape'][0] * 100:.2f}%)")
    print(f"  Columns removed: {stats['columns_removed']}")
    print(f"  Rows dropped (missing): {stats['rows_dropped_missing']}")
    print(f"  Values imputed: {stats['values_imputed']}")
    print(f"  Remaining nulls: {stats['remaining_nulls']}")
    
    print(f"\nRemaining columns ({df_processed.shape[1]}):")
    for i, col in enumerate(df_processed.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Save processed data
    processed_file = PROCESSED_DATA_DIR / "processed_data.csv"
    df_processed.to_csv(processed_file, index=False)
    print(f"\n✓ Processed data saved to: {processed_file}")
    
    # Display sample of processed data
    print("\n" + "="*60)
    print("SAMPLE OF PROCESSED DATA (First 5 rows)")
    print("="*60)
    print(df_processed.head())
    
    print("\n" + "="*60)
    print("✓ PREPROCESSING TEST COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
