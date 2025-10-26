"""
Bank Churn Prediction - Main Pipeline
Clean, modular architecture for churn analysis
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import create_directories, print_folder_structure, PROCESSED_DATA_DIR
from data_loader import load_data, get_dataset_info
from data_analysis import (
    analyze_missing_values, 
    analyze_categorical_columns, 
    analyze_target_distribution,
    print_churn_summary
)
from visualizations import plot_all_churn_patterns
from correlation_analysis import run_full_correlation_analysis
from preprocessing import preprocess_data


def main():
    """Main execution pipeline"""
    
    # ============================================================
    # STEP 1: Initialize Configuration
    # ============================================================
    print("Initializing project configuration...")
    create_directories()
    print_folder_structure()
    
    # ============================================================
    # STEP 2: Load Dataset
    # ============================================================
    df = load_data()
    
    # ============================================================
    # STEP 3: Basic Dataset Information
    # ============================================================
    get_dataset_info(df)
    
    # ============================================================
    # STEP 4: Data Quality Analysis
    # ============================================================
    analyze_missing_values(df)
    categorical_cols, numerical_cols = analyze_categorical_columns(df)
    analyze_target_distribution(df)
    
    # ============================================================
    # STEP 5: Exploratory Data Analysis - Churn Patterns
    # ============================================================
    plot_all_churn_patterns(df)
    print_churn_summary(df)
    
    # ============================================================
    # STEP 6: Correlation Analysis
    # ============================================================
    corr_matrix, high_corr_pairs = run_full_correlation_analysis(df, threshold=0.7)
    
    # ============================================================
    # STEP 7: Data Preprocessing
    # ============================================================
    # Preprocess data: drop unnecessary columns, handle correlated features
    df_processed, preprocessing_stats = preprocess_data(
        df, 
        corr_matrix=corr_matrix,
        drop_corr_threshold=0.95,  # Only drop features with very high correlation
        missing_value_strategy='drop'  # Change to 'impute' if needed
    )
    
    # Save processed data
    processed_file_path = PROCESSED_DATA_DIR / "processed_data.csv"
    df_processed.to_csv(processed_file_path, index=False)
    print(f"\n✓ Processed data saved to: {processed_file_path}")
    
    # ============================================================
    # COMPLETION
    # ============================================================
    print("\n" + "="*60)
    print("✓ EXPLORATORY DATA ANALYSIS & PREPROCESSING COMPLETED")
    print("="*60)
    print("\nAll visualizations saved to: artifacts/visualizations/")
    print("\nGenerated files:")
    print("  1. churn_vs_gender.png")
    print("  2. churn_vs_age.png")
    print("  3. churn_vs_education.png")
    print("  4. churn_vs_income.png")
    print("  5. correlation_matrix.png")
    print("  6. credit_limit_vs_avg_open_to_buy.png")
    print(f"\nProcessed data: {processed_file_path}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
