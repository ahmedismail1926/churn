"""
Complete Bank Churn Prediction Pipeline
Runs all stages from data loading to model evaluation
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_SEED
import pandas as pd


def run_stage(stage_name, stage_function):
    """Run a pipeline stage with error handling"""
    print("\n" + "="*80)
    print(f"STAGE: {stage_name}")
    print("="*80)
    try:
        stage_function()
        print(f"\n✓ {stage_name} completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ {stage_name} failed with error:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def stage_1_eda_preprocessing():
    """Stage 1: EDA and Preprocessing"""
    from config import create_directories, print_folder_structure
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
    
    print("\n[1/7] Initializing configuration...")
    create_directories()
    print_folder_structure()
    
    print("\n[2/7] Loading dataset...")
    df = load_data()
    get_dataset_info(df)
    
    print("\n[3/7] Data quality analysis...")
    analyze_missing_values(df)
    categorical_cols, numerical_cols = analyze_categorical_columns(df)
    analyze_target_distribution(df)
    
    print("\n[4/7] Exploratory data analysis - churn patterns...")
    plot_all_churn_patterns(df)
    print_churn_summary(df)
    
    print("\n[5/7] Correlation analysis...")
    corr_matrix, high_corr_pairs = run_full_correlation_analysis(df, threshold=0.7)
    
    print("\n[6/7] Data preprocessing...")
    df_processed, preprocessing_stats = preprocess_data(
        df, 
        corr_matrix=corr_matrix,
        drop_corr_threshold=0.95,
        missing_value_strategy='drop'
    )
    
    print("\n[7/7] Saving processed data...")
    processed_file_path = PROCESSED_DATA_DIR / "processed_data.csv"
    df_processed.to_csv(processed_file_path, index=False)
    print(f"✓ Processed data saved to: {processed_file_path}")


def stage_2_feature_engineering():
    """Stage 2: Feature Engineering"""
    from feature_engineering import (
        label_encode_binary_columns,
        one_hot_encode_categorical_columns,
        scale_numerical_features
    )
    
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


def stage_3_data_splitting():
    """Stage 3: Train-Test Split"""
    from data_splitting import prepare_data_for_modeling
    
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


def stage_4_model_training():
    """Stage 4: Train Models"""
    print("\n[INFO] Starting model training...")
    print("This may take 10-15 minutes depending on your hardware.")
    
    import src.model_training as model_training
    model_training.main()


def stage_5_ensemble():
    """Stage 5: Stacking Ensemble"""
    print("\n[INFO] Creating stacking ensemble...")
    
    import src.ensemble_stacking as ensemble_stacking
    ensemble_stacking.main()


def stage_6_results():
    """Stage 6: Display Results"""
    print("\n[INFO] Displaying final results...")
    
    import scripts.show_results as show_results
    show_results.main()


def stage_7_check_status():
    """Stage 7: Final Status Check"""
    print("\n[INFO] Running final status check...")
    
    import scripts.check_status as check_status
    check_status.main()


def main():
    """Run complete pipeline"""
    print("\n" + "="*80)
    print("COMPLETE BANK CHURN PREDICTION PIPELINE")
    print("="*80)
    print("\nThis pipeline will execute all stages:")
    print("  1. EDA & Preprocessing")
    print("  2. Feature Engineering")
    print("  3. Data Splitting")
    print("  4. Model Training (7 models)")
    print("  5. Stacking Ensemble")
    print("  6. Results Display")
    print("  7. Status Check")
    print("\n" + "="*80)
    
    input("\nPress Enter to start the complete pipeline...")
    
    stages = [
        ("1. EDA & Preprocessing", stage_1_eda_preprocessing),
        ("2. Feature Engineering", stage_2_feature_engineering),
        ("3. Data Splitting", stage_3_data_splitting),
        ("4. Model Training", stage_4_model_training),
        ("5. Stacking Ensemble", stage_5_ensemble),
        ("6. Results Display", stage_6_results),
        ("7. Status Check", stage_7_check_status),
    ]
    
    completed_stages = []
    failed_stages = []
    
    for stage_name, stage_func in stages:
        success = run_stage(stage_name, stage_func)
        if success:
            completed_stages.append(stage_name)
        else:
            failed_stages.append(stage_name)
            print(f"\n[ERROR] Pipeline stopped at: {stage_name}")
            break
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)
    print(f"\n✓ Completed stages: {len(completed_stages)}/{len(stages)}")
    for stage in completed_stages:
        print(f"  ✓ {stage}")
    
    if failed_stages:
        print(f"\n✗ Failed stages: {len(failed_stages)}")
        for stage in failed_stages:
            print(f"  ✗ {stage}")
    else:
        print("\n🎉 ALL STAGES COMPLETED SUCCESSFULLY! 🎉")
        print("\nYour models are ready in: artifacts/models/")
        print("View results with: python scripts/show_results.py")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
