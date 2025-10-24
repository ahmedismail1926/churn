"""
Data Preprocessing Module
Handles data cleaning, feature selection, and missing value treatment
"""
import pandas as pd
import numpy as np


def drop_unnecessary_columns(df, columns_to_drop=None):
    """
    Drop unnecessary columns from the dataset
    
    Args:
        df: DataFrame to process
        columns_to_drop: List of column names to drop. If None, drops default columns.
    
    Returns:
        DataFrame with columns removed
    """
    print("\n" + "="*60)
    print("DROPPING UNNECESSARY COLUMNS")
    print("="*60)
    
    initial_cols = df.shape[1]
    initial_shape = df.shape
    
    # Default columns to drop
    if columns_to_drop is None:
        columns_to_drop = []
        
        # Always drop CLIENTNUM (identifier, no predictive value)
        if 'CLIENTNUM' in df.columns:
            columns_to_drop.append('CLIENTNUM')
        
        # Drop Naive Bayes classifier columns (target leakage)
        naive_bayes_cols = [col for col in df.columns if col.startswith('Naive_Bayes')]
        columns_to_drop.extend(naive_bayes_cols)
    
    # Check which columns exist before dropping
    existing_cols = [col for col in columns_to_drop if col in df.columns]
    missing_cols = [col for col in columns_to_drop if col not in df.columns]
    
    if missing_cols:
        print(f"\n⚠ Warning: These columns don't exist: {missing_cols}")
    
    if not existing_cols:
        print("\n✓ No columns to drop.")
        return df
    
    # Drop columns
    df_cleaned = df.drop(columns=existing_cols)
    
    print(f"\n✓ Dropped {len(existing_cols)} columns:")
    for col in existing_cols:
        print(f"  - {col}")
    
    print(f"\nDataset shape:")
    print(f"  Before: {initial_shape}")
    print(f"  After:  {df_cleaned.shape}")
    print(f"  Columns removed: {initial_cols - df_cleaned.shape[1]}")
    
    return df_cleaned


def drop_highly_correlated_features(df, corr_matrix, threshold=0.95):
    """
    Drop one feature from each highly correlated pair
    
    Args:
        df: DataFrame to process
        corr_matrix: Correlation matrix
        threshold: Correlation threshold (default 0.95)
    
    Returns:
        DataFrame with correlated features removed
    """
    print("\n" + "="*60)
    print(f"DROPPING HIGHLY CORRELATED FEATURES (threshold > {threshold})")
    print("="*60)
    
    initial_shape = df.shape
    columns_to_drop = set()
    
    # Get upper triangle of correlation matrix
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features with correlation greater than threshold
    for column in upper_triangle.columns:
        for index in upper_triangle.index:
            if abs(upper_triangle.loc[index, column]) > threshold:
                # Drop the second feature in the pair
                if column in df.columns:
                    columns_to_drop.add(column)
                    print(f"\n⚠ High correlation detected:")
                    print(f"  {index} ↔ {column}: {upper_triangle.loc[index, column]:.4f}")
                    print(f"  → Marking '{column}' for removal")
    
    if columns_to_drop:
        df_cleaned = df.drop(columns=list(columns_to_drop))
        print(f"\n✓ Dropped {len(columns_to_drop)} highly correlated features:")
        for col in columns_to_drop:
            print(f"  - {col}")
        
        print(f"\nDataset shape:")
        print(f"  Before: {initial_shape}")
        print(f"  After:  {df_cleaned.shape}")
        
        return df_cleaned
    else:
        print(f"\n✓ No features found with correlation > {threshold}")
        return df


def handle_missing_values(df, strategy='drop', numerical_strategy='median', 
                         categorical_strategy='most_frequent'):
    """
    Handle missing values in the dataset
    
    Args:
        df: DataFrame to process
        strategy: 'drop' to remove rows with missing values, 'impute' to fill them
        numerical_strategy: Strategy for numerical columns ('mean', 'median', 'mode')
        categorical_strategy: Strategy for categorical columns ('most_frequent', 'constant')
    
    Returns:
        DataFrame with missing values handled
    """
    print("\n" + "="*60)
    print("HANDLING MISSING VALUES")
    print("="*60)
    
    initial_rows = len(df)
    initial_shape = df.shape
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    
    print(f"\nInitial missing value count: {total_missing}")
    
    if total_missing == 0:
        print("\n✓ No missing values found! Dataset is clean.")
        return df, 0, 0
    
    # Display columns with missing values
    cols_with_missing = missing_counts[missing_counts > 0]
    print(f"\nColumns with missing values:")
    for col, count in cols_with_missing.items():
        pct = (count / len(df)) * 100
        print(f"  - {col}: {count} ({pct:.2f}%)")
    
    rows_dropped = 0
    values_imputed = 0
    
    if strategy == 'drop':
        # Drop rows with any missing values
        df_cleaned = df.dropna()
        rows_dropped = initial_rows - len(df_cleaned)
        
        print(f"\n✓ Strategy: DROP rows with missing values")
        print(f"  Rows dropped: {rows_dropped}")
        print(f"  Remaining rows: {len(df_cleaned)}")
        
    elif strategy == 'impute':
        df_cleaned = df.copy()
        
        # Separate numerical and categorical columns
        numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
        
        print(f"\n✓ Strategy: IMPUTE missing values")
        
        # Impute numerical columns
        for col in numerical_cols:
            if df_cleaned[col].isnull().sum() > 0:
                if numerical_strategy == 'mean':
                    fill_value = df_cleaned[col].mean()
                elif numerical_strategy == 'median':
                    fill_value = df_cleaned[col].median()
                elif numerical_strategy == 'mode':
                    fill_value = df_cleaned[col].mode()[0]
                else:
                    fill_value = df_cleaned[col].median()
                
                missing_count = df_cleaned[col].isnull().sum()
                df_cleaned[col].fillna(fill_value, inplace=True)
                values_imputed += missing_count
                print(f"  - {col}: Imputed {missing_count} values with {numerical_strategy} = {fill_value:.2f}")
        
        # Impute categorical columns
        for col in categorical_cols:
            if df_cleaned[col].isnull().sum() > 0:
                if categorical_strategy == 'most_frequent':
                    fill_value = df_cleaned[col].mode()[0]
                elif categorical_strategy == 'constant':
                    fill_value = 'Unknown'
                else:
                    fill_value = df_cleaned[col].mode()[0]
                
                missing_count = df_cleaned[col].isnull().sum()
                df_cleaned[col].fillna(fill_value, inplace=True)
                values_imputed += missing_count
                print(f"  - {col}: Imputed {missing_count} values with '{fill_value}'")
        
        print(f"\n  Total values imputed: {values_imputed}")
    
    else:
        print(f"\n⚠ Unknown strategy: {strategy}. Returning original DataFrame.")
        return df, 0, 0
    
    # Verify no missing values remain
    remaining_missing = df_cleaned.isnull().sum().sum()
    
    print(f"\n{'='*60}")
    print("MISSING VALUES SUMMARY")
    print("="*60)
    print(f"Initial shape: {initial_shape}")
    print(f"Final shape:   {df_cleaned.shape}")
    print(f"Rows dropped:  {rows_dropped}")
    print(f"Values imputed: {values_imputed}")
    print(f"Remaining null counts: {remaining_missing}")
    
    if remaining_missing > 0:
        print(f"\n⚠ Warning: {remaining_missing} missing values still remain!")
        remaining_cols = df_cleaned.isnull().sum()
        remaining_cols = remaining_cols[remaining_cols > 0]
        for col, count in remaining_cols.items():
            print(f"  - {col}: {count}")
    else:
        print(f"\n✓ All missing values handled successfully!")
    
    return df_cleaned, rows_dropped, values_imputed


def preprocess_data(df, corr_matrix=None, drop_corr_threshold=0.95, 
                   missing_value_strategy='drop'):
    """
    Complete preprocessing pipeline
    
    Args:
        df: DataFrame to preprocess
        corr_matrix: Correlation matrix (optional)
        drop_corr_threshold: Threshold for dropping correlated features
        missing_value_strategy: 'drop' or 'impute'
    
    Returns:
        Preprocessed DataFrame and statistics
    """
    print("\n" + "="*60)
    print("DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    initial_shape = df.shape
    print(f"\nInitial dataset shape: {initial_shape}")
    
    # Step 1: Drop unnecessary columns
    df_cleaned = drop_unnecessary_columns(df)
    
    # Step 2: Drop highly correlated features (if correlation matrix provided)
    if corr_matrix is not None:
        # Filter correlation matrix to only include columns still in df
        cols_in_df = [col for col in corr_matrix.columns if col in df_cleaned.columns]
        if len(cols_in_df) > 0:
            corr_matrix_filtered = corr_matrix.loc[cols_in_df, cols_in_df]
            df_cleaned = drop_highly_correlated_features(
                df_cleaned, corr_matrix_filtered, drop_corr_threshold
            )
    
    # Step 3: Handle missing values
    df_cleaned, rows_dropped, values_imputed = handle_missing_values(
        df_cleaned, strategy=missing_value_strategy
    )
    
    # Final summary
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Initial shape:  {initial_shape}")
    print(f"Final shape:    {df_cleaned.shape}")
    print(f"Rows removed:   {initial_shape[0] - df_cleaned.shape[0]} ({((initial_shape[0] - df_cleaned.shape[0]) / initial_shape[0] * 100):.2f}%)")
    print(f"Columns removed: {initial_shape[1] - df_cleaned.shape[1]}")
    print(f"Rows dropped (missing): {rows_dropped}")
    print(f"Values imputed: {values_imputed}")
    print(f"Remaining nulls: {df_cleaned.isnull().sum().sum()}")
    print("="*60)
    
    stats = {
        'initial_shape': initial_shape,
        'final_shape': df_cleaned.shape,
        'rows_removed': initial_shape[0] - df_cleaned.shape[0],
        'columns_removed': initial_shape[1] - df_cleaned.shape[1],
        'rows_dropped_missing': rows_dropped,
        'values_imputed': values_imputed,
        'remaining_nulls': df_cleaned.isnull().sum().sum()
    }
    
    return df_cleaned, stats
