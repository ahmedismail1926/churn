"""
Feature Engineering Module
Handles encoding categorical variables and scaling numerical features
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from config import PROCESSED_DATA_DIR, ARTIFACTS_DIR
import pickle


def label_encode_binary_columns(df, binary_columns=None):
    """
    Label encode binary categorical columns
    
    Args:
        df: DataFrame to process
        binary_columns: List of binary columns to encode. If None, auto-detects.
    
    Returns:
        DataFrame with encoded columns, dictionary of encoders
    """
    print("\n" + "="*60)
    print("LABEL ENCODING BINARY COLUMNS")
    print("="*60)
    
    df_encoded = df.copy()
    encoders = {}
    
    if binary_columns is None:
        # Auto-detect binary columns
        binary_columns = []
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() == 2:
                binary_columns.append(col)
    
    if not binary_columns:
        print("\n⚠ No binary columns found to encode.")
        return df_encoded, encoders
    
    print(f"\nFound {len(binary_columns)} binary columns to encode:")
    
    for col in binary_columns:
        if col in df.columns:
            # Create label encoder
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col])
            encoders[col] = le
            
            # Print mapping
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            print(f"\n  {col}:")
            for original, encoded in mapping.items():
                print(f"    {original} → {encoded}")
            
            print(f"    ✓ Encoded: {df[col].dtype} → {df_encoded[col].dtype}")
    
    print(f"\n✓ Successfully encoded {len(binary_columns)} binary columns")
    
    return df_encoded, encoders


def one_hot_encode_categorical_columns(df, categorical_columns=None, drop_first=True):
    """
    One-hot encode multi-category columns
    
    Args:
        df: DataFrame to process
        categorical_columns: List of columns to one-hot encode. If None, auto-detects.
        drop_first: Whether to drop first dummy variable to avoid multicollinearity
    
    Returns:
        DataFrame with one-hot encoded columns, list of new column names
    """
    print("\n" + "="*60)
    print("ONE-HOT ENCODING CATEGORICAL COLUMNS")
    print("="*60)
    
    df_encoded = df.copy()
    new_columns = []
    
    if categorical_columns is None:
        # Auto-detect multi-category columns (>2 unique values)
        categorical_columns = []
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() > 2:
                categorical_columns.append(col)
    
    if not categorical_columns:
        print("\n⚠ No multi-category columns found to encode.")
        return df_encoded, new_columns
    
    print(f"\nFound {len(categorical_columns)} multi-category columns to encode:")
    
    for col in categorical_columns:
        if col in df_encoded.columns:
            unique_values = df_encoded[col].unique()
            n_unique = len(unique_values)
            
            print(f"\n  {col}: {n_unique} unique values")
            print(f"    Values: {list(unique_values)}")
            
            # One-hot encode
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=drop_first)
            new_cols = dummies.columns.tolist()
            new_columns.extend(new_cols)
            
            # Add dummies to dataframe and drop original
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(columns=[col])
            
            n_created = len(new_cols)
            print(f"    ✓ Created {n_created} binary columns:")
            for new_col in new_cols:
                print(f"      - {new_col}")
    
    print(f"\n✓ Successfully one-hot encoded {len(categorical_columns)} columns")
    print(f"✓ Total new binary features created: {len(new_columns)}")
    
    return df_encoded, new_columns


def scale_numerical_features(df, numerical_columns=None, method='standardize'):
    """
    Scale or normalize numerical features
    
    Args:
        df: DataFrame to process
        numerical_columns: List of numerical columns to scale. If None, scales all numeric.
        method: 'standardize' (z-score) or 'normalize' (min-max scaling)
    
    Returns:
        DataFrame with scaled columns, scaler object
    """
    print("\n" + "="*60)
    print(f"SCALING NUMERICAL FEATURES - Method: {method.upper()}")
    print("="*60)
    
    df_scaled = df.copy()
    
    if numerical_columns is None:
        # Get all numerical columns
        numerical_columns = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numerical_columns:
        print("\n⚠ No numerical columns found to scale.")
        return df_scaled, None
    
    print(f"\nScaling {len(numerical_columns)} numerical features:")
    
    # Create scaler
    if method == 'standardize':
        scaler = StandardScaler()
        print("\n  Method: StandardScaler (Z-score normalization)")
        print("  Formula: (X - mean) / std")
    elif method == 'normalize':
        scaler = MinMaxScaler()
        print("\n  Method: MinMaxScaler (Range [0, 1])")
        print("  Formula: (X - min) / (max - min)")
    else:
        print(f"\n⚠ Unknown method: {method}. Using StandardScaler.")
        scaler = StandardScaler()
    
    # Print original statistics
    print("\n  Original Statistics (first 5 features):")
    print("  " + "-"*56)
    for col in numerical_columns[:5]:
        mean_val = df_scaled[col].mean()
        std_val = df_scaled[col].std()
        min_val = df_scaled[col].min()
        max_val = df_scaled[col].max()
        print(f"  {col:30s} | μ={mean_val:8.2f} σ={std_val:7.2f}")
    
    if len(numerical_columns) > 5:
        print(f"  ... and {len(numerical_columns) - 5} more features")
    
    # Fit and transform
    df_scaled[numerical_columns] = scaler.fit_transform(df_scaled[numerical_columns])
    
    # Print scaled statistics
    print("\n  Scaled Statistics (first 5 features):")
    print("  " + "-"*56)
    for col in numerical_columns[:5]:
        mean_val = df_scaled[col].mean()
        std_val = df_scaled[col].std()
        min_val = df_scaled[col].min()
        max_val = df_scaled[col].max()
        print(f"  {col:30s} | μ={mean_val:8.2f} σ={std_val:7.2f}")
    
    if len(numerical_columns) > 5:
        print(f"  ... and {len(numerical_columns) - 5} more features")
    
    print(f"\n✓ Successfully scaled {len(numerical_columns)} numerical features")
    
    return df_scaled, scaler


def engineer_features(df, scale_method='standardize', drop_first=True, save_artifacts=True):
    """
    Complete feature engineering pipeline
    
    Args:
        df: DataFrame to process
        scale_method: 'standardize' or 'normalize'
        drop_first: Drop first dummy in one-hot encoding
        save_artifacts: Save encoders and scalers
    
    Returns:
        Transformed DataFrame, dictionary of artifacts
    """
    print("\n" + "="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    initial_shape = df.shape
    print(f"\nInitial dataset shape: {initial_shape}")
    print(f"Initial columns: {list(df.columns)[:5]}... (showing first 5)")
    
    # Store original column order for reference
    original_columns = df.columns.tolist()
    
    # Step 1: Identify column types
    print("\n" + "-"*60)
    print("STEP 1: Identifying Column Types")
    print("-"*60)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Separate binary and multi-category
    binary_cols = [col for col in categorical_cols if df[col].nunique() == 2]
    multi_category_cols = [col for col in categorical_cols if df[col].nunique() > 2]
    
    print(f"\nBinary categorical columns ({len(binary_cols)}): {binary_cols}")
    print(f"Multi-category columns ({len(multi_category_cols)}): {multi_category_cols}")
    print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols[:5]}..." if len(numerical_cols) > 5 else f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    
    artifacts = {
        'original_columns': original_columns,
        'binary_columns': binary_cols,
        'multi_category_columns': multi_category_cols,
        'numerical_columns': numerical_cols
    }
    
    # Step 2: Label encode binary columns
    df_transformed, label_encoders = label_encode_binary_columns(df, binary_cols)
    artifacts['label_encoders'] = label_encoders
    
    # Step 3: One-hot encode multi-category columns
    df_transformed, new_columns = one_hot_encode_categorical_columns(
        df_transformed, multi_category_cols, drop_first=drop_first
    )
    artifacts['one_hot_columns'] = new_columns
    
    # Step 4: Scale numerical features
    df_transformed, scaler = scale_numerical_features(
        df_transformed, numerical_cols, method=scale_method
    )
    artifacts['scaler'] = scaler
    artifacts['scaling_method'] = scale_method
    
    # Final summary
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"\nInitial shape:  {initial_shape}")
    print(f"Final shape:    {df_transformed.shape}")
    print(f"Features added: {df_transformed.shape[1] - initial_shape[1]}")
    
    print(f"\nTransformation Summary:")
    print(f"  • Binary columns label-encoded: {len(binary_cols)}")
    print(f"  • Multi-category columns one-hot encoded: {len(multi_category_cols)}")
    print(f"  • New binary features created: {len(new_columns)}")
    print(f"  • Numerical features scaled: {len(numerical_cols)}")
    print(f"  • Scaling method: {scale_method}")
    
    # Save artifacts
    if save_artifacts:
        artifacts_path = ARTIFACTS_DIR / "feature_engineering"
        artifacts_path.mkdir(exist_ok=True)
        
        with open(artifacts_path / "encoders_scalers.pkl", 'wb') as f:
            pickle.dump(artifacts, f)
        print(f"\n✓ Saved encoders and scalers to: {artifacts_path / 'encoders_scalers.pkl'}")
    
    artifacts['final_shape'] = df_transformed.shape
    artifacts['final_columns'] = df_transformed.columns.tolist()
    
    return df_transformed, artifacts


def print_transformation_summary(df_original, df_transformed, artifacts):
    """
    Print detailed transformation summary with samples
    """
    print("\n" + "="*60)
    print("DETAILED TRANSFORMATION SUMMARY")
    print("="*60)
    
    print("\n" + "-"*60)
    print("1. TRANSFORMED FEATURE NAMES")
    print("-"*60)
    
    all_features = df_transformed.columns.tolist()
    print(f"\nTotal features: {len(all_features)}")
    
    # Categorize features
    binary_encoded = artifacts['binary_columns']
    one_hot_features = artifacts['one_hot_columns']
    numerical_features = artifacts['numerical_columns']
    
    print(f"\n  Label-encoded binary features ({len(binary_encoded)}):")
    for col in binary_encoded:
        if col in df_transformed.columns:
            print(f"    • {col}")
    
    print(f"\n  One-hot encoded features ({len(one_hot_features)}):")
    for col in one_hot_features:
        print(f"    • {col}")
    
    print(f"\n  Numerical features (scaled) ({len(numerical_features)}):")
    for col in numerical_features:
        print(f"    • {col}")
    
    # Sample of encoded values
    print("\n" + "-"*60)
    print("2. SAMPLE OF LABEL-ENCODED VALUES")
    print("-"*60)
    
    for col in binary_encoded:
        if col in df_transformed.columns:
            print(f"\n  {col}:")
            value_counts = df_transformed[col].value_counts().sort_index()
            for val, count in value_counts.items():
                original_val = artifacts['label_encoders'][col].inverse_transform([val])[0]
                print(f"    {val} (was '{original_val}'): {count} samples ({count/len(df_transformed)*100:.1f}%)")
    
    # Sample of scaled values
    print("\n" + "-"*60)
    print("3. SAMPLE OF SCALED VALUES (First 5 rows, First 5 numerical features)")
    print("-"*60)
    
    num_features_sample = numerical_features[:5]
    sample_df = df_transformed[num_features_sample].head()
    
    print("\nScaled values:")
    print(sample_df.to_string())
    
    print(f"\nScaling method: {artifacts['scaling_method'].upper()}")
    if artifacts['scaling_method'] == 'standardize':
        print("  • Mean ≈ 0, Std ≈ 1")
    else:
        print("  • Range: [0, 1]")
    
    # Compare original vs scaled
    print("\n" + "-"*60)
    print("4. BEFORE vs AFTER SCALING (First 3 rows, First 3 features)")
    print("-"*60)
    
    comparison_features = numerical_features[:3]
    print("\nBefore (Original):")
    print(df_original[comparison_features].head(3).to_string())
    
    print("\nAfter (Scaled):")
    print(df_transformed[comparison_features].head(3).to_string())
    
    print("\n" + "="*60)
