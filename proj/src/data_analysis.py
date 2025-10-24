"""
Data Analysis Module
Handles missing values detection, categorical analysis, and target distribution
"""
import pandas as pd
import numpy as np


def analyze_missing_values(df):
    """Detect and report missing values in the dataset"""
    print("\n" + "="*60)
    print("MISSING VALUES SUMMARY")
    print("="*60)
    
    missing_values = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    missing_summary = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': missing_values.values,
        'Missing_Percentage': missing_percentage.values
    })
    missing_summary = missing_summary[missing_summary['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    if len(missing_summary) > 0:
        print(missing_summary.to_string(index=False))
    else:
        print("✓ No missing values found in the dataset!")
    
    print(f"\nTotal missing values: {df.isnull().sum().sum()}")
    
    return missing_summary


def analyze_categorical_columns(df):
    """Identify and analyze categorical and numerical columns"""
    print("\n" + "="*60)
    print("CATEGORICAL COLUMNS")
    print("="*60)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\nCategorical columns ({len(categorical_cols)}):")
    for col in categorical_cols:
        print(f"  - {col}: {df[col].nunique()} unique values")
    
    print(f"\nNumerical columns ({len(numerical_cols)}):")
    for col in numerical_cols:
        print(f"  - {col}")
    
    return categorical_cols, numerical_cols


def analyze_target_distribution(df, target_col='Attrition_Flag'):
    """Analyze the distribution of the target variable"""
    print("\n" + "="*60)
    print("TARGET CLASS DISTRIBUTION")
    print("="*60)
    
    if target_col not in df.columns:
        print(f"⚠ Target column '{target_col}' not found in dataset.")
        return None
    
    print(f"Target column: '{target_col}'")
    print(f"\nValue counts:")
    print(df[target_col].value_counts())
    print(f"\nPercentage distribution:")
    print(df[target_col].value_counts(normalize=True) * 100)
    
    # Visualize distribution
    print(f"\nClass balance:")
    for value in df[target_col].unique():
        count = (df[target_col] == value).sum()
        percentage = (count / len(df)) * 100
        bar = '█' * int(percentage / 2)
        print(f"  {value}: {count:,} ({percentage:.2f}%) {bar}")
    
    return df[target_col].value_counts()


def print_churn_summary(df, target_col='Attrition_Flag'):
    """Print comprehensive churn distribution summary"""
    print("\n" + "="*60)
    print("CHURN DISTRIBUTION SUMMARY")
    print("="*60)
    
    print("\nKey Insights:")
    print(f"• Total Customers: {len(df):,}")
    print(f"• Attrited Customers: {(df[target_col] == 'Attrited Customer').sum():,} ({(df[target_col] == 'Attrited Customer').sum() / len(df) * 100:.2f}%)")
    print(f"• Existing Customers: {(df[target_col] == 'Existing Customer').sum():,} ({(df[target_col] == 'Existing Customer').sum() / len(df) * 100:.2f}%)")
    
    if 'Gender' in df.columns:
        print("\nChurn Rate by Gender:")
        gender_churn = df.groupby('Gender')[target_col].apply(lambda x: (x == 'Attrited Customer').sum() / len(x) * 100)
        for gender, rate in gender_churn.items():
            print(f"  {gender}: {rate:.2f}%")
    
    if 'Customer_Age' in df.columns:
        print("\nAverage Age by Churn Status:")
        age_by_churn = df.groupby(target_col)['Customer_Age'].mean()
        for status, age in age_by_churn.items():
            print(f"  {status}: {age:.2f} years")
