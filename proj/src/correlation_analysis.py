"""
Correlation Analysis Module
Handles correlation analysis and redundant feature detection
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import ARTIFACTS_DIR

# Visualization directory
VIZ_DIR = ARTIFACTS_DIR / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)


def analyze_correlations(df, threshold=0.7):
    """Analyze correlations and identify redundant features"""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS - REDUNDANT FEATURES")
    print("="*60)
    
    # Select only numerical columns (excluding Naive Bayes columns and CLIENTNUM)
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features = [col for col in numerical_features 
                         if not col.startswith('Naive_Bayes') and col != 'CLIENTNUM']
    
    print(f"\nAnalyzing {len(numerical_features)} numerical features for correlations...")
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_features].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix - Numerical Features', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VIZ_DIR / 'correlation_matrix.png'}")
    plt.close()
    
    return corr_matrix


def find_highly_correlated_pairs(corr_matrix, threshold=0.7):
    """Find highly correlated feature pairs"""
    print("\n" + "="*60)
    print("HIGHLY CORRELATED FEATURE PAIRS")
    print("="*60)
    
    # Get upper triangle of correlation matrix
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find pairs with high correlation
    high_corr_pairs = []
    for column in upper_triangle.columns:
        for index in upper_triangle.index:
            corr_value = upper_triangle.loc[index, column]
            if abs(corr_value) > threshold:
                high_corr_pairs.append({
                    'Feature_1': index,
                    'Feature_2': column,
                    'Correlation': corr_value
                })
    
    # Sort by absolute correlation value
    high_corr_df = pd.DataFrame(high_corr_pairs)
    if len(high_corr_df) > 0:
        high_corr_df['Abs_Correlation'] = high_corr_df['Correlation'].abs()
        high_corr_df = high_corr_df.sort_values('Abs_Correlation', ascending=False)
        
        print(f"\nFound {len(high_corr_df)} highly correlated pairs (|correlation| > {threshold}):\n")
        print(high_corr_df[['Feature_1', 'Feature_2', 'Correlation']].to_string(index=False))
        
        print("\n" + "-"*60)
        print("REDUNDANT FEATURE RECOMMENDATIONS")
        print("-"*60)
        
        for idx, row in high_corr_df.iterrows():
            print(f"\n⚠ {row['Feature_1']} ↔ {row['Feature_2']}")
            print(f"   Correlation: {row['Correlation']:.4f}")
            print(f"   Recommendation: Consider removing one of these features to reduce multicollinearity")
    else:
        print(f"\n✓ No highly correlated pairs found (|correlation| > {threshold})")
    
    return high_corr_df if len(high_corr_df) > 0 else None


def analyze_credit_limit_vs_open_to_buy(df):
    """Specific analysis for Credit_Limit vs Avg_Open_To_Buy"""
    print("\n" + "="*60)
    print("SPECIFIC ANALYSIS: Credit_Limit vs Avg_Open_To_Buy")
    print("="*60)
    
    if 'Credit_Limit' not in df.columns or 'Avg_Open_To_Buy' not in df.columns:
        print("⚠ Required columns not found.")
        return
    
    corr_credit_open = df[['Credit_Limit', 'Avg_Open_To_Buy']].corr().iloc[0, 1]
    print(f"\nCorrelation: {corr_credit_open:.4f}")
    
    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Credit_Limit'], df['Avg_Open_To_Buy'], alpha=0.3, s=20)
    plt.xlabel('Credit Limit', fontsize=12)
    plt.ylabel('Avg Open To Buy', fontsize=12)
    plt.title(f'Credit Limit vs Avg Open To Buy\nCorrelation: {corr_credit_open:.4f}', 
              fontsize=14, fontweight='bold')
    
    # Add regression line
    z = np.polyfit(df['Credit_Limit'], df['Avg_Open_To_Buy'], 1)
    p = np.poly1d(z)
    plt.plot(df['Credit_Limit'], p(df['Credit_Limit']), "r--", linewidth=2, label='Trend line')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'credit_limit_vs_avg_open_to_buy.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VIZ_DIR / 'credit_limit_vs_avg_open_to_buy.png'}")
    plt.close()
    
    print(f"\nStatistics:")
    print(f"  Credit_Limit - Mean: ${df['Credit_Limit'].mean():,.2f}, Std: ${df['Credit_Limit'].std():,.2f}")
    print(f"  Avg_Open_To_Buy - Mean: ${df['Avg_Open_To_Buy'].mean():,.2f}, Std: ${df['Avg_Open_To_Buy'].std():,.2f}")


def run_full_correlation_analysis(df, threshold=0.7):
    """Run complete correlation analysis"""
    corr_matrix = analyze_correlations(df, threshold)
    high_corr_pairs = find_highly_correlated_pairs(corr_matrix, threshold)
    analyze_credit_limit_vs_open_to_buy(df)
    
    return corr_matrix, high_corr_pairs
