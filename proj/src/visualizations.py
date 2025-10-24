"""
Visualization Module
Handles all plotting and visualization tasks
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import ARTIFACTS_DIR

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create visualization directory
VIZ_DIR = ARTIFACTS_DIR / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)


def plot_churn_vs_gender(df, target_col='Attrition_Flag'):
    """Plot churn distribution by gender"""
    print("\n1. CHURN vs GENDER")
    print("-" * 40)
    
    churn_gender = pd.crosstab(df['Gender'], df[target_col], normalize='index') * 100
    print(churn_gender)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    sns.countplot(data=df, x='Gender', hue=target_col, ax=axes[0])
    axes[0].set_title('Churn Distribution by Gender (Count)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Gender')
    axes[0].set_ylabel('Count')
    
    # Percentage plot
    churn_gender_count = pd.crosstab(df['Gender'], df[target_col])
    churn_gender_count.plot(kind='bar', stacked=False, ax=axes[1])
    axes[1].set_title('Churn Distribution by Gender (Grouped)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Gender')
    axes[1].set_ylabel('Count')
    axes[1].legend(title='Attrition Flag')
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'churn_vs_gender.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VIZ_DIR / 'churn_vs_gender.png'}")
    plt.close()


def plot_churn_vs_age(df, target_col='Attrition_Flag'):
    """Plot churn distribution by age"""
    print("\n2. CHURN vs AGE")
    print("-" * 40)
    
    age_stats = df.groupby(target_col)['Customer_Age'].describe()
    print(age_stats)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Distribution plot
    for attrition in df[target_col].unique():
        sns.kdeplot(data=df[df[target_col] == attrition], 
                   x='Customer_Age', label=attrition, ax=axes[0], fill=True, alpha=0.5)
    axes[0].set_title('Age Distribution by Churn Status', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Customer Age')
    axes[0].legend()
    
    # Box plot
    sns.boxplot(data=df, x=target_col, y='Customer_Age', ax=axes[1])
    axes[1].set_title('Age Box Plot by Churn Status', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Attrition Flag')
    axes[1].set_ylabel('Customer Age')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=15)
    
    # Violin plot
    sns.violinplot(data=df, x=target_col, y='Customer_Age', ax=axes[2])
    axes[2].set_title('Age Violin Plot by Churn Status', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Attrition Flag')
    axes[2].set_ylabel('Customer Age')
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=15)
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'churn_vs_age.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VIZ_DIR / 'churn_vs_age.png'}")
    plt.close()


def plot_churn_vs_education(df, target_col='Attrition_Flag'):
    """Plot churn distribution by education level"""
    print("\n3. CHURN vs EDUCATION LEVEL")
    print("-" * 40)
    
    churn_education = pd.crosstab(df['Education_Level'], df[target_col], normalize='index') * 100
    print(churn_education)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Count plot
    sns.countplot(data=df, x='Education_Level', hue=target_col, ax=axes[0])
    axes[0].set_title('Churn Distribution by Education Level (Count)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Education Level')
    axes[0].set_ylabel('Count')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Percentage plot
    churn_education_pct = pd.crosstab(df['Education_Level'], df[target_col], normalize='index') * 100
    churn_education_pct.plot(kind='bar', stacked=True, ax=axes[1])
    axes[1].set_title('Churn Rate by Education Level (%)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Education Level')
    axes[1].set_ylabel('Percentage')
    axes[1].legend(title='Attrition Flag')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'churn_vs_education.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VIZ_DIR / 'churn_vs_education.png'}")
    plt.close()


def plot_churn_vs_income(df, target_col='Attrition_Flag'):
    """Plot churn distribution by income category"""
    print("\n4. CHURN vs INCOME CATEGORY")
    print("-" * 40)
    
    churn_income = pd.crosstab(df['Income_Category'], df[target_col], normalize='index') * 100
    print(churn_income)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Count plot
    income_order = ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +', 'Unknown']
    income_order_present = [cat for cat in income_order if cat in df['Income_Category'].unique()]
    
    sns.countplot(data=df, x='Income_Category', hue=target_col, order=income_order_present, ax=axes[0])
    axes[0].set_title('Churn Distribution by Income Category (Count)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Income Category')
    axes[0].set_ylabel('Count')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Percentage plot
    churn_income_pct = pd.crosstab(df['Income_Category'], df[target_col], normalize='index') * 100
    churn_income_pct = churn_income_pct.reindex(income_order_present)
    churn_income_pct.plot(kind='bar', stacked=True, ax=axes[1])
    axes[1].set_title('Churn Rate by Income Category (%)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Income Category')
    axes[1].set_ylabel('Percentage')
    axes[1].legend(title='Attrition Flag')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'churn_vs_income.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VIZ_DIR / 'churn_vs_income.png'}")
    plt.close()


def plot_all_churn_patterns(df, target_col='Attrition_Flag'):
    """Generate all churn pattern visualizations"""
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS - CHURN PATTERNS")
    print("="*60)
    
    if target_col not in df.columns:
        print(f"⚠ Target column '{target_col}' not found.")
        return
    
    plot_churn_vs_gender(df, target_col)
    plot_churn_vs_age(df, target_col)
    plot_churn_vs_education(df, target_col)
    plot_churn_vs_income(df, target_col)
    
    print(f"\n✓ All churn pattern visualizations saved to: {VIZ_DIR}")
