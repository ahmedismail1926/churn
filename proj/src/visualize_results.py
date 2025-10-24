"""
Model Comparison and Visualization
Creates comprehensive visualizations comparing all models
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from config import MODELS_DIR, ARTIFACTS_DIR

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_all_results():
    """Load results from training and ensemble"""
    models_dir = Path(MODELS_DIR)
    
    results = {}
    
    # Load training results
    training_results_path = models_dir / 'training_results.pkl'
    if training_results_path.exists():
        with open(training_results_path, 'rb') as f:
            training_results = pickle.load(f)
        results['training'] = training_results
    
    # Load ensemble results
    ensemble_results_path = models_dir / 'ensemble_results.pkl'
    if ensemble_results_path.exists():
        with open(ensemble_results_path, 'rb') as f:
            ensemble_results = pickle.load(f)
        results['ensemble'] = ensemble_results
    
    return results


def create_comparison_dataframe(results):
    """Create a dataframe with all model metrics"""
    data = []
    
    # Baseline models
    if 'training' in results:
        for model_name, result in results['training'].get('baseline_results', {}).items():
            metrics = result['test_metrics']
            data.append({
                'Model': f"[B] {model_name}",
                'Category': 'Baseline',
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'ROC-AUC': metrics.get('roc_auc', 0)
            })
        
        # Main models
        for model_name, result in results['training'].get('main_results', {}).items():
            metrics = result['test_metrics']
            category = 'Tuned' if 'tuned' in model_name.lower() else 'Main'
            display_name = model_name.replace('_tuned', ' (Tuned)')
            data.append({
                'Model': f"[{category[0]}] {display_name}",
                'Category': category,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'ROC-AUC': metrics.get('roc_auc', 0)
            })
    
    # Ensemble
    if 'ensemble' in results:
        metrics = results['ensemble']['ensemble_metrics']
        data.append({
            'Model': '[E] Stacking Ensemble',
            'Category': 'Ensemble',
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'ROC-AUC': metrics.get('roc_auc', 0)
        })
    
    return pd.DataFrame(data)


def plot_metrics_comparison(df, save_path=None):
    """Create bar plots comparing all metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Performance Comparison - All Metrics', fontsize=16, fontweight='bold')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    colors = {'Baseline': '#3498db', 'Main': '#2ecc71', 'Tuned': '#e74c3c', 'Ensemble': '#f39c12'}
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        # Sort by metric value
        df_sorted = df.sort_values(metric, ascending=True)
        
        # Create horizontal bar plot
        bars = ax.barh(df_sorted['Model'], df_sorted[metric], 
                       color=[colors[cat] for cat in df_sorted['Category']])
        
        # Customize
        ax.set_xlabel(metric, fontweight='bold')
        ax.set_xlim(0.8, 1.0)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (model, value) in enumerate(zip(df_sorted['Model'], df_sorted[metric])):
            ax.text(value + 0.002, i, f'{value:.4f}', 
                   va='center', fontsize=8)
        
        # Highlight best
        best_idx = df_sorted[metric].idxmax()
        ax.get_children()[best_idx].set_edgecolor('black')
        ax.get_children()[best_idx].set_linewidth(2)
    
    # Remove extra subplot
    axes[1, 2].axis('off')
    
    # Add legend in the empty subplot
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=cat) 
                      for cat, color in colors.items()]
    axes[1, 2].legend(handles=legend_elements, loc='center', 
                     title='Model Category', fontsize=12, title_fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()


def plot_f1_recall_scatter(df, save_path=None):
    """Create scatter plot of F1 vs Recall (key metrics for churn)"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {'Baseline': '#3498db', 'Main': '#2ecc71', 'Tuned': '#e74c3c', 'Ensemble': '#f39c12'}
    markers = {'Baseline': 'o', 'Main': 's', 'Tuned': '^', 'Ensemble': '*'}
    
    for category in df['Category'].unique():
        df_cat = df[df['Category'] == category]
        ax.scatter(df_cat['Recall'], df_cat['F1-Score'],
                  c=colors[category], marker=markers[category],
                  s=200 if category == 'Ensemble' else 150,
                  label=category, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Add model labels
    for _, row in df.iterrows():
        ax.annotate(row['Model'].split('] ')[1], 
                   (row['Recall'], row['F1-Score']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.8)
    
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('F1-Score vs Recall: Key Metrics for Churn Prediction', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(title='Model Category', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add reference lines for "good" performance
    ax.axhline(y=0.97, color='gray', linestyle='--', alpha=0.5, label='F1 = 0.97')
    ax.axvline(x=0.97, color='gray', linestyle='--', alpha=0.5, label='Recall = 0.97')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()


def plot_metric_heatmap(df, save_path=None):
    """Create heatmap of all metrics"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare data
    metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    heatmap_data = df.set_index('Model')[metrics_cols]
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlGn',
               vmin=0.85, vmax=1.0, center=0.95,
               linewidths=1, linecolor='white',
               cbar_kws={'label': 'Score'}, ax=ax)
    
    ax.set_title('Model Performance Heatmap', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()


def plot_improvement_bars(df, save_path=None):
    """Plot improvement of ensemble vs best baseline"""
    # Find best baseline model
    baseline_df = df[df['Category'] == 'Baseline']
    if len(baseline_df) == 0:
        print("No baseline models found")
        return
    
    best_baseline = baseline_df.loc[baseline_df['F1-Score'].idxmax()]
    
    # Find ensemble
    ensemble_df = df[df['Category'] == 'Ensemble']
    if len(ensemble_df) == 0:
        print("No ensemble found")
        return
    
    ensemble = ensemble_df.iloc[0]
    
    # Calculate improvements
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    improvements = []
    for metric in metrics:
        baseline_val = best_baseline[metric]
        ensemble_val = ensemble[metric]
        improvement = ((ensemble_val - baseline_val) / baseline_val) * 100
        improvements.append(improvement)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['green' if x > 0 else 'red' for x in improvements]
    bars = ax.bar(metrics, improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:+.2f}%', ha='center', va='bottom' if height > 0 else 'top',
               fontsize=11, fontweight='bold')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Ensemble Improvement over Best Baseline ({best_baseline["Model"]})',
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()


def create_summary_table(df):
    """Create and print summary table"""
    print("\n" + "="*100)
    print("MODEL PERFORMANCE SUMMARY TABLE")
    print("="*100)
    
    # Sort by F1-Score
    df_sorted = df.sort_values('F1-Score', ascending=False)
    
    print(f"\n{'Rank':<6} {'Model':<30} {'Category':<12} {'Accuracy':<10} {'Precision':<10} "
          f"{'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
    print("-"*100)
    
    for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"{rank:<6} {row['Model']:<30} {row['Category']:<12} "
              f"{row['Accuracy']:<10.4f} {row['Precision']:<10.4f} "
              f"{row['Recall']:<10.4f} {row['F1-Score']:<10.4f} {row['ROC-AUC']:<10.4f}")
    
    print("\n" + "="*100)
    
    # Best performers
    print("\nBEST PERFORMERS:")
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
        best_model = df.loc[df[metric].idxmax(), 'Model']
        best_value = df[metric].max()
        print(f"  Best {metric:<12}: {best_model:<30} ({best_value:.4f})")
    
    print("\n" + "="*100)


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("MODEL COMPARISON AND VISUALIZATION")
    print("="*80)
    
    # Load results
    print("\nLoading results...")
    results = load_all_results()
    
    if not results:
        print("✗ No results found! Please run model_training.py and ensemble_stacking.py first.")
        return
    
    print(f"✓ Loaded results from {len(results)} sources")
    
    # Create comparison dataframe
    df = create_comparison_dataframe(results)
    print(f"✓ Prepared comparison data for {len(df)} models")
    
    # Create summary table
    create_summary_table(df)
    
    # Create visualizations
    viz_dir = Path(ARTIFACTS_DIR) / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    print("\n1. Creating metrics comparison plot...")
    plot_metrics_comparison(df, viz_dir / 'model_comparison_all_metrics.png')
    
    print("\n2. Creating F1 vs Recall scatter plot...")
    plot_f1_recall_scatter(df, viz_dir / 'f1_vs_recall_scatter.png')
    
    print("\n3. Creating performance heatmap...")
    plot_metric_heatmap(df, viz_dir / 'performance_heatmap.png')
    
    print("\n4. Creating improvement comparison...")
    plot_improvement_bars(df, viz_dir / 'ensemble_improvement.png')
    
    print("\n" + "="*80)
    print("✓ ALL VISUALIZATIONS COMPLETE")
    print("="*80)
    print(f"\nVisualizations saved to: {viz_dir}")
    print("\nGenerated files:")
    print("  • model_comparison_all_metrics.png")
    print("  • f1_vs_recall_scatter.png")
    print("  • performance_heatmap.png")
    print("  • ensemble_improvement.png")


if __name__ == "__main__":
    main()
