"""
Stacking Ensemble Module
Implements stacking ensemble by combining predictions from multiple models
Uses cross-validation predictions as meta-features for a meta-classifier
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
import time

# Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Local imports
from config import RANDOM_SEED, MODELS_DIR
from resampling import resample_data_smoteenn


class StackingEnsemble:
    """
    Stacking Ensemble Classifier
    Combines predictions from base models using a meta-classifier
    """
    
    def __init__(self, base_models, meta_classifier=None, n_folds=5, 
                 use_proba=True, random_state=RANDOM_SEED):
        """
        Initialize stacking ensemble
        
        Args:
            base_models: Dictionary of {model_name: model} to use as base models
            meta_classifier: Meta-classifier to train on base predictions (default: LogisticRegression)
            n_folds: Number of CV folds for generating meta-features
            use_proba: Use predicted probabilities instead of class labels
            random_state: Random seed for reproducibility
        """
        self.base_models = base_models
        self.meta_classifier = meta_classifier or LogisticRegression(
            random_state=random_state, 
            max_iter=1000,
            solver='lbfgs'
        )
        self.n_folds = n_folds
        self.use_proba = use_proba
        self.random_state = random_state
        self.is_fitted = False
        
        print("\n" + "="*80)
        print("STACKING ENSEMBLE INITIALIZED")
        print("="*80)
        print(f"Base models: {len(base_models)}")
        for name in base_models.keys():
            print(f"  • {name}")
        print(f"Meta-classifier: {type(self.meta_classifier).__name__}")
        print(f"CV folds: {n_folds}")
        print(f"Using probabilities: {use_proba}")
    
    
    def _get_oof_predictions(self, X, y, verbose=True):
        """
        Generate out-of-fold predictions for meta-features
        
        Args:
            X: Training features
            y: Training labels
            verbose: Whether to print progress
        
        Returns:
            Meta-features array (n_samples, n_base_models)
        """
        if verbose:
            print("\n" + "="*80)
            print("GENERATING OUT-OF-FOLD PREDICTIONS FOR META-FEATURES")
            print("="*80)
            print(f"Training data shape: {X.shape}")
            print(f"Number of base models: {len(self.base_models)}")
            print(f"CV folds: {self.n_folds}")
        
        # Initialize meta-features array
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        meta_features = np.zeros((n_samples, n_models))
        
        # Setup cross-validation
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # For each base model
        for model_idx, (model_name, model) in enumerate(self.base_models.items()):
            if verbose:
                print(f"\n[{model_idx + 1}/{n_models}] Generating OOF predictions for: {model_name}")
                start_time = time.time()
            
            # Cross-validation to get out-of-fold predictions
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_fold_train = X.iloc[train_idx]
                X_fold_val = X.iloc[val_idx]
                y_fold_train = y.iloc[train_idx]
                
                # Train model on fold
                model.fit(X_fold_train, y_fold_train)
                
                # Predict on validation fold
                if self.use_proba and hasattr(model, 'predict_proba'):
                    preds = model.predict_proba(X_fold_val)[:, 1]
                else:
                    preds = model.predict(X_fold_val)
                
                # Store predictions
                meta_features[val_idx, model_idx] = preds
                
                if verbose and fold_idx == 0:
                    print(f"  Fold {fold_idx + 1}/{self.n_folds}...", end="")
                elif verbose and (fold_idx + 1) % 2 == 0:
                    print(f" {fold_idx + 1}...", end="")
            
            if verbose:
                elapsed = time.time() - start_time
                print(f" Done! ({elapsed:.2f}s)")
                print(f"  OOF predictions shape: {meta_features[:, model_idx].shape}")
                print(f"  OOF predictions range: [{meta_features[:, model_idx].min():.4f}, "
                      f"{meta_features[:, model_idx].max():.4f}]")
        
        if verbose:
            print("\n" + "="*80)
            print(f"Meta-features generated: {meta_features.shape}")
            print("="*80)
        
        return meta_features
    
    
    def fit(self, X_train, y_train, apply_smote=True, verbose=True):
        """
        Fit the stacking ensemble
        
        Args:
            X_train: Training features
            y_train: Training labels
            apply_smote: Whether to apply SMOTE-ENN resampling
            verbose: Whether to print progress
        """
        print("\n" + "="*80)
        print("TRAINING STACKING ENSEMBLE")
        print("="*80)
        
        # Apply SMOTE-ENN if requested
        if apply_smote:
            print("\nApplying SMOTE-ENN resampling...")
            X_train_resampled, y_train_resampled, _ = resample_data_smoteenn(
                X_train, y_train,
                random_state=self.random_state,
                verbose=verbose
            )
        else:
            X_train_resampled = X_train
            y_train_resampled = y_train
            print("\nUsing original imbalanced data (no resampling)")
        
        print(f"\nTraining data shape: {X_train_resampled.shape}")
        
        # Generate out-of-fold predictions (meta-features)
        meta_features = self._get_oof_predictions(X_train_resampled, y_train_resampled, verbose)
        
        # Train meta-classifier on meta-features
        print("\n" + "="*80)
        print("TRAINING META-CLASSIFIER")
        print("="*80)
        print(f"Meta-classifier: {type(self.meta_classifier).__name__}")
        print(f"Meta-features shape: {meta_features.shape}")
        
        start_time = time.time()
        self.meta_classifier.fit(meta_features, y_train_resampled)
        elapsed = time.time() - start_time
        
        print(f"✓ Meta-classifier trained in {elapsed:.2f} seconds")
        
        # Train final base models on full training set
        print("\n" + "="*80)
        print("TRAINING FINAL BASE MODELS ON FULL TRAINING SET")
        print("="*80)
        
        for model_name, model in self.base_models.items():
            print(f"  Training {model_name}...", end="")
            start_time = time.time()
            model.fit(X_train_resampled, y_train_resampled)
            elapsed = time.time() - start_time
            print(f" Done! ({elapsed:.2f}s)")
        
        self.is_fitted = True
        print("\n" + "="*80)
        print("✓ STACKING ENSEMBLE TRAINING COMPLETE")
        print("="*80)
    
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the ensemble
        
        Args:
            X: Features to predict
        
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get predictions from base models
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        meta_features = np.zeros((n_samples, n_models))
        
        for model_idx, (model_name, model) in enumerate(self.base_models.items()):
            if self.use_proba and hasattr(model, 'predict_proba'):
                meta_features[:, model_idx] = model.predict_proba(X)[:, 1]
            else:
                meta_features[:, model_idx] = model.predict(X)
        
        # Get meta-classifier predictions
        return self.meta_classifier.predict_proba(meta_features)
    
    
    def predict(self, X):
        """
        Predict class labels using the ensemble
        
        Args:
            X: Features to predict
        
        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get predictions from base models
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        meta_features = np.zeros((n_samples, n_models))
        
        for model_idx, (model_name, model) in enumerate(self.base_models.items()):
            if self.use_proba and hasattr(model, 'predict_proba'):
                meta_features[:, model_idx] = model.predict_proba(X)[:, 1]
            else:
                meta_features[:, model_idx] = model.predict(X)
        
        # Get meta-classifier predictions
        return self.meta_classifier.predict(meta_features)
    
    
    def evaluate(self, X_test, y_test, verbose=True):
        """
        Evaluate the ensemble on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
            verbose: Whether to print results
        
        Returns:
            Dictionary of metrics
        """
        if verbose:
            print("\n" + "="*80)
            print("EVALUATING STACKING ENSEMBLE ON TEST SET")
            print("="*80)
        
        # Predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        if verbose:
            print(f"\nTest Set Performance:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1']:.4f}")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"\nConfusion Matrix:")
            print(f"                Predicted")
            print(f"                0       1")
            print(f"  Actual 0    {cm[0,0]:5d}   {cm[0,1]:5d}")
            print(f"         1    {cm[1,0]:5d}   {cm[1,1]:5d}")
            
            # Classification report
            print(f"\nDetailed Classification Report:")
            print(classification_report(y_test, y_pred, 
                                       target_names=['Existing', 'Churned'],
                                       digits=4))
        
        return metrics
    
    
    def get_base_model_predictions(self, X_test, y_test):
        """
        Get individual predictions from each base model
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary of metrics for each base model
        """
        print("\n" + "="*80)
        print("BASE MODEL PREDICTIONS ON TEST SET")
        print("="*80)
        
        base_metrics = {}
        
        for model_name, model in self.base_models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }
            
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            
            base_metrics[model_name] = metrics
            
            print(f"\n{model_name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1']:.4f}")
            if 'roc_auc' in metrics:
                print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return base_metrics


def load_trained_models(models_dir=None):
    """
    Load trained models from disk
    
    Args:
        models_dir: Directory containing saved models
    
    Returns:
        Dictionary of loaded models
    """
    if models_dir is None:
        models_dir = MODELS_DIR
    
    models_dir = Path(models_dir)
    
    print("\n" + "="*80)
    print("LOADING TRAINED MODELS")
    print("="*80)
    print(f"Models directory: {models_dir}")
    
    # Define models to load (best performing ones)
    model_files = {
        'Random Forest': 'baseline_Random Forest.pkl',
        'XGBoost': 'main_XGBoost.pkl',
        'LightGBM': 'main_LightGBM.pkl',
        'XGBoost (Tuned)': 'xgboost_tuned.pkl',
        'LightGBM (Tuned)': 'lightgbm_tuned.pkl'
    }
    
    loaded_models = {}
    
    for model_name, filename in model_files.items():
        filepath = models_dir / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            loaded_models[model_name] = model
            print(f"✓ Loaded: {model_name}")
        else:
            print(f"✗ Not found: {model_name} ({filename})")
    
    print(f"\n✓ Successfully loaded {len(loaded_models)} models")
    return loaded_models


def compare_ensemble_vs_best_model(ensemble_metrics, base_metrics, best_model_name):
    """
    Compare ensemble performance to the best individual model
    
    Args:
        ensemble_metrics: Metrics from stacking ensemble
        base_metrics: Dictionary of metrics from base models
        best_model_name: Name of the best individual model
    """
    print("\n" + "="*80)
    print("ENSEMBLE VS BEST MODEL COMPARISON")
    print("="*80)
    
    # Find best model
    if best_model_name not in base_metrics:
        best_f1 = max(base_metrics.items(), key=lambda x: x[1]['f1'])
        best_model_name = best_f1[0]
    
    best_metrics = base_metrics[best_model_name]
    
    print(f"\nBest Individual Model: {best_model_name}")
    print(f"{'Metric':<12} {'Best Model':<12} {'Ensemble':<12} {'Improvement':<12}")
    print("-"*60)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        if metric in ensemble_metrics and metric in best_metrics:
            best_val = best_metrics[metric]
            ensemble_val = ensemble_metrics[metric]
            improvement = ((ensemble_val - best_val) / best_val) * 100
            
            symbol = "↑" if improvement > 0 else "↓" if improvement < 0 else "="
            print(f"{metric.capitalize():<12} {best_val:<12.4f} {ensemble_val:<12.4f} "
                  f"{symbol} {abs(improvement):<10.2f}%")
    
    # Summary
    print("\n" + "="*80)
    ensemble_f1 = ensemble_metrics['f1']
    best_f1 = best_metrics['f1']
    
    if ensemble_f1 > best_f1:
        improvement = ((ensemble_f1 - best_f1) / best_f1) * 100
        print(f"✓ Stacking Ensemble OUTPERFORMS best model by {improvement:.2f}% (F1-Score)")
    elif ensemble_f1 < best_f1:
        degradation = ((best_f1 - ensemble_f1) / best_f1) * 100
        print(f"✗ Stacking Ensemble UNDERPERFORMS best model by {degradation:.2f}% (F1-Score)")
    else:
        print(f"= Stacking Ensemble performs EQUAL to best model (F1-Score)")
    
    print("="*80)


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("STACKING ENSEMBLE - BANK CHURN PREDICTION")
    print("="*80)
    print("This script creates a stacking ensemble from trained models")
    print("="*80)
    
    # Load data
    from model_training import load_training_data
    X_train, X_test, y_train, y_test = load_training_data()
    
    # Load trained models
    base_models = load_trained_models()
    
    if len(base_models) == 0:
        print("\n✗ No trained models found!")
        print("Please run model_training.py first to train base models.")
        return
    
    # Create stacking ensemble
    ensemble = StackingEnsemble(
        base_models=base_models,
        meta_classifier=LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
        n_folds=5,
        use_proba=True,
        random_state=RANDOM_SEED
    )
    
    # Train ensemble
    ensemble.fit(X_train, y_train, apply_smote=True, verbose=True)
    
    # Evaluate ensemble
    ensemble_metrics = ensemble.evaluate(X_test, y_test, verbose=True)
    
    # Get base model predictions
    base_metrics = ensemble.get_base_model_predictions(X_test, y_test)
    
    # Compare ensemble vs best model
    compare_ensemble_vs_best_model(ensemble_metrics, base_metrics, 'XGBoost (Tuned)')
    
    # Save ensemble
    print("\n" + "="*80)
    print("SAVING STACKING ENSEMBLE")
    print("="*80)
    
    ensemble_path = MODELS_DIR / 'stacking_ensemble.pkl'
    with open(ensemble_path, 'wb') as f:
        pickle.dump(ensemble, f)
    print(f"✓ Ensemble saved to: {ensemble_path}")
    
    # Save results
    results = {
        'ensemble_metrics': ensemble_metrics,
        'base_metrics': base_metrics
    }
    results_path = MODELS_DIR / 'ensemble_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Results saved to: {results_path}")
    
    print("\n" + "="*80)
    print("STACKING ENSEMBLE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
