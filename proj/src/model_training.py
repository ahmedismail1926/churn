"""
Model Training Module
Trains baseline and main models with hyperparameter tuning using Optuna
Includes cross-validation with detailed progress tracking
"""
import pandas as pd
import numpy as np
import pickle
import time
import warnings
from pathlib import Path
from collections import defaultdict

# Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# XGBoost and LightGBM
import xgboost as xgb
import lightgbm as lgb

# Optuna for hyperparameter tuning
import optuna
from optuna.samplers import TPESampler

# Imbalanced-learn for SMOTE-ENN
from imblearn.combine import SMOTEENN

# Local imports
from config import RANDOM_SEED, PROCESSED_DATA_DIR, MODELS_DIR
from resampling import resample_data_smoteenn

warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Comprehensive model training class with baseline and advanced models
    """
    
    def __init__(self, X_train, y_train, X_test, y_test, n_folds=5, random_state=RANDOM_SEED):
        """
        Initialize the model trainer
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            n_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_folds = n_folds
        self.random_state = random_state
        
        # Storage for results
        self.baseline_results = {}
        self.main_results = {}
        self.best_models = {}
        
        print("\n" + "="*80)
        print("MODEL TRAINER INITIALIZED")
        print("="*80)
        print(f"Training set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        print(f"Cross-validation folds: {n_folds}")
        print(f"Random state: {random_state}")
        print(f"Class distribution in training set:")
        print(f"  Class 0 (Existing): {sum(y_train == 0):,} ({sum(y_train == 0)/len(y_train)*100:.2f}%)")
        print(f"  Class 1 (Churned): {sum(y_train == 1):,} ({sum(y_train == 1)/len(y_train)*100:.2f}%)")
    
    
    def _print_fold_metrics(self, fold, metrics, indent="    "):
        """Print metrics for a specific fold"""
        print(f"{indent}Fold {fold + 1}:")
        print(f"{indent}  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"{indent}  Precision: {metrics['precision']:.4f}")
        print(f"{indent}  Recall:    {metrics['recall']:.4f}")
        print(f"{indent}  F1-Score:  {metrics['f1']:.4f}")
        if 'roc_auc' in metrics:
            print(f"{indent}  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    
    def _print_cv_summary(self, cv_results, model_name):
        """Print cross-validation summary statistics"""
        print(f"\n  Cross-Validation Summary for {model_name}:")
        print(f"  {'Metric':<12} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print(f"  {'-'*52}")
        
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            if metric in cv_results:
                scores = cv_results[metric]
                print(f"  {metric.capitalize():<12} {np.mean(scores):<10.4f} "
                      f"{np.std(scores):<10.4f} {np.min(scores):<10.4f} "
                      f"{np.max(scores):<10.4f}")
    
    
    def train_with_cv(self, model, model_name, X, y, verbose=True):
        """
        Train model with cross-validation and detailed progress tracking
        
        Args:
            model: Model instance
            model_name: Name of the model
            X: Features
            y: Labels
            verbose: Whether to print detailed progress
        
        Returns:
            Dictionary with CV results and trained model
        """
        if verbose:
            print(f"\n  Training {model_name} with {self.n_folds}-fold CV...")
            start_time = time.time()
        
        # Setup cross-validation
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # Store metrics for each fold
        cv_results = defaultdict(list)
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model.fit(X_fold_train, y_fold_train)
            
            # Predict
            y_pred = model.predict(X_fold_val)
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            fold_metrics = {
                'accuracy': accuracy_score(y_fold_val, y_pred),
                'precision': precision_score(y_fold_val, y_pred, zero_division=0),
                'recall': recall_score(y_fold_val, y_pred, zero_division=0),
                'f1': f1_score(y_fold_val, y_pred, zero_division=0)
            }
            
            if y_pred_proba is not None:
                fold_metrics['roc_auc'] = roc_auc_score(y_fold_val, y_pred_proba)
            
            # Store results
            for metric, value in fold_metrics.items():
                cv_results[metric].append(value)
            
            # Print fold results
            if verbose:
                self._print_fold_metrics(fold, fold_metrics)
        
        # Train final model on all data
        model.fit(X, y)
        
        # Evaluate on test set
        y_test_pred = model.predict(self.X_test)
        y_test_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        test_metrics = {
            'accuracy': accuracy_score(self.y_test, y_test_pred),
            'precision': precision_score(self.y_test, y_test_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_test_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_test_pred, zero_division=0)
        }
        
        if y_test_proba is not None:
            test_metrics['roc_auc'] = roc_auc_score(self.y_test, y_test_proba)
        
        if verbose:
            self._print_cv_summary(cv_results, model_name)
            elapsed_time = time.time() - start_time
            print(f"\n  Test Set Performance:")
            print(f"    Accuracy:  {test_metrics['accuracy']:.4f}")
            print(f"    Precision: {test_metrics['precision']:.4f}")
            print(f"    Recall:    {test_metrics['recall']:.4f}")
            print(f"    F1-Score:  {test_metrics['f1']:.4f}")
            if 'roc_auc' in test_metrics:
                print(f"    ROC-AUC:   {test_metrics['roc_auc']:.4f}")
            print(f"\n  Training completed in {elapsed_time:.2f} seconds")
        
        return {
            'model': model,
            'cv_results': cv_results,
            'test_metrics': test_metrics,
            'training_time': elapsed_time if verbose else 0
        }
    
    
    def train_baseline_models(self):
        """
        Train baseline models: Logistic Regression, Naive Bayes, Random Forest
        """
        print("\n" + "="*80)
        print("TRAINING BASELINE MODELS")
        print("="*80)
        print("\nBaseline models use the original imbalanced training data.")
        print("These serve as a performance benchmark.\n")
        
        baseline_models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='lbfgs'
            ),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        }
        
        for name, model in baseline_models.items():
            print(f"\n{'─'*80}")
            print(f"[BASELINE] {name}")
            print(f"{'─'*80}")
            
            result = self.train_with_cv(model, name, self.X_train, self.y_train)
            self.baseline_results[name] = result
            self.best_models[f'baseline_{name}'] = result['model']
        
        print("\n" + "="*80)
        print("BASELINE MODELS TRAINING COMPLETE")
        print("="*80)
        self._print_baseline_summary()
    
    
    def _print_baseline_summary(self):
        """Print summary comparison of baseline models"""
        print("\nBaseline Models Comparison (Test Set):")
        print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-"*75)
        
        for name, result in self.baseline_results.items():
            metrics = result['test_metrics']
            print(f"{name:<25} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")
    
    
    def train_main_models_with_smote(self):
        """
        Train main models (XGBoost, LightGBM) with SMOTE-ENN resampling
        """
        print("\n" + "="*80)
        print("TRAINING MAIN MODELS WITH SMOTE-ENN")
        print("="*80)
        print("\nApplying SMOTE-ENN resampling to handle class imbalance...")
        
        # Apply SMOTE-ENN resampling
        X_resampled, y_resampled, stats = resample_data_smoteenn(
            self.X_train, self.y_train, 
            random_state=self.random_state,
            verbose=True
        )
        
        print(f"\nResampled training set size: {X_resampled.shape}")
        print(f"Class distribution after SMOTE-ENN:")
        print(f"  Class 0: {sum(y_resampled == 0):,} ({sum(y_resampled == 0)/len(y_resampled)*100:.2f}%)")
        print(f"  Class 1: {sum(y_resampled == 1):,} ({sum(y_resampled == 1)/len(y_resampled)*100:.2f}%)")
        
        # Define main models
        main_models = {
            'XGBoost': xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            'LightGBM': lgb.LGBMClassifier(
                random_state=self.random_state,
                verbose=-1
            )
        }
        
        for name, model in main_models.items():
            print(f"\n{'─'*80}")
            print(f"[MAIN MODEL] {name} (with SMOTE-ENN)")
            print(f"{'─'*80}")
            
            result = self.train_with_cv(model, name, X_resampled, y_resampled)
            self.main_results[name] = result
            self.best_models[f'main_{name}'] = result['model']
        
        print("\n" + "="*80)
        print("MAIN MODELS TRAINING COMPLETE")
        print("="*80)
        self._print_main_summary()
    
    
    def _print_main_summary(self):
        """Print summary comparison of main models"""
        print("\nMain Models Comparison (Test Set - with SMOTE-ENN):")
        print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-"*75)
        
        for name, result in self.main_results.items():
            metrics = result['test_metrics']
            print(f"{name:<25} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")
    
    
    def tune_xgboost(self, n_trials=50, timeout=600):
        """
        Hyperparameter tuning for XGBoost using Optuna
        
        Args:
            n_trials: Number of optimization trials
            timeout: Maximum time in seconds for optimization
        
        Returns:
            Best parameters and tuned model
        """
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING: XGBoost with Optuna")
        print("="*80)
        print(f"Number of trials: {n_trials}")
        print(f"Timeout: {timeout} seconds")
        print(f"Optimization metric: F1-Score (focus on recall for churn)")
        
        # Apply SMOTE-ENN for tuning
        print("\nApplying SMOTE-ENN resampling...")
        X_resampled, y_resampled, _ = resample_data_smoteenn(
            self.X_train, self.y_train,
            random_state=self.random_state,
            verbose=False
        )
        print(f"Resampled data shape: {X_resampled.shape}")
        
        def objective(trial):
            """Optuna objective function for XGBoost"""
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'random_state': self.random_state,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            }
            
            # Cross-validation
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            f1_scores = []
            
            for train_idx, val_idx in skf.split(X_resampled, y_resampled):
                X_fold_train = X_resampled.iloc[train_idx]
                X_fold_val = X_resampled.iloc[val_idx]
                y_fold_train = y_resampled.iloc[train_idx]
                y_fold_val = y_resampled.iloc[val_idx]
                
                model = xgb.XGBClassifier(**params)
                model.fit(X_fold_train, y_fold_train, verbose=False)
                
                y_pred = model.predict(X_fold_val)
                f1 = f1_score(y_fold_val, y_pred)
                f1_scores.append(f1)
            
            return np.mean(f1_scores)
        
        # Create study and optimize
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        
        print("\nStarting optimization...\n")
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
            callbacks=[self._optuna_callback]
        )
        
        # Print results
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"\nBest trial: #{study.best_trial.number}")
        print(f"Best F1-Score (CV): {study.best_value:.4f}")
        print(f"\nBest Parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # Train final model with best parameters
        print("\nTraining final model with best parameters...")
        best_params = study.best_params.copy()
        best_params.update({
            'random_state': self.random_state,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        })
        
        best_model = xgb.XGBClassifier(**best_params)
        result = self.train_with_cv(best_model, "XGBoost (Tuned)", X_resampled, y_resampled, verbose=True)
        
        self.main_results['XGBoost_tuned'] = result
        self.best_models['xgboost_tuned'] = result['model']
        
        return study.best_params, result['model']
    
    
    def tune_lightgbm(self, n_trials=50, timeout=600):
        """
        Hyperparameter tuning for LightGBM using Optuna
        
        Args:
            n_trials: Number of optimization trials
            timeout: Maximum time in seconds for optimization
        
        Returns:
            Best parameters and tuned model
        """
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING: LightGBM with Optuna")
        print("="*80)
        print(f"Number of trials: {n_trials}")
        print(f"Timeout: {timeout} seconds")
        print(f"Optimization metric: F1-Score (focus on recall for churn)")
        
        # Apply SMOTE-ENN for tuning
        print("\nApplying SMOTE-ENN resampling...")
        X_resampled, y_resampled, _ = resample_data_smoteenn(
            self.X_train, self.y_train,
            random_state=self.random_state,
            verbose=False
        )
        print(f"Resampled data shape: {X_resampled.shape}")
        
        def objective(trial):
            """Optuna objective function for LightGBM"""
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'random_state': self.random_state,
                'verbose': -1
            }
            
            # Cross-validation
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            f1_scores = []
            
            for train_idx, val_idx in skf.split(X_resampled, y_resampled):
                X_fold_train = X_resampled.iloc[train_idx]
                X_fold_val = X_resampled.iloc[val_idx]
                y_fold_train = y_resampled.iloc[train_idx]
                y_fold_val = y_resampled.iloc[val_idx]
                
                model = lgb.LGBMClassifier(**params)
                model.fit(X_fold_train, y_fold_train)
                
                y_pred = model.predict(X_fold_val)
                f1 = f1_score(y_fold_val, y_pred)
                f1_scores.append(f1)
            
            return np.mean(f1_scores)
        
        # Create study and optimize
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        
        print("\nStarting optimization...\n")
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
            callbacks=[self._optuna_callback]
        )
        
        # Print results
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"\nBest trial: #{study.best_trial.number}")
        print(f"Best F1-Score (CV): {study.best_value:.4f}")
        print(f"\nBest Parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # Train final model with best parameters
        print("\nTraining final model with best parameters...")
        best_params = study.best_params.copy()
        best_params.update({
            'random_state': self.random_state,
            'verbose': -1
        })
        
        best_model = lgb.LGBMClassifier(**best_params)
        result = self.train_with_cv(best_model, "LightGBM (Tuned)", X_resampled, y_resampled, verbose=True)
        
        self.main_results['LightGBM_tuned'] = result
        self.best_models['lightgbm_tuned'] = result['model']
        
        return study.best_params, result['model']
    
    
    def _optuna_callback(self, study, trial):
        """Callback to print trial progress"""
        if trial.number % 5 == 0:
            print(f"Trial {trial.number}: F1-Score = {trial.value:.4f}")
    
    
    def print_final_comparison(self):
        """Print final comparison of all models"""
        print("\n" + "="*80)
        print("FINAL MODEL COMPARISON")
        print("="*80)
        
        # Combine all results
        all_results = {}
        
        for name, result in self.baseline_results.items():
            all_results[f"[BASELINE] {name}"] = result['test_metrics']
        
        for name, result in self.main_results.items():
            model_label = f"[MAIN] {name}"
            if 'tuned' in name:
                model_label = f"[TUNED] {name.replace('_tuned', '')}"
            all_results[model_label] = result['test_metrics']
        
        # Print comparison table
        print(f"\n{'Model':<35} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
        print("-"*95)
        
        for name, metrics in all_results.items():
            roc_auc = metrics.get('roc_auc', 0)
            print(f"{name:<35} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f} {roc_auc:<10.4f}")
        
        # Find best model
        best_f1_model = max(all_results.items(), key=lambda x: x[1]['f1'])
        best_recall_model = max(all_results.items(), key=lambda x: x[1]['recall'])
        
        print(f"\n{'='*80}")
        print(f"Best F1-Score: {best_f1_model[0]} ({best_f1_model[1]['f1']:.4f})")
        print(f"Best Recall: {best_recall_model[0]} ({best_recall_model[1]['recall']:.4f})")
        print(f"{'='*80}")
    
    
    def save_models(self, save_dir=None):
        """Save all trained models"""
        if save_dir is None:
            save_dir = MODELS_DIR
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print("SAVING MODELS")
        print(f"{'='*80}")
        print(f"Save directory: {save_dir}")
        
        for model_name, model in self.best_models.items():
            filepath = save_dir / f"{model_name}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ Saved: {model_name}.pkl")
        
        # Save results summary
        results_filepath = save_dir / "training_results.pkl"
        results = {
            'baseline_results': self.baseline_results,
            'main_results': self.main_results
        }
        with open(results_filepath, 'wb') as f:
            pickle.dump(results, f)
        print(f"✓ Saved: training_results.pkl")
        
        print(f"\n{'='*80}")
        print(f"All models saved successfully!")
        print(f"{'='*80}")


def load_training_data(data_dir=None):
    """Load training and test data from splits"""
    if data_dir is None:
        data_dir = PROCESSED_DATA_DIR / "splits"
    
    print("\n" + "="*80)
    print("LOADING TRAINING DATA")
    print("="*80)
    print(f"Data directory: {data_dir}")
    
    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()
    
    print(f"\n✓ Loaded X_train: {X_train.shape}")
    print(f"✓ Loaded X_test: {X_test.shape}")
    print(f"✓ Loaded y_train: {y_train.shape}")
    print(f"✓ Loaded y_test: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("BANK CHURN PREDICTION - MODEL TRAINING PIPELINE")
    print("="*80)
    print("This script trains baseline and main models with hyperparameter tuning")
    print("="*80)
    
    # Load data
    X_train, X_test, y_train, y_test = load_training_data()
    
    # Initialize trainer
    trainer = ModelTrainer(X_train, y_train, X_test, y_test, n_folds=5)
    
    # Train baseline models
    trainer.train_baseline_models()
    
    # Train main models with SMOTE-ENN
    trainer.train_main_models_with_smote()
    
    # Hyperparameter tuning
    print("\n" + "="*80)
    print("STARTING HYPERPARAMETER TUNING")
    print("="*80)
    
    # Tune XGBoost
    trainer.tune_xgboost(n_trials=30, timeout=600)
    
    # Tune LightGBM
    trainer.tune_lightgbm(n_trials=30, timeout=600)
    
    # Print final comparison
    trainer.print_final_comparison()
    
    # Save models
    trainer.save_models()
    
    print("\n" + "="*80)
    print("MODEL TRAINING PIPELINE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
