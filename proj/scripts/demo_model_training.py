"""
Quick Demo of Model Training
Trains baseline and main models with reduced parameters for faster execution
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Import the trainer
from model_training import ModelTrainer, load_training_data
from config import RANDOM_SEED

def main():
    """Quick demo with reduced parameters"""
    print("\n" + "="*80)
    print("QUICK MODEL TRAINING DEMO")
    print("="*80)
    print("This is a faster version with fewer CV folds and tuning trials")
    print("="*80)
    
    # Load data
    X_train, X_test, y_train, y_test = load_training_data()
    
    # Initialize trainer with fewer folds for speed
    trainer = ModelTrainer(X_train, y_train, X_test, y_test, n_folds=3)
    
    # Train baseline models
    trainer.train_baseline_models()
    
    # Train main models with SMOTE-ENN
    trainer.train_main_models_with_smote()
    
    # Quick hyperparameter tuning (fewer trials)
    print("\n" + "="*80)
    print("QUICK HYPERPARAMETER TUNING")
    print("="*80)
    print("Using only 10 trials for demonstration")
    
    # Tune XGBoost (quick)
    trainer.tune_xgboost(n_trials=10, timeout=300)
    
    # Tune LightGBM (quick)
    trainer.tune_lightgbm(n_trials=10, timeout=300)
    
    # Print final comparison
    trainer.print_final_comparison()
    
    # Save models
    trainer.save_models()
    
    print("\n" + "="*80)
    print("QUICK DEMO COMPLETE")
    print("="*80)
    print("\nFor full training with more trials, run: python model_training.py")


if __name__ == "__main__":
    main()
