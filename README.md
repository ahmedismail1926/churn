# 🏦 Bank Customer Churn Prediction

A comprehensive machine learning project for predicting bank customer churn using advanced ensemble techniques and class imbalance handling.

## 🎯 Overview

This project implements a complete ML pipeline for predicting customer churn with:
- **7 trained models** including baseline, SMOTE-enhanced, and hyperparameter-tuned models
- **Stacking ensemble** with out-of-fold predictions
- **Advanced preprocessing** with feature engineering and class balancing
- **Comprehensive evaluation** with cross-validation and multiple metrics

## 📊 Dataset

**Source:** Bank Churners dataset with 10,127 customers and 21 features

**Class Distribution:**
- Existing Customers: 16% (minority)
- Churned Customers: 84% (majority)
- **Handled with SMOTE-ENN resampling** (5.22:1 → 1.25:1 ratio)

## 🚀 Quick Start

### Prerequisites
```bash
python 3.12+
pip install -r requirements.txt
```

### Installation
```bash
git clone https://github.com/ahmedismail1926/churn.git
cd churn/proj
pip install pandas numpy scikit-learn xgboost lightgbm optuna imbalanced-learn
```

### Usage
```bash
# Full pipeline (preprocessing → training → ensemble)
python main.py

# Check training status
python scripts/check_status.py

# View results
python scripts/show_results.py

# Visualize performance
python scripts/visualize_results.py
```

## 🏗️ Project Structure

```
proj/
├── src/              # Core modules (preprocessing, training, ensemble)
├── scripts/          # Utilities and demos
├── tests/            # Unit tests
├── data/             # Raw and processed datasets
├── artifacts/        # Trained models and visualizations
├── docs/             # Documentation
└── main.py           # Main pipeline
```

## 🤖 Models

### Baseline Models
- Logistic Regression
- Naive Bayes
- Random Forest ⭐ Best baseline

### Main Models (SMOTE-ENN)
- XGBoost
- LightGBM

### Hyperparameter Tuned (Optuna - 30 trials each)
- XGBoost Tuned ⭐ Best overall
- LightGBM Tuned

### Ensemble
- **Stacking Ensemble** with 5 base models + LogisticRegression meta-classifier

## 📈 Performance Highlights

- **7 models trained** with 5-fold cross-validation
- **Best CV F1-Score**: 0.9885 (XGBoost Tuned during optimization)
- **Hyperparameter tuning**: 60 total trials (30 per model) using Optuna TPE sampler
- **Class imbalance handled**: SMOTE-ENN improved balance from 5.22:1 to 1.25:1
- **Comprehensive metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC tracked for all models

*All test set metrics available in `artifacts/models/training_results.pkl`*

## 🔧 Key Features

- **Preprocessing**: Missing value imputation, categorical encoding, feature scaling
- **Feature Engineering**: Interaction features, ratio features, polynomial features
- **Class Balancing**: SMOTE-ENN resampling (8,101 → 12,234 samples)
- **Hyperparameter Tuning**: Optuna with TPE sampler (30 trials per model)
- **Cross-Validation**: 5-fold stratified CV for robust evaluation
- **Ensemble Learning**: Stacking with out-of-fold predictions

## 📁 Key Files

- `src/model_training.py` - Complete training pipeline (666 lines)
- `src/ensemble_stacking.py` - Stacking ensemble implementation (450+ lines)
- `src/preprocessing.py` - Data preprocessing module
- `src/feature_engineering.py` - Feature creation and transformation
- `src/resampling.py` - SMOTE-ENN and class balancing

## 📊 Results

All models and results are saved in `artifacts/`:
- `models/` - 10 trained model files (.pkl)
- `visualizations/` - Performance plots and charts
- `training_results.pkl` - Complete training metrics
- `ensemble_results.pkl` - Ensemble performance

## 🔍 Model Details

### Training Process
1. **Baseline models** trained on original data (3 models)
2. **Main models** trained on SMOTE-ENN resampled data (2 models)
3. **Hyperparameter tuning** with Optuna (2 models, 60 total trials)
4. **Stacking ensemble** combining best models with meta-classifier

### Evaluation
- 5-fold stratified cross-validation
- Multiple metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Test set holdout (20%) for final evaluation
- Per-fold metrics tracking with detailed logging

## 📚 Documentation

- `docs/PROJECT_STRUCTURE.md` - Complete structure guide
- `QUICK_START.md` - Quick reference
- `docs/FEATURE_ENGINEERING_SUMMARY.md` - Feature details
- `docs/PREPROCESSING_SUMMARY.md` - Preprocessing steps

## 🛠️ Tech Stack

- **Python 3.12** - Core language
- **scikit-learn** - ML algorithms and preprocessing
- **XGBoost** - Gradient boosting
- **LightGBM** - Fast gradient boosting
- **Optuna** - Hyperparameter optimization
- **imbalanced-learn** - SMOTE-ENN resampling
- **pandas/numpy** - Data manipulation

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Ahmed Ismail**
- GitHub: [@ahmedismail1926](https://github.com/ahmedismail1926)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## ⭐ Show Your Support

Give a ⭐️ if this project helped you!

---

*Built with ❤️ for advanced churn prediction and machine learning*
