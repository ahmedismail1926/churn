# 🚀 Pipeline Execution Guide

## Complete Pipeline Stages

The Bank Churn Prediction project consists of **7 sequential stages**:

```
Stage 1: EDA & Preprocessing
   ↓
Stage 2: Feature Engineering
   ↓
Stage 3: Train-Test Splitting
   ↓
Stage 4: Model Training (7 models)
   ↓
Stage 5: Stacking Ensemble
   ↓
Stage 6: Results Display
   ↓
Stage 7: Status Check
```

## 🎯 Quick Start

### Option 1: Run Complete Pipeline (Recommended)
```cmd
run_pipeline.bat
```
This runs all 7 stages automatically. Estimated time: **15-20 minutes**

### Option 2: Run Individual Stages
```cmd
run_stage.bat
```
Interactive menu to select specific stages.

### Option 3: Manual Python Commands

#### Run Full Pipeline
```cmd
D:\GP\churn\.venv\Scripts\activate.bat
python full_pipeline.py
```

#### Run Individual Stages
```cmd
# Stage 1: EDA & Preprocessing
python main.py

# Stage 2: Feature Engineering
python scripts/demo_feature_engineering.py

# Stage 3: Data Splitting
python -c "import sys; sys.path.insert(0, 'src'); from data_splitting import prepare_data_for_modeling; from config import PROCESSED_DATA_DIR, RANDOM_SEED; prepare_data_for_modeling(filepath=PROCESSED_DATA_DIR / 'engineered_data.csv', save_splits=True)"

# Stage 4: Model Training
python src/model_training.py

# Stage 5: Stacking Ensemble
python src/ensemble_stacking.py

# Stage 6: View Results
python scripts/show_results.py

# Stage 7: Check Status
python scripts/check_status.py
```

## 📊 What Each Stage Does

### Stage 1: EDA & Preprocessing
- Loads raw data from `data/raw/BankChurners.csv`
- Performs exploratory data analysis
- Generates visualizations (correlation matrix, churn patterns)
- Drops unnecessary columns
- Handles highly correlated features (>0.95 threshold)
- **Output**: `data/processed/processed_data.csv`

### Stage 2: Feature Engineering
- Label encodes binary categorical variables
- One-hot encodes multi-category variables
- Scales numerical features using StandardScaler
- **Output**: `data/processed/engineered_data.csv`

### Stage 3: Train-Test Splitting
- Stratified split (80% train, 20% test)
- Maintains class balance in both sets
- Creates 5-fold cross-validation splits
- **Output**: `data/processed/splits/` (X_train, X_test, y_train, y_test)

### Stage 4: Model Training
Trains 7 models with 5-fold cross-validation:
1. **Baseline Models** (original data):
   - Logistic Regression
   - Naive Bayes
   - Random Forest

2. **Main Models** (SMOTE-ENN resampled):
   - XGBoost
   - LightGBM

3. **Tuned Models** (Optuna hyperparameter optimization):
   - XGBoost Tuned (30 trials)
   - LightGBM Tuned (30 trials)

**Output**: `artifacts/models/` (10 .pkl files)  
**Time**: ~10-15 minutes

### Stage 5: Stacking Ensemble
- Combines 5 best base models
- Generates out-of-fold predictions as meta-features
- Trains LogisticRegression meta-classifier
- Evaluates on test set
- **Output**: `artifacts/models/stacking_ensemble.pkl`

### Stage 6: Results Display
- Shows all model performance metrics
- Displays best models
- Compares baseline vs. tuned vs. ensemble
- **Output**: Console display

### Stage 7: Status Check
- Verifies all models are trained
- Checks data files exist
- Shows file sizes and completion status
- **Output**: Console summary

## 📁 Expected Outputs

After running the complete pipeline:

```
artifacts/
├── models/
│   ├── baseline_Logistic Regression.pkl
│   ├── baseline_Naive Bayes.pkl
│   ├── baseline_Random Forest.pkl
│   ├── main_XGBoost.pkl
│   ├── main_LightGBM.pkl
│   ├── xgboost_tuned.pkl
│   ├── lightgbm_tuned.pkl
│   ├── stacking_ensemble.pkl
│   ├── training_results.pkl
│   └── ensemble_results.pkl
├── visualizations/
│   ├── correlation_matrix.png
│   ├── churn_vs_gender.png
│   ├── churn_vs_age.png
│   ├── churn_vs_education.png
│   ├── churn_vs_income.png
│   └── credit_limit_vs_avg_open_to_buy.png
└── feature_engineering/
    └── encoders_scalers.pkl

data/
├── processed/
│   ├── processed_data.csv
│   ├── engineered_data.csv
│   └── splits/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       └── train_test_splits.pkl
└── raw/
    └── BankChurners.csv
```

## ⚠️ Prerequisites

1. **Virtual Environment Active**
   ```cmd
   D:\GP\churn\.venv\Scripts\activate.bat
   ```

2. **Required Packages Installed**
   ```cmd
   pip install pandas numpy scikit-learn xgboost lightgbm optuna imbalanced-learn matplotlib seaborn
   ```

3. **Raw Data Present**
   - File: `data/raw/BankChurners.csv`
   - Should contain 10,127 rows × 23 columns

## 🔧 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'matplotlib'"
**Solution**: Install visualization packages
```cmd
pip install matplotlib seaborn
```

### Error: "Training data not found"
**Solution**: Run stages in order (1 → 2 → 3 → 4)
```cmd
python main.py
python scripts/demo_feature_engineering.py
# Then continue with training
```

### Error: "No models found"
**Solution**: Run model training first
```cmd
python src/model_training.py
```

### Pipeline Stops Midway
**Solution**: Check the error message and run individual stages to identify the issue
```cmd
run_stage.bat
# Select the failed stage to run it separately
```

## 📈 Performance Expectations

- **EDA & Preprocessing**: ~30 seconds
- **Feature Engineering**: ~10 seconds
- **Data Splitting**: ~5 seconds
- **Model Training**: ~10-15 minutes
  - Baseline models: ~2 minutes
  - Main models (SMOTE-ENN): ~3 minutes
  - Tuned models (Optuna): ~10 minutes (30 trials each)
- **Ensemble**: ~2 minutes
- **Results & Status**: ~5 seconds

**Total Time**: 15-20 minutes for complete pipeline

## 🎓 Rerunning Stages

You can safely rerun any stage. The pipeline will:
- Overwrite previous outputs
- Use existing data from previous stages
- Update model files if retraining

To completely start fresh:
```cmd
# Delete processed data and models
del /Q data\processed\*.csv
del /Q data\processed\splits\*.csv
del /Q artifacts\models\*.pkl

# Then run complete pipeline
run_pipeline.bat
```

## 📝 Quick Commands Reference

| Command | Purpose |
|---------|---------|
| `run_pipeline.bat` | Run complete pipeline (all stages) |
| `run_stage.bat` | Interactive stage selector |
| `python main.py` | Stage 1 only |
| `python src/model_training.py` | Stage 4 only |
| `python scripts/check_status.py` | Verify completion |
| `python scripts/show_results.py` | Display results |

## 🚀 Production Deployment

For production use:
1. Run complete pipeline once: `run_pipeline.bat`
2. Save best model from `artifacts/models/`
3. Use saved model for predictions:
   ```python
   import pickle
   model = pickle.load(open('artifacts/models/stacking_ensemble.pkl', 'rb'))
   predictions = model.predict(new_data)
   ```

---

**Need Help?** Check `QUICK_START.md` or `docs/PROJECT_STRUCTURE.md`
