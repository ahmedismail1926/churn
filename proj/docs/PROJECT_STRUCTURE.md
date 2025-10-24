# Bank Churn Prediction - Project Structure

## 📁 Organized Project Structure

```
proj/
│
├── 📂 src/                          # Core source code modules
│   ├── __init__.py                  # Package initializer
│   ├── config.py                    # Configuration & paths
│   ├── data_loader.py               # Data loading utilities
│   ├── preprocessing.py             # Data preprocessing
│   ├── feature_engineering.py       # Feature creation
│   ├── data_splitting.py            # Train/test splitting
│   ├── resampling.py                # SMOTE-ENN & resampling
│   ├── model_training.py            # Model training pipeline
│   ├── ensemble_stacking.py         # Stacking ensemble
│   ├── visualizations.py            # Basic visualizations
│   ├── visualize_results.py         # Results visualization
│   ├── data_analysis.py             # Data analysis
│   └── correlation_analysis.py      # Correlation analysis
│
├── 📂 scripts/                      # Utility & helper scripts
│   ├── demo_preprocessing.py        # Quick preprocessing demo
│   ├── demo_feature_engineering.py  # Feature engineering demo
│   ├── demo_resampling.py           # Resampling demo
│   ├── demo_model_training.py       # Model training demo
│   ├── demo_ensemble.py             # Ensemble demo
│   ├── run_model_training.py        # Training with logging
│   ├── check_status.py              # Project status checker
│   ├── show_results.py              # Display results
│   └── fix_unicode.py               # Unicode fixer utility
│
├── 📂 tests/                        # Test suite
│   ├── __init__.py                  # Test package init
│   ├── test_preprocessing.py        # Preprocessing tests
│   ├── test_feature_engineering.py  # Feature engineering tests
│   ├── test_resampling.py           # Resampling tests
│   ├── test_data_splitting.py       # Splitting tests
│   ├── test_model_training.py       # Model training tests
│   ├── verify_engineered.py         # Verify engineered features
│   └── verify_splits.py             # Verify data splits
│
├── 📂 docs/                         # Documentation
│   ├── README.md                    # Main project documentation
│   ├── PREPROCESSING_SUMMARY.md     # Preprocessing guide
│   ├── FEATURE_ENGINEERING_SUMMARY.md # Feature engineering guide
│   ├── MODEL_TRAINING_SUMMARY.md    # Training documentation
│   ├── ENSEMBLE_STACKING_SUMMARY.md # Ensemble documentation
│   ├── FINAL_PROJECT_SUMMARY.md     # Complete project summary
│   └── REFACTORING_SUMMARY.md       # Refactoring notes
│
├── 📂 outputs/                      # Generated output files
│   ├── output.txt                   # General output
│   ├── preprocessing_output.txt     # Preprocessing logs
│   ├── feature_eng_output.txt       # Feature engineering logs
│   ├── splitting_output.txt         # Data splitting logs
│   ├── resampling_output_new.txt    # Resampling logs
│   └── model_training_output.txt    # Training logs
│
├── 📂 data/                         # Data directory
│   ├── raw/                         # Original data
│   │   └── BankChurners.csv         # Raw dataset
│   └── processed/                   # Processed data
│       ├── processed_data.csv       # Preprocessed data
│       ├── engineered_data.csv      # Feature-engineered data
│       └── splits/                  # Train/test splits
│           ├── X_train.csv
│           ├── X_test.csv
│           ├── y_train.csv
│           ├── y_test.csv
│           └── train_test_splits.pkl
│
├── 📂 artifacts/                    # Generated artifacts
│   ├── models/                      # Trained models
│   │   ├── baseline_Logistic Regression.pkl
│   │   ├── baseline_Naive Bayes.pkl
│   │   ├── baseline_Random Forest.pkl
│   │   ├── main_XGBoost.pkl
│   │   ├── main_LightGBM.pkl
│   │   ├── xgboost_tuned.pkl
│   │   ├── lightgbm_tuned.pkl
│   │   ├── stacking_ensemble.pkl
│   │   ├── training_results.pkl
│   │   └── ensemble_results.pkl
│   │
│   ├── visualizations/              # Generated plots
│   │   ├── churn_vs_age.png
│   │   ├── churn_vs_gender.png
│   │   ├── churn_vs_education.png
│   │   ├── churn_vs_income.png
│   │   ├── correlation_matrix.png
│   │   └── credit_limit_vs_avg_open_to_buy.png
│   │
│   ├── feature_engineering/         # Feature artifacts
│   │   └── encoders_scalers.pkl
│   │
│   └── logs/                        # Log files
│
├── 📂 __pycache__/                  # Python cache (ignored)
│
├── main.py                          # Main pipeline entry point
│
└── .gitignore                       # Git ignore rules (to create)

```

---

## 📋 Module Descriptions

### 🔧 Core Modules (`src/`)

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `config.py` | Project configuration, paths, constants | `create_directories()`, paths |
| `data_loader.py` | Load and inspect data | `load_data()`, `get_dataset_info()` |
| `preprocessing.py` | Clean and preprocess data | `preprocess_data()` |
| `feature_engineering.py` | Create new features | `engineer_features()` |
| `data_splitting.py` | Split data for training | `stratified_train_test_split()` |
| `resampling.py` | Handle class imbalance | `resample_data_smoteenn()` |
| `model_training.py` | Train ML models | `ModelTrainer` class |
| `ensemble_stacking.py` | Stacking ensemble | `StackingEnsemble` class |
| `visualizations.py` | Create visualizations | `plot_all_churn_patterns()` |
| `visualize_results.py` | Visualize model results | Comparison plots |
| `data_analysis.py` | Analyze data patterns | `analyze_missing_values()` |
| `correlation_analysis.py` | Correlation analysis | `run_full_correlation_analysis()` |

### 🎯 Scripts (`scripts/`)

| Script | Purpose | Usage |
|--------|---------|-------|
| `demo_*.py` | Quick demonstrations | Fast testing of modules |
| `run_model_training.py` | Training with logging | Production training |
| `check_status.py` | Check project status | Verify completeness |
| `show_results.py` | Display all results | Quick results view |
| `fix_unicode.py` | Fix unicode issues | Utility |

### 🧪 Tests (`tests/`)

| Test File | Purpose |
|-----------|---------|
| `test_preprocessing.py` | Test preprocessing functions |
| `test_feature_engineering.py` | Test feature creation |
| `test_resampling.py` | Test resampling methods |
| `test_data_splitting.py` | Test data splitting |
| `test_model_training.py` | Test model training |
| `verify_*.py` | Verification scripts |

---

## 🚀 How to Run

### Setup Python Path

For Windows (cmd):
```cmd
set PYTHONPATH=D:\GP\churn\proj;D:\GP\churn\proj\src
```

For PowerShell:
```powershell
$env:PYTHONPATH="D:\GP\churn\proj;D:\GP\churn\proj\src"
```

### Run Main Pipeline
```bash
# From project root
D:\GP\churn\.venv\Scripts\python.exe main.py
```

### Run Individual Modules
```bash
# Model training
D:\GP\churn\.venv\Scripts\python.exe src\model_training.py

# Ensemble
D:\GP\churn\.venv\Scripts\python.exe src\ensemble_stacking.py

# Check status
D:\GP\churn\.venv\Scripts\python.exe scripts\check_status.py

# Show results
D:\GP\churn\.venv\Scripts\python.exe scripts\show_results.py
```

### Run Demos (Quick Testing)
```bash
D:\GP\churn\.venv\Scripts\python.exe scripts\demo_preprocessing.py
D:\GP\churn\.venv\Scripts\python.exe scripts\demo_feature_engineering.py
D:\GP\churn\.venv\Scripts\python.exe scripts\demo_model_training.py
D:\GP\churn\.venv\Scripts\python.exe scripts\demo_ensemble.py
```

### Run Tests
```bash
D:\GP\churn\.venv\Scripts\python.exe tests\test_preprocessing.py
D:\GP\churn\.venv\Scripts\python.exe tests\test_feature_engineering.py
```

---

## 📊 Data Flow

```
1. Raw Data (data/raw/)
   ↓
2. Load & Analyze (data_loader.py, data_analysis.py)
   ↓
3. Preprocess (preprocessing.py)
   ↓ → processed_data.csv
4. Feature Engineering (feature_engineering.py)
   ↓ → engineered_data.csv
5. Split Data (data_splitting.py)
   ↓ → X_train, X_test, y_train, y_test
6. Resample (resampling.py with SMOTE-ENN)
   ↓
7. Train Models (model_training.py)
   ↓ → 7 trained models
8. Create Ensemble (ensemble_stacking.py)
   ↓ → stacking_ensemble.pkl
9. Visualize (visualize_results.py)
   ↓ → Plots & comparisons
```

---

## 📦 Dependencies

```python
# Core
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Resampling
imbalanced-learn>=0.11.0

# Advanced Models
xgboost>=2.0.0
lightgbm>=4.0.0

# Hyperparameter Tuning
optuna>=3.5.0

# Visualization (optional)
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## 🎯 Key Features of Organization

### ✅ **Separation of Concerns**
- **src/**: Core logic and algorithms
- **scripts/**: Utilities and demos
- **tests/**: Quality assurance
- **docs/**: Documentation
- **outputs/**: Generated files

### ✅ **Easy Navigation**
- Clear folder names
- Logical grouping
- Consistent naming

### ✅ **Maintainable**
- Each module has single responsibility
- Easy to find and update code
- Clear dependencies

### ✅ **Scalable**
- Easy to add new modules
- Can split further if needed
- Follows Python best practices

### ✅ **Professional**
- Industry-standard structure
- Proper package organization
- Ready for version control

---

## 🔄 Migration Notes

### What Changed:
- ✅ Created organized folder structure
- ✅ Moved all files to appropriate folders
- ✅ Added `__init__.py` files for packages
- ✅ Updated import paths in main.py

### What to Update:
If you have custom scripts, update imports:
```python
# Old
from config import MODELS_DIR

# New
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from config import MODELS_DIR
```

Or run from project root with PYTHONPATH set.

---

## 📝 Best Practices

### Adding New Modules:
1. Add to appropriate folder (`src/`, `scripts/`, `tests/`)
2. Follow naming convention: `lowercase_with_underscores.py`
3. Add docstrings and type hints
4. Update this documentation

### Running Scripts:
Always run from project root:
```bash
cd D:\GP\churn\proj
D:\GP\churn\.venv\Scripts\python.exe <path_to_script>
```

### Version Control:
Create `.gitignore`:
```
__pycache__/
*.pyc
*.pyo
.venv/
artifacts/models/*.pkl
artifacts/visualizations/*.png
data/processed/
outputs/*.txt
```

---

## 🎉 Benefits of This Structure

1. **Professional**: Follows industry standards
2. **Clear**: Easy to understand and navigate
3. **Maintainable**: Easy to update and extend
4. **Testable**: Tests are separate and organized
5. **Documented**: Comprehensive documentation
6. **Scalable**: Can grow without becoming messy

---

**Last Updated**: October 25, 2025  
**Project**: Bank Churn Prediction  
**Status**: ✅ Fully Organized
