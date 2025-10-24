# Quick Start Guide - Bank Churn Prediction

## 🚀 Get Started in 5 Minutes

### Prerequisites
- Python 3.12+ installed
- Virtual environment activated

---

## Step 1: Setup Environment

```cmd
# Navigate to project
cd D:\GP\churn\proj

# Activate virtual environment
D:\GP\churn\.venv\Scripts\activate

# Set Python path (Windows CMD)
set PYTHONPATH=D:\GP\churn\proj;D:\GP\churn\proj\src

# Verify installation
python -c "import pandas, sklearn, xgboost, lightgbm, optuna; print('✅ All packages installed')"
```

---

## Step 2: Check Project Status

```cmd
python scripts\check_status.py
```

This will show:
- ✅ Which models are trained
- ✅ Which data files exist
- ✅ What needs to be done next

---

## Step 3: Run What You Need

### Option A: View Existing Results (Fastest - 5 seconds)
```cmd
python scripts\show_results.py
```

### Option B: Run Quick Demos (1-5 minutes each)
```cmd
# Test preprocessing
python scripts\demo_preprocessing.py

# Test feature engineering
python scripts\demo_feature_engineering.py

# Test model training (quick version)
python scripts\demo_model_training.py

# Test ensemble
python scripts\demo_ensemble.py
```

### Option C: Run Full Pipeline (30-40 minutes)
```cmd
# Complete training from scratch
python src\model_training.py

# Then create ensemble
python src\ensemble_stacking.py

# Generate visualizations
python src\visualize_results.py
```

### Option D: Run Main Pipeline (Complete - 45 minutes)
```cmd
# Full end-to-end pipeline
python main.py
```

---

## Common Commands

### Check Status
```cmd
python scripts\check_status.py
```

### View Results
```cmd
python scripts\show_results.py
```

### Train Models
```cmd
# With logging
python scripts\run_model_training.py

# Direct
python src\model_training.py
```

### Create Ensemble
```cmd
python src\ensemble_stacking.py
```

### Run Tests
```cmd
python tests\test_preprocessing.py
python tests\test_feature_engineering.py
```

---

## Quick Reference

### Project Structure
```
proj/
├── src/          # Core modules
├── scripts/      # Utilities & demos
├── tests/        # Test suite
├── docs/         # Documentation
├── outputs/      # Generated logs
├── data/         # Data files
├── artifacts/    # Models & plots
└── main.py       # Main pipeline
```

### Key Files
- `src/model_training.py` - Train all models
- `src/ensemble_stacking.py` - Create stacking ensemble
- `scripts/check_status.py` - Check what's complete
- `scripts/show_results.py` - View all results
- `docs/PROJECT_STRUCTURE.md` - Full documentation

---

## Expected Results

### Training Complete When You See:
- ✅ 3 baseline models trained
- ✅ 2 main models trained (with SMOTE-ENN)
- ✅ 2 tuned models (Optuna optimization)
- ✅ 1 stacking ensemble created

### Performance Targets:
- **F1-Score**: >97%
- **Recall**: >98%
- **Precision**: >99% (with SMOTE-ENN)

---

## Troubleshooting

### Import Errors
```cmd
# Set Python path first
set PYTHONPATH=D:\GP\churn\proj;D:\GP\churn\proj\src
```

### Models Not Found
```cmd
# Train models first
python src\model_training.py
```

### Slow Training
```cmd
# Use quick demos instead
python scripts\demo_model_training.py
```

---

## Next Steps After Setup

1. ✅ Check status: `python scripts\check_status.py`
2. ✅ View results: `python scripts\show_results.py`
3. ✅ Read documentation: `docs/FINAL_PROJECT_SUMMARY.md`
4. ✅ Explore code: Start with `src/` modules
5. ✅ Run tests: `tests/` directory

---

## Need Help?

📚 **Documentation**: See `docs/` folder
- `PROJECT_STRUCTURE.md` - Project organization
- `MODEL_TRAINING_SUMMARY.md` - Training guide
- `ENSEMBLE_STACKING_SUMMARY.md` - Ensemble guide
- `FINAL_PROJECT_SUMMARY.md` - Complete overview

🧪 **Examples**: See `scripts/demo_*.py` files

🧪 **Tests**: See `tests/` folder for validation

---

**That's it! You're ready to go!** 🎉

Run `python scripts\check_status.py` to see what's available.
