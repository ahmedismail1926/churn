# 🚀 QUICK START - One Command Setup

## Run Complete Pipeline

### Windows (Easiest):
```cmd
run_pipeline.bat
```

**Done!** Wait 15-20 minutes. All 7 models will be trained.

---

## Alternative: Stage Selection

```cmd
run_stage.bat
```

Choose stages 1-8 from interactive menu.

---

## Check Results

```cmd
python scripts\check_status.py    # Verify completion
python scripts\show_results.py    # View model metrics
```

---

## What Gets Created

✅ 7 trained models (Logistic, Naive Bayes, Random Forest, XGBoost, LightGBM, + 2 tuned, + ensemble)  
✅ All data splits (train/test CSV files)  
✅ 6 visualization plots  
✅ Complete results file  

**Location**: `artifacts/models/` and `data/processed/`

---

## Troubleshooting

**Error: Module not found**
```cmd
pip install pandas numpy scikit-learn xgboost lightgbm optuna imbalanced-learn matplotlib seaborn
```

**Want to start fresh?**
```cmd
del /Q data\processed\*.csv
del /Q artifacts\models\*.pkl
run_pipeline.bat
```

---

## More Info

- `PIPELINE_GUIDE.md` - Full documentation
- `PIPELINE_SUMMARY.md` - What was created
- `README.md` - Project overview
