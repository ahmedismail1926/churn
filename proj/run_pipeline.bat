@echo off
REM ===========================================================================
REM Bank Churn Prediction - Complete Pipeline Runner
REM Runs all stages: EDA, Feature Engineering, Splitting, Training, Ensemble
REM ===========================================================================

echo.
echo ============================================================================
echo           BANK CHURN PREDICTION - COMPLETE PIPELINE
echo ============================================================================
echo.
echo This script will run the complete machine learning pipeline:
echo   [Stage 1] EDA ^& Preprocessing
echo   [Stage 2] Feature Engineering  
echo   [Stage 3] Train-Test Splitting
echo   [Stage 4] Model Training (7 models)
echo   [Stage 5] Stacking Ensemble
echo   [Stage 6] Results Display
echo   [Stage 7] Status Check
echo.
echo Estimated time: 15-20 minutes
echo ============================================================================
echo.

REM Change to project directory
cd /d D:\GP\churn\proj

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call D:\GP\churn\.venv\Scripts\activate.bat

REM Verify Python environment
echo [INFO] Verifying Python environment...
python --version
echo.

REM Run the complete pipeline
echo [INFO] Starting complete pipeline...
echo.
python full_pipeline.py

echo.
echo ============================================================================
echo                         PIPELINE EXECUTION COMPLETE
echo ============================================================================
echo.
echo Check the results in:
echo   - artifacts/models/         (trained models)
echo   - artifacts/visualizations/ (plots and charts)
echo   - data/processed/           (processed datasets)
echo.
echo Quick commands:
echo   python scripts/check_status.py  - Check training status
echo   python scripts/show_results.py  - Display model results
echo.
echo ============================================================================

pause
