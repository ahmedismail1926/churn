@echo off
REM Quick Start - Run Individual Pipeline Stages

echo.
echo ============================================================================
echo           BANK CHURN PREDICTION - STAGE SELECTOR
echo ============================================================================
echo.
echo Select which stage to run:
echo.
echo   [1] EDA ^& Preprocessing only
echo   [2] Feature Engineering only
echo   [3] Data Splitting only
echo   [4] Model Training only (requires stages 1-3)
echo   [5] Stacking Ensemble only (requires stage 4)
echo   [6] Show Results
echo   [7] Check Status
echo   [8] Run COMPLETE Pipeline (all stages)
echo   [9] Exit
echo.
echo ============================================================================
echo.

set /p choice="Enter your choice (1-9): "

cd /d D:\GP\churn\proj
call D:\GP\churn\.venv\Scripts\activate.bat

if "%choice%"=="1" (
    echo.
    echo [Stage 1] Running EDA ^& Preprocessing...
    python main.py
) else if "%choice%"=="2" (
    echo.
    echo [Stage 2] Running Feature Engineering...
    python scripts/demo_feature_engineering.py
) else if "%choice%"=="3" (
    echo.
    echo [Stage 3] Running Data Splitting...
    python -c "import sys; sys.path.insert(0, 'src'); from data_splitting import prepare_data_for_modeling; from config import PROCESSED_DATA_DIR, RANDOM_SEED; prepare_data_for_modeling(filepath=PROCESSED_DATA_DIR / 'engineered_data.csv', target_column='Attrition_Flag', test_size=0.2, n_folds=5, random_state=RANDOM_SEED, save_splits=True)"
) else if "%choice%"=="4" (
    echo.
    echo [Stage 4] Running Model Training...
    echo This will take 10-15 minutes...
    python src/model_training.py
) else if "%choice%"=="5" (
    echo.
    echo [Stage 5] Running Stacking Ensemble...
    python src/ensemble_stacking.py
) else if "%choice%"=="6" (
    echo.
    echo [Stage 6] Showing Results...
    python scripts/show_results.py
) else if "%choice%"=="7" (
    echo.
    echo [Stage 7] Checking Status...
    python scripts/check_status.py
) else if "%choice%"=="8" (
    echo.
    echo [Complete Pipeline] Running all stages...
    python full_pipeline.py
) else if "%choice%"=="9" (
    echo.
    echo Exiting...
    exit /b
) else (
    echo.
    echo Invalid choice. Please run again and select 1-9.
)

echo.
echo ============================================================================
echo                         EXECUTION COMPLETE
echo ============================================================================
pause
