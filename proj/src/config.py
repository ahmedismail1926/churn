"""
Configuration file for Bank Churn Prediction Project
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent  # Project root (one level up from src)
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
LOGS_DIR = ARTIFACTS_DIR / "logs"

# Data file paths
RAW_DATA_FILE = RAW_DATA_DIR / "BankChurners.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "processed_data.csv"

# Model parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Create directories if they don't exist
def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        LOGS_DIR
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    print("✓ All directories created/verified")

# Print folder structure
def print_folder_structure():
    """Print the folder structure confirmation"""
    print("\n" + "="*50)
    print("FOLDER STRUCTURE CONFIRMATION")
    print("="*50)
    print(f"\n📁 Base Directory: {BASE_DIR}")
    print(f"\n📂 Data:")
    print(f"   ├── Raw Data: {RAW_DATA_DIR}")
    print(f"   │   └── {RAW_DATA_FILE.name}: {'✓ Exists' if RAW_DATA_FILE.exists() else '✗ Not Found'}")
    print(f"   └── Processed Data: {PROCESSED_DATA_DIR}")
    print(f"\n📂 Artifacts:")
    print(f"   ├── Models: {MODELS_DIR}")
    print(f"   └── Logs: {LOGS_DIR}")
    print(f"\n⚙️  Configuration:")
    print(f"   ├── Random Seed: {RANDOM_SEED}")
    print(f"   ├── Test Size: {TEST_SIZE}")
    print(f"   └── Validation Size: {VALIDATION_SIZE}")
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    create_directories()
    print_folder_structure()
