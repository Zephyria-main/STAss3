# config.py
# I keep all paths and shared constants here so nothing is hard-coded
# across the project. If we move the dataset or change the image size,
# this is the only file we need to touch.

from pathlib import Path

# I resolve the project root relative to this file so the paths work
# regardless of where the user runs the scripts from.
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output directories
OUTPUTS_DIR = BASE_DIR / "outputs"
EDA_OUTPUT_DIR = OUTPUTS_DIR / "eda"
MODEL_OUTPUT_DIR = OUTPUTS_DIR / "models"
REPORT_OUTPUT_DIR = OUTPUTS_DIR / "reports"

# I use 128x128 as the standard resize target for the baseline classifier.
# Larger sizes improve accuracy but dramatically increase training time.
IMAGE_SIZE = (128, 128)

# Only these file extensions will be treated as valid images during indexing.
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# The saved model file name used by both the training and prediction steps.
MODEL_FILE_NAME = "macro_classifier.joblib"

# Random seed so our results are reproducible every run.
RANDOM_SEED = 42
