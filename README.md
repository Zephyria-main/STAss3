# Macroinvertebrate Image Analysis System

**Unit:** Software Technology 1 (4483/8995) — Assignment 3  
**Dataset:** [Kaggle Stream Macroinvertebrates](https://www.kaggle.com/datasets/kennethtm/stream-macroinvertebrates)

---

## Project Goal

This application analyses a macroinvertebrate image dataset through three stages: exploratory data analysis (Stage 1), image classification (Stage 2), and an interactive deployment interface (Stage 3). The goal is to demonstrate object-oriented software design using Python packages including Pandas, NumPy, OpenCV, Matplotlib, Seaborn, Scikit-learn, and Tkinter.

---

## Main Features

- Dataset indexing — scans the image folder and builds a structured table
- Class distribution analysis — bar chart of images per class
- Image size and channel distribution charts
- Aspect ratio scatter plot by class
- Sample image grid
- EDA summary text report
- Baseline Random Forest image classifier (grayscale, 128×128 features)
- Confusion matrix heatmap and classification report
- Interactive console menu application
- Tkinter GUI with image preview and live prediction

---

## Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd macro_project

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Dataset Setup

1. Download the dataset from Kaggle: https://www.kaggle.com/datasets/kennethtm/stream-macroinvertebrates
2. Extract the contents into `data/raw/`
3. The expected structure is: `data/raw/<ClassName>/<image>.jpg`

---

## How to Run

### Full non-interactive pipeline (Stage 1 + Stage 2)
```bash
python main.py
```

### Interactive console menu (all stages)
```bash
python -m src.console_app
```

### Tkinter GUI (all stages)
```bash
python -m src.app
```

---

## Folder Structure

```
macro_project/
├── data/
│   ├── raw/              ← Place Kaggle dataset here
│   └── processed/
├── outputs/
│   ├── eda/              ← EDA charts saved here
│   ├── models/           ← Trained model saved here
│   └── reports/          ← Classification report and confusion matrix
├── src/
│   ├── config.py         ← Central path and settings configuration
│   ├── main.py           ← (see root main.py)
│   ├── app.py            ← Tkinter GUI entry point
│   ├── console_app.py    ← Console menu entry point
│   ├── models/
│   │   └── records.py    ← ImageRecord dataclass
│   ├── services/
│   │   ├── dataset_indexer.py    ← Scans folders, builds DataFrame
│   │   ├── eda_service.py        ← Stage 1 charts and summaries
│   │   ├── image_preprocessor.py ← Image to feature vector
│   │   ├── classifier_service.py ← Train, evaluate, save model
│   │   └── workflow_service.py   ← Coordinates all pipeline steps
│   └── utils/
├── main.py               ← Non-interactive pipeline runner
├── requirements.txt
├── MANUAL_TESTING.md
└── README.md
```

---

## Class Overview

| Class | Responsibility |
|---|---|
| `DatasetIndexer` | Scans dataset folders and builds the image index DataFrame |
| `EDAService` | Generates charts, sample grids, and summary statistics |
| `ImagePreprocessor` | Converts images to normalised, flattened feature vectors |
| `ClassifierService` | Trains, evaluates, saves, and loads the prediction model |
| `WorkflowService` | Coordinates all services; called by both interface layers |
| `ConsoleApp` | Menu-driven console interface for all stages |
| `MacroApp` | Tkinter GUI interface for all stages |

---

## Packages Used

| Package | Purpose |
|---|---|
| `pathlib` | Clean, readable file and path handling |
| `pandas` | Stores the image index as a structured DataFrame |
| `numpy` | Numerical arrays for image features |
| `opencv-python` | Reads, resizes, and converts images |
| `matplotlib` | Chart generation for EDA and model evaluation |
| `seaborn` | Styled statistical charts (count plots, heatmaps) |
| `scikit-learn` | Train/test split, RandomForestClassifier, metrics |
| `joblib` | Save and load the trained model |
| `Pillow` | Image preview in the Tkinter GUI |
| `tkinter` | Desktop GUI for Stage 3 deployment |
