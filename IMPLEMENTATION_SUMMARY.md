# Implementation Summary

**Project Title:** Macroinvertebrate Image Analysis System  
**Unit:** Software Technology 1 (4483/8995) — Assignment 3  
**Group Members:** Kes Jayaweera · [Member 2 Name] · [Member 3 Name]

---

## 1. Project Goal

We built a modular Python application that analyses freshwater macroinvertebrate images across three stages: exploratory data analysis (Stage 1), baseline image classification (Stage 2), and interactive deployment via both a console menu and a Tkinter GUI (Stage 3). The system can index a local image dataset, generate EDA charts and statistics, train a Random Forest classifier, and allow a user to predict the class of a new image.

---

## 2. System Design Overview

I designed the system as a service-oriented Python application with clear separation between data management, analysis, machine learning, and user interaction. Each class has one clear responsibility and dependencies are injected at construction time so the classes are easy to test and replace independently.

The `WorkflowService` acts as the coordinator. Both the `ConsoleApp` and `MacroApp` classes call the `WorkflowService` instead of the individual services directly. This means the logic is never duplicated between the two interface types.

---

## 3. Class and Module Overview

| Class / Module | Location | Responsibility |
|---|---|---|
| `config.py` | `src/config.py` | Central path and constant configuration |
| `ImageRecord` | `src/models/records.py` | Dataclass for one indexed image's metadata |
| `DatasetIndexer` | `src/services/dataset_indexer.py` | Scans dataset folders recursively, builds DataFrame |
| `EDAService` | `src/services/eda_service.py` | Generates Stage 1 charts, summaries, and reports |
| `ImagePreprocessor` | `src/services/image_preprocessor.py` | Converts images to normalised feature vectors |
| `ClassifierService` | `src/services/classifier_service.py` | Trains, evaluates, saves, and loads the classifier |
| `WorkflowService` | `src/services/workflow_service.py` | Coordinates all pipeline steps for both interfaces |
| `ConsoleApp` | `src/console_app.py` | Menu-driven Stage 3 console interface |
| `MacroApp` | `src/app.py` | Tkinter Stage 3 GUI with image preview |

---

## 4. Python Packages Used and Why

- **Pandas** — I used Pandas to store the image index as a structured DataFrame so EDA and training both work from the same table.
- **NumPy** — I used NumPy for the feature matrix and label arrays that the classifier requires.
- **OpenCV (cv2)** — I used OpenCV to read image files from disk, convert to grayscale, and resize to a consistent dimension.
- **Matplotlib and Seaborn** — I used these for all EDA visualisations including bar charts, histograms, scatter plots, and the confusion matrix heatmap.
- **Scikit-learn** — I used scikit-learn for the train/test split, RandomForestClassifier, and evaluation metrics.
- **joblib** — I used joblib to save and load the trained model so prediction can happen in a separate session.
- **Pillow** — I used Pillow to load and thumbnail images for preview inside the Tkinter GUI.
- **tkinter** — I used the built-in Tkinter library for the Stage 3 desktop GUI.
- **pathlib** — I used pathlib throughout for readable, cross-platform file path handling.

---

## 5. Key Features Implemented

- Recursive dataset scanning with class label extraction from folder names
- Five EDA charts: class distribution, image size histograms, sample grid, channel distribution, aspect ratio scatter
- EDA summary statistics written to a text report
- Grayscale + resize + normalise + flatten image preprocessing pipeline
- RandomForestClassifier with stratified 80/20 train/test split and balanced class weights
- Confusion matrix heatmap and classification report saved to outputs/reports/
- Model persistence via joblib
- Console menu with six options including error handling for missing model, bad paths, and invalid input
- Tkinter GUI with image preview, threaded background tasks, progress indicator, and output log
- Non-interactive `main.py` pipeline runner for batch operation

---

## 6. What the EDA Revealed

The class distribution chart confirmed significant imbalance across macroinvertebrate species — some classes have many more images than others. This informed our decision to use `class_weight="balanced"` in the RandomForestClassifier. The image size distribution showed that most images are not a consistent size, which confirmed that resizing to 128×128 was the right preprocessing step.

---

## 7. Testing Summary

I tested the application manually across ten scenarios documented in `MANUAL_TESTING.md`. These included missing dataset folder, invalid image paths, predicting before training, invalid menu choices, and a successful end-to-end run. All critical error paths show a helpful message to the user rather than an unhandled exception.

---

## 8. Reused or Adapted Code

The following code was adapted from the Assignment 3 Full Guidance document:

- The `DatasetIndexer.build_dataframe()` method is adapted from the example in Step 4 of the guidance document. I extended it with a progress counter, better error messages, and a save/load method pair.
- The `ImagePreprocessor.transform()` method follows the same grayscale + resize + normalise + flatten pattern shown in Step 6. I extended it with a `transform_batch()` convenience method.
- The `ClassifierService` structure and `RandomForestClassifier` configuration is based on Step 7. I extended it with balanced class weights, a `predict()` method, and separate methods for saving the report and confusion matrix.
- The `MacroApp` (Tkinter GUI) base structure is adapted from Step 9. I extended it with a dark theme, threaded task execution, a scrolled output log, a progress bar, and a status bar.
- The `ConsoleApp` menu structure is adapted from the Step 9 console example. I extended it with option 5 (file listing) and comprehensive error handling on all options.

All adaptations are acknowledged in the relevant source file comments.

---

## 9. Work Division

| Member | Primary Responsibilities |
|---|---|
| Member 1 | `DatasetIndexer`, `ImageRecord`, `EDAService`, `config.py` |
| Member 2 | `ImagePreprocessor`, `ClassifierService`, `WorkflowService` |
| Member 3 | `ConsoleApp`, `MacroApp` (Tkinter GUI), `MANUAL_TESTING.md`, `README.md`, integration |

All members understood the overall system design and were able to explain any part of it during the Week 13 presentation.
