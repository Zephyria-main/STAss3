Implementation Summary

Project Title: Macroinvertebrate Image Analysis System  
Unit: Software Technology 1 (4483/8995) — Assignment 3  
Group Members: Kes Jayaweera · [Member 2 Name] · [Member 3 Name]

---

1. Project Goal

We built a modular Python application that analyses freshwater macroinvertebrate images across three stages: exploratory data analysis (Stage 1), baseline image classification (Stage 2), and interactive deployment via both a console menu and a Tkinter GUI (Stage 3). The system can index a local image dataset, generate EDA charts and statistics, train a Random Forest classifier, and allow a user to predict the class of a new image.

---

2. System Design Overview

I designed the system as a service-oriented Python application with clear separation between data management, analysis, machine learning, and user interaction. Each class has one clear responsibility and dependencies are injected at construction time so the classes are easy to test and replace independently.

The WorkflowService acts as the coordinator. Both the ConsoleApp and MacroApp classes call the WorkflowService instead of the individual services directly. This means the logic is never duplicated between the two interface types.

---

3. Class and Module Overview

| Class / Module | Location | Responsibility |
|---|---|---|
| config.py | src/config.py | Central path and constant configuration |
| ImageRecord | src/models/records.py | Dataclass for one indexed image's metadata |
| DatasetIndexer | src/services/dataset_indexer.py | Scans dataset folders recursively, builds DataFrame |
| EDAService | src/services/eda_service.py | Generates Stage 1 charts, summaries, and reports |
| ImagePreprocessor | src/services/image_preprocessor.py | Converts images to normalised feature vectors |
| ClassifierService | src/services/classifier_service.py | Trains, evaluates, saves, and loads the classifier |
| WorkflowService | src/services/workflow_service.py | Coordinates all pipeline steps for both interfaces |
| ConsoleApp | src/console_app.py | Menu-driven Stage 3 console interface |
| MacroApp | src/app.py | Tkinter Stage 3 GUI with image preview |

---

4. Python Packages Used and Why

- Pandas — I used Pandas to store the image index as a structured DataFrame so EDA and training both work from the same table.
- NumPy — I used NumPy for the feature matrix and label arrays that the classifier requires.
- OpenCV (cv2) — I used OpenCV to read image files from disk, convert to grayscale, and resize to a consistent dimension.
- Matplotlib and Seaborn — I used these for all EDA visualisations including bar charts, histograms, scatter plots, and the confusion matrix heatmap.
- Scikit-learn — I used scikit-learn for the train/test split, RandomForestClassifier, and evaluation metrics.
- joblib — I used joblib to save and load the trained model so prediction can happen in a separate session.
- Pillow — I used Pillow to load and thumbnail images for preview inside the Tkinter GUI.
- tkinter — I used the built-in Tkinter library for the Stage 3 desktop GUI.
- pathlib — I used pathlib throughout for readable, cross-platform file path handling.

---

5. Key Features Implemented

- Recursive dataset scanning with class label extraction from folder names
- Five EDA charts: class distribution, image size histograms, sample grid, channel distribution, aspect ratio scatter
- EDA summary statistics written to a text report
- Grayscale + resize + normalise + flatten image preprocessing pipeline
- RandomForestClassifier with stratified 80/20 train/test split and balanced class weights
- Confusion matrix heatmap and classification report saved to outputs/reports/
- Model persistence via joblib
- Console menu with six options including error handling for missing model, bad paths, and invalid input
- Tkinter GUI with image preview, threaded background tasks, progress indicator, and output log
- Non-interactive main.py pipeline runner for batch operation

---

6. What the EDA Revealed

The class distribution chart confirmed significant imbalance across macroinvertebrate species — some classes have many more images than others. This informed our decision to use class_weight="balanced" in the RandomForestClassifier. The image size distribution showed that most images are not a consistent size, which confirmed that resizing to 128×128 was the right preprocessing step.

---

7. Testing Summary

I tested the application manually across ten scenarios documented in MANUAL_TESTING.md. These included missing dataset folder, invalid image paths, predicting before training, invalid menu choices, and a successful end-to-end run. All critical error paths show a helpful message to the user rather than an unhandled exception.

---

8. Reused or Adapted Code

Per unit instructions: reused or adapted tutorial/lab or guidance code is acknowledged in source comments (module-level blocks cross-reference Step numbers), summarised here, and each entry below states how the code was modified, extended, or integrated into this submission.

Primary written source: Assignment 3 Full Guidance and Coding Examples (Software Technology 1, 4483/8995). Weekly lab notebooks and practical examples (OpenCV, sklearn, Tkinter patterns) informed API usage consistent with that guidance.

- config.py — Based on Step 2 (path constants and IMAGE_SIZE). Extended with PROCESSED_DATA_DIR, REPORT_OUTPUT_DIR, MODEL_FILE_NAME, and RANDOM_SEED; paths aligned to outputs/eda, outputs/models, outputs/reports.
- ImageRecord (records.py) — Based on Step 3 dataclass. Integrated as the typed row concept; file_path uses pathlib.Path for clarity.
- DatasetIndexer.build_dataframe() — Adapted from Step 4. Extended with skip warnings for corrupt files, empty-table guard, scan summary printout, and save_index()/load_index() CSV helpers for repeat runs without rescanning.
- EDAService — Adapted from Step 5 (class chart, size histograms, summary) plus the guidance sample-grid pattern. Extended with styled countplot and bar annotations, KDE on size plots, channel and aspect-ratio charts, richer summary dict, save_summary_text(), and run_all().
- ImagePreprocessor.transform() — Based on Step 6 pipeline. Extended with clear errors on failed reads, config-driven IMAGE_SIZE, and transform_batch() with skip logging.
- ClassifierService — Adapted from Steps 7–8 (training loop, metrics, joblib save, report and confusion matrix outputs). Integrated class_weight="balanced", stratified split with explicit labels for the matrix, prepare_features skip behaviour, predict() returning confidence for Stage 3, and report files under outputs/reports.
- WorkflowService — Adapted from Step 10 coordinator pattern. Integrated cached dataframe, generate_eda via EDAService.run_all(), train_model wiring report and matrix saves, predict_image using ClassifierService.predict(), shared use by main.py, console_app, and app.
- main.py — Based on Step 10 thin entry point. Extended by placing the batch runner at the repository root as main.py per submission expectations.
- MacroApp (app.py) — Adapted from Step 9 Tkinter example (dialogs, PIL preview, prediction). Integrated WorkflowService for all stages; added threading, UI layout, logging, and status feedback.
- ConsoleApp — Adapted from Step 9 menu loop. Integrated six options including viewing saved outputs, error handling, and exclusive delegation to WorkflowService.

Full citations, URLs, and licence notes are in CODE_SOURCES_AND_LICENSES.md. Inline acknowledgements appear under “Unit tutorial / guidance” in each listed module.

---

9. Work Division

| Member | Primary Responsibilities |
|---|---|
| Member 1 | DatasetIndexer, ImageRecord, EDAService, config.py |
| Member 2 | ImagePreprocessor, ClassifierService, WorkflowService |
| Member 3 | ConsoleApp, MacroApp (Tkinter GUI), MANUAL_TESTING.md, README.md, integration |

All members understood the overall system design and were able to explain any part of it during the Week 13 presentation.
