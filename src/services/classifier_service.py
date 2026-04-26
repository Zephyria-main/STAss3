# src/services/classifier_service.py
# I put all model-related logic here so the interface and workflow layers
# never need to import scikit-learn directly. If I ever swap the classifier
# for a Keras model, only this file changes.

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from src.config import MODEL_FILE_NAME, RANDOM_SEED
from src.services.image_preprocessor import ImagePreprocessor


class ClassifierService:
    """Train, evaluate, and persist the baseline image classifier.

    I use a RandomForestClassifier as the baseline because it is
    interpretable, handles multi-class problems natively, and requires
    no normalisation beyond what the preprocessor already does. It also
    trains quickly enough to demo live in Week 13.
    """

    def __init__(
        self,
        preprocessor: ImagePreprocessor,
        model_output_dir: Path,
    ) -> None:
        self.preprocessor = preprocessor
        self.model_output_dir = model_output_dir
        self.model_output_dir.mkdir(parents=True, exist_ok=True)

        # I use 200 estimators as a good balance between accuracy and
        # training speed. n_jobs=-1 uses all available CPU cores.
        self.model = RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight="balanced",  # I add this to handle class imbalance
        )

        # I track the class labels seen during training so the confusion
        # matrix and classification report stay consistent.
        self._label_names: list = []

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------

    def prepare_features(
        self, dataframe: pd.DataFrame
    ) -> tuple:
        """Convert indexed file paths into a feature matrix and label array.

        I loop over the DataFrame rows and call the preprocessor on each
        image. Rows that fail preprocessing are skipped so a small number
        of corrupt files do not abort the whole training run.

        Args:
            dataframe: The indexed image DataFrame from DatasetIndexer.

        Returns:
            Tuple of (X: np.ndarray shape (n, features),
                      y: np.ndarray shape (n,) of string labels).
        """
        print(f"  [Classifier] Preprocessing {len(dataframe)} images...")
        features = []
        labels = []

        for i, (_, row) in enumerate(dataframe.iterrows()):
            try:
                features.append(self.preprocessor.transform(row["file_path"]))
                labels.append(row["label"])
            except ValueError as error:
                print(f"  [Classifier] Skipping row {i}: {error}")

            # I print progress every 500 images so long runs do not look frozen.
            if (i + 1) % 500 == 0:
                print(f"  [Classifier] Processed {i + 1}/{len(dataframe)} images...")

        print(f"  [Classifier] Feature preparation complete: {len(features)} samples.")
        return np.array(features), np.array(labels)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, dataframe: pd.DataFrame) -> dict:
        """Fit the model and return a dictionary of evaluation outputs.

        I use an 80/20 train/test split with stratification so every class
        is represented proportionally in both sets — this is important for
        imbalanced datasets.

        Args:
            dataframe: The indexed image DataFrame.

        Returns:
            dict with keys: accuracy, report, confusion_matrix, labels.
        """
        X, y = self.prepare_features(dataframe)
        self._label_names = sorted(list(set(y)))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=RANDOM_SEED,
            stratify=y,
        )

        print(
            f"  [Classifier] Training on {len(X_train)} samples, "
            f"testing on {len(X_test)} samples..."
        )
        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, zero_division=0)
        cm = confusion_matrix(y_test, predictions, labels=self._label_names)

        print(f"  [Classifier] Training complete. Test accuracy: {acc:.4f}")

        results = {
            "accuracy": acc,
            "report": report,
            "confusion_matrix": cm,
            "labels": self._label_names,
        }
        return results

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, file_path: str) -> dict:
        """Predict the class of a single image.

        I return both the predicted label and the confidence score so the
        GUI and console app can display meaningful feedback to the user.

        Args:
            file_path: Path to the image file.

        Returns:
            dict with keys: label (str), confidence (float, 0–1).
        """
        features = self.preprocessor.transform(file_path).reshape(1, -1)
        label = self.model.predict(features)[0]

        if hasattr(self.model, "predict_proba"):
            confidence = float(self.model.predict_proba(features).max())
        else:
            confidence = 1.0

        return {"label": label, "confidence": confidence}

    # ------------------------------------------------------------------
    # Save and load
    # ------------------------------------------------------------------

    def save_model(self, file_name: str = MODEL_FILE_NAME) -> Path:
        """Persist the trained model to disk using joblib.

        I save to the configured model output directory so the app and
        console always know where to look for the model file.
        """
        output_path = self.model_output_dir / file_name
        joblib.dump(self.model, output_path)
        print(f"  [Classifier] Model saved to {output_path}")
        return output_path

    def load_model(self, file_name: str = MODEL_FILE_NAME) -> None:
        """Load a previously trained model from disk.

        Raises:
            FileNotFoundError: If the model file does not exist yet.
        """
        model_path = self.model_output_dir / file_name
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                "Please run option 3 (Train classifier) first."
            )
        self.model = joblib.load(model_path)
        print(f"  [Classifier] Model loaded from {model_path}")

    # ------------------------------------------------------------------
    # Evaluation output helpers
    # ------------------------------------------------------------------

    def save_classification_report(self, results: dict) -> None:
        """Write the sklearn classification report to a text file."""
        report_dir = self.model_output_dir.parent / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        output_path = report_dir / "classification_report.txt"

        content = (
            f"Test Accuracy: {results['accuracy']:.4f}\n\n"
            f"{results['report']}"
        )
        output_path.write_text(content, encoding="utf-8")
        print(f"  [Classifier] Classification report saved to {output_path}")

    def save_confusion_matrix_plot(self, results: dict) -> None:
        """Save a heatmap of the confusion matrix as a PNG.

        I skip annotation on large matrices (>15 classes) because the
        numbers overlap and become unreadable.
        """
        labels = results.get("labels", self._label_names)
        cm = results["confusion_matrix"]
        annotate = len(labels) <= 15

        plt.figure(figsize=(max(8, len(labels)), max(6, len(labels) - 1)))
        sns.heatmap(
            cm,
            annot=annotate,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title("Confusion Matrix — Baseline Classifier", fontsize=13)
        plt.xlabel("Predicted Class")
        plt.ylabel("Actual Class")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        report_dir = self.model_output_dir.parent / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        output_path = report_dir / "confusion_matrix.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Classifier] Confusion matrix saved to {output_path}")
