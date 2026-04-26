# src/services/workflow_service.py
# I created WorkflowService as the single coordinator for the whole project.
# Both the console app and the GUI app call this service instead of importing
# the individual services directly. This means I only need to change one place
# if the pipeline steps ever change.

from pathlib import Path

import pandas as pd

from src.config import EDA_OUTPUT_DIR, MODEL_OUTPUT_DIR, OUTPUTS_DIR
from src.services.classifier_service import ClassifierService
from src.services.dataset_indexer import DatasetIndexer
from src.services.eda_service import EDAService
from src.services.image_preprocessor import ImagePreprocessor


class WorkflowService:
    """Coordinate the shared pipeline used by the console and GUI apps.

    I initialise all the individual service classes here so they are
    constructed once and reused across multiple method calls. The
    dataframe is cached after the first load to avoid re-scanning the
    disk on every menu option.
    """

    def __init__(self) -> None:
        # I create the output directories early so any method can write
        # files without needing to check if the folder exists.
        EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUTS_DIR / "reports").mkdir(parents=True, exist_ok=True)

        # I build each service as an instance variable so they share the
        # same preprocessor and model across methods.
        self.indexer = DatasetIndexer()
        self.preprocessor = ImagePreprocessor()
        self.classifier = ClassifierService(self.preprocessor, MODEL_OUTPUT_DIR)

        # I cache the dataframe here so we only scan the dataset once per
        # session — scanning a large folder every menu selection would be slow.
        self._dataframe: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def load_dataframe(self) -> pd.DataFrame:
        """Return the indexed dataset, scanning the disk if needed.

        I use the cached version if it exists so the first load is the
        only time we pay the full scan cost.
        """
        if self._dataframe is None or self._dataframe.empty:
            print("[Workflow] Scanning dataset folder...")
            self._dataframe = self.indexer.build_dataframe()
        return self._dataframe

    # ------------------------------------------------------------------
    # Stage 1 — EDA
    # ------------------------------------------------------------------

    def show_summary(self) -> dict:
        """Build and return the dataset summary statistics.

        I print the summary here as well as returning it so the console
        app does not need to format output itself.
        """
        dataframe = self.load_dataframe()
        if dataframe.empty:
            print("[Workflow] No data available for summary.")
            return {}

        eda = EDAService(dataframe, EDA_OUTPUT_DIR)
        summary = eda.build_summary()

        print("\n" + "=" * 48)
        print("  DATASET SUMMARY")
        print("=" * 48)
        for key, value in summary.items():
            print(f"  {key:<38} {value}")
        print("=" * 48 + "\n")
        return summary

    def generate_eda(self) -> None:
        """Run all Stage 1 EDA outputs and save them to outputs/eda/."""
        dataframe = self.load_dataframe()
        if dataframe.empty:
            print("[Workflow] No data to analyse. Add images to data/raw/ first.")
            return

        eda = EDAService(dataframe, EDA_OUTPUT_DIR)
        eda.run_all()
        print(f"[Workflow] EDA outputs saved to {EDA_OUTPUT_DIR}")

    # ------------------------------------------------------------------
    # Stage 2 — Classification
    # ------------------------------------------------------------------

    def train_model(self) -> dict:
        """Train the classifier and save it plus evaluation reports.

        I call both save methods here so the user always gets the
        confusion matrix and classification report automatically.
        """
        dataframe = self.load_dataframe()
        if dataframe.empty:
            print("[Workflow] No data available for training.")
            return {}

        print("[Workflow] Starting model training...")
        results = self.classifier.train(dataframe)
        self.classifier.save_model()
        self.classifier.save_classification_report(results)
        self.classifier.save_confusion_matrix_plot(results)

        print(f"\n[Workflow] Training complete.")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        return results

    # ------------------------------------------------------------------
    # Stage 3 — Prediction
    # ------------------------------------------------------------------

    def predict_image(self, file_path: str) -> dict:
        """Predict the macroinvertebrate class for one image.

        I load the saved model on first use rather than requiring it to
        be trained in the same session — this is more realistic for a
        deployed app.

        Args:
            file_path: String path to the image file.

        Returns:
            dict with keys: label (str), confidence (float).

        Raises:
            FileNotFoundError: If no trained model exists yet.
            ValueError: If the image cannot be read.
        """
        self.classifier.load_model()
        result = self.classifier.predict(file_path)
        print(
            f"[Workflow] Predicted: {result['label']} "
            f"(confidence: {result['confidence']:.2%})"
        )
        return result

    # ------------------------------------------------------------------
    # Convenience pipeline for main.py
    # ------------------------------------------------------------------

    def run_full_pipeline(self) -> None:
        """Run Stage 1 EDA and Stage 2 training in sequence.

        I provide this for the non-interactive main.py entry point so
        a single command generates all outputs without any menu prompts.
        """
        print("\n=== Macroinvertebrate Image Analysis System ===\n")
        self.show_summary()
        self.generate_eda()
        results = self.train_model()
        if results:
            print("\n[Pipeline] Classification Report:")
            print(results.get("report", "No report available."))
        print("\n[Pipeline] Full pipeline complete. Check the outputs/ folder.\n")
