# src/console_app.py
# I designed the console app to be the primary Stage 3 deployment option.
# It wraps WorkflowService in a clean menu loop so every option is reachable
# from one place and the program always returns to the main menu cleanly.

import os
from pathlib import Path

from src.services.workflow_service import WorkflowService


class ConsoleApp:
    """Menu-driven console application for the full project workflow.

    I structured this as a class rather than a script so it is easier to
    test, extend, and call from the entry point without globals. Each menu
    option maps to exactly one WorkflowService method so the logic stays
    in the right layer.
    """

    def __init__(self, workflow_service: WorkflowService) -> None:
        self.workflow_service = workflow_service

    # ------------------------------------------------------------------
    # Main menu loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the menu loop and run until the user chooses Exit.

        I keep the loop in this method so the entry point is clean. The
        loop handles invalid input gracefully and never crashes on a bad
        menu choice.
        """
        self._print_banner()

        while True:
            self._print_menu()
            choice = input("  Enter your choice: ").strip()

            if choice == "1":
                self._show_summary()
            elif choice == "2":
                self._generate_eda()
            elif choice == "3":
                self._train_model()
            elif choice == "4":
                self._predict_image()
            elif choice == "5":
                self._view_saved_outputs()
            elif choice == "6":
                print("\n  Exiting. Thank you for using the Macroinvertebrate Analysis System.\n")
                break
            else:
                print("\n  [!] Invalid option. Please enter a number between 1 and 6.\n")

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _print_banner(self) -> None:
        """Print the application title banner on startup."""
        print("\n" + "=" * 58)
        print("    MACROINVERTEBRATE IMAGE ANALYSIS SYSTEM")
        print("    Software Technology 1 — Group Assignment 3")
        print("=" * 58 + "\n")

    def _print_menu(self) -> None:
        """Print the main menu options."""
        print("-" * 40)
        print("  MAIN MENU")
        print("-" * 40)
        print("  1. Show dataset summary")
        print("  2. Generate EDA outputs (Stage 1)")
        print("  3. Train baseline classifier (Stage 2)")
        print("  4. Predict image class (Stage 3)")
        print("  5. View saved output files")
        print("  6. Exit")
        print("-" * 40)

    # ------------------------------------------------------------------
    # Menu action handlers
    # ------------------------------------------------------------------

    def _show_summary(self) -> None:
        """Handle menu option 1: display dataset summary statistics."""
        print("\n[Option 1] Loading dataset summary...\n")
        try:
            self.workflow_service.show_summary()
        except Exception as error:
            print(f"  [Error] Could not load summary: {error}")

    def _generate_eda(self) -> None:
        """Handle menu option 2: generate and save all EDA charts."""
        print("\n[Option 2] Running Stage 1 EDA...\n")
        try:
            self.workflow_service.generate_eda()
            print("\n  All EDA charts have been saved to outputs/eda/")
        except Exception as error:
            print(f"  [Error] EDA failed: {error}")

    def _train_model(self) -> None:
        """Handle menu option 3: train the classifier and save the model."""
        print("\n[Option 3] Training baseline classifier...\n")
        print("  This may take several minutes depending on dataset size.")
        print("  Please wait...\n")
        try:
            results = self.workflow_service.train_model()
            if results:
                print(f"\n  Test Accuracy: {results['accuracy']:.4f}")
                print(
                    "\n  Model and evaluation reports saved to outputs/models/ "
                    "and outputs/reports/"
                )
        except Exception as error:
            print(f"  [Error] Training failed: {error}")

    def _predict_image(self) -> None:
        """Handle menu option 4: predict the class of a user-specified image.

        I wrap the prediction in a try/except block so a bad file path or
        missing model gives the user a helpful message instead of a crash.
        """
        print("\n[Option 4] Image Prediction\n")
        image_path = input("  Enter the full path to an image file: ").strip()

        # I strip surrounding quotes in case the user drag-dropped the file
        # into the terminal, which often adds them on macOS and Windows.
        image_path = image_path.strip("\"'")

        if not image_path:
            print("  [!] No path entered. Returning to menu.")
            return

        if not Path(image_path).exists():
            print(f"  [!] File not found: {image_path}")
            print("  Please check the path and try again.")
            return

        try:
            result = self.workflow_service.predict_image(image_path)
            print("\n  " + "=" * 40)
            print(f"  Predicted Class : {result['label']}")
            print(f"  Confidence      : {result['confidence']:.2%}")
            print("  " + "=" * 40 + "\n")
        except FileNotFoundError as error:
            print(f"\n  [!] {error}")
        except ValueError as error:
            print(f"\n  [!] Could not read image: {error}")
        except Exception as error:
            print(f"\n  [!] Unexpected error: {error}")

    def _view_saved_outputs(self) -> None:
        """Handle menu option 5: list all saved output files.

        I list files from the outputs/ folder so the user can confirm
        which charts and reports have been generated without leaving
        the application.
        """
        print("\n[Option 5] Saved Output Files\n")
        outputs_dir = Path(__file__).resolve().parent.parent / "outputs"

        if not outputs_dir.exists():
            print("  No outputs folder found yet. Run EDA or training first.")
            return

        found_any = False
        for subdir in sorted(outputs_dir.iterdir()):
            if subdir.is_dir():
                files = sorted(subdir.iterdir())
                if files:
                    print(f"  {subdir.name}/")
                    for f in files:
                        size_kb = f.stat().st_size / 1024
                        print(f"    {f.name:<45} {size_kb:>7.1f} KB")
                    found_any = True

        if not found_any:
            print("  No output files found yet. Run EDA or training first.")

        print()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    """Start the menu-driven console application."""
    workflow = WorkflowService()
    app = ConsoleApp(workflow)
    app.run()


if __name__ == "__main__":
    main()
