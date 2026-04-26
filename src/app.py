# src/app.py
# I built the Tkinter GUI as an alternative Stage 3 deployment option.
# The design is intentionally simple so it is easy to explain during the
# Week 13 presentation. Every button maps to one WorkflowService method.

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from PIL import Image, ImageTk

from src.services.workflow_service import WorkflowService


class MacroApp(tk.Tk):
    """Desktop GUI for macroinvertebrate image analysis and prediction.

    I extend tk.Tk directly so the application window IS the app — no
    extra outer frame is needed. The WorkflowService handles all data
    and model logic so this class only manages the interface.
    """

    def __init__(self, workflow_service: WorkflowService) -> None:
        super().__init__()
        self.workflow_service = workflow_service
        self.selected_file: str | None = None

        # I configure the window appearance here so it is consistent
        # across all platforms.
        self.title("Macroinvertebrate Image Analysis System")
        self.geometry("1000x700")
        self.resizable(True, True)
        self.configure(bg="#1e1e2e")

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build and lay out all widgets in the main window.

        I split the window into a left control panel and a right results
        panel so the layout scales well when the window is resized.
        """
        # --- Header ---
        header = tk.Frame(self, bg="#313244", pady=10)
        header.pack(fill="x")
        tk.Label(
            header,
            text="Macroinvertebrate Image Analysis System",
            font=("Helvetica", 16, "bold"),
            fg="#cdd6f4",
            bg="#313244",
        ).pack()
        tk.Label(
            header,
            text="Software Technology 1 — Assignment 3",
            font=("Helvetica", 10),
            fg="#a6adc8",
            bg="#313244",
        ).pack()

        # --- Main content area ---
        content = tk.Frame(self, bg="#1e1e2e")
        content.pack(fill="both", expand=True, padx=10, pady=10)

        # Left panel: controls
        left = tk.Frame(content, bg="#313244", width=260)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)
        self._build_left_panel(left)

        # Right panel: image preview + output log
        right = tk.Frame(content, bg="#1e1e2e")
        right.pack(side="left", fill="both", expand=True)
        self._build_right_panel(right)

        # --- Status bar ---
        self.status_var = tk.StringVar(value="Ready.")
        status_bar = tk.Label(
            self,
            textvariable=self.status_var,
            font=("Helvetica", 9),
            fg="#a6adc8",
            bg="#181825",
            anchor="w",
            padx=8,
        )
        status_bar.pack(fill="x", side="bottom")

    def _build_left_panel(self, parent: tk.Frame) -> None:
        """Build the left-side control panel with stage buttons."""
        tk.Label(
            parent,
            text="ACTIONS",
            font=("Helvetica", 11, "bold"),
            fg="#cba6f7",
            bg="#313244",
            pady=12,
        ).pack(fill="x")

        # I define buttons as (label, command) pairs so adding new
        # options later is a one-liner.
        buttons = [
            ("1. Dataset Summary", self._run_summary),
            ("2. Generate EDA (Stage 1)", self._run_eda),
            ("3. Train Classifier (Stage 2)", self._run_training),
            ("─────────────────", None),
            ("4. Choose Image", self._choose_image),
            ("5. Predict Class (Stage 3)", self._predict),
            ("─────────────────", None),
            ("Clear Log", self._clear_log),
        ]

        for text, cmd in buttons:
            if cmd is None:
                tk.Label(
                    parent, text=text, font=("Helvetica", 9),
                    fg="#585b70", bg="#313244",
                ).pack(pady=2)
            else:
                btn = tk.Button(
                    parent,
                    text=text,
                    command=cmd,
                    font=("Helvetica", 10),
                    bg="#45475a",
                    fg="#cdd6f4",
                    activebackground="#585b70",
                    activeforeground="#ffffff",
                    relief="flat",
                    cursor="hand2",
                    padx=10,
                    pady=6,
                    width=22,
                )
                btn.pack(pady=3, padx=10, fill="x")

        # I add a progress bar that I can show during long operations.
        self.progress = ttk.Progressbar(parent, mode="indeterminate")
        self.progress.pack(fill="x", padx=10, pady=(20, 5))

    def _build_right_panel(self, parent: tk.Frame) -> None:
        """Build the right panel with image preview and output log."""
        # --- Image preview area ---
        preview_frame = tk.LabelFrame(
            parent,
            text=" Image Preview ",
            font=("Helvetica", 10),
            fg="#cba6f7",
            bg="#1e1e2e",
            pady=5,
        )
        preview_frame.pack(fill="x", pady=(0, 8))

        self.image_label = tk.Label(
            preview_frame,
            text="No image selected.\nUse 'Choose Image' to load one.",
            font=("Helvetica", 10),
            fg="#6c7086",
            bg="#1e1e2e",
            height=12,
        )
        self.image_label.pack(pady=5)

        # --- Prediction result display ---
        self.result_var = tk.StringVar(value="")
        self.result_label = tk.Label(
            parent,
            textvariable=self.result_var,
            font=("Helvetica", 13, "bold"),
            fg="#a6e3a1",
            bg="#1e1e2e",
        )
        self.result_label.pack()

        # --- Output log ---
        log_frame = tk.LabelFrame(
            parent,
            text=" Output Log ",
            font=("Helvetica", 10),
            fg="#cba6f7",
            bg="#1e1e2e",
        )
        log_frame.pack(fill="both", expand=True)

        self.log_box = scrolledtext.ScrolledText(
            log_frame,
            font=("Courier", 9),
            bg="#181825",
            fg="#cdd6f4",
            insertbackground="#cdd6f4",
            relief="flat",
            wrap="word",
            state="disabled",
        )
        self.log_box.pack(fill="both", expand=True, padx=5, pady=5)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        """Append a message to the output log widget."""
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _clear_log(self) -> None:
        """Clear all text from the output log."""
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")
        self.result_var.set("")

    def _set_status(self, text: str) -> None:
        """Update the status bar at the bottom of the window."""
        self.status_var.set(text)
        self.update_idletasks()

    # ------------------------------------------------------------------
    # Button action handlers
    # ------------------------------------------------------------------

    def _run_summary(self) -> None:
        """Load and display the dataset summary statistics."""
        self._log("\n[Summary] Loading dataset...")
        self._set_status("Loading dataset summary...")

        def task():
            try:
                summary = self.workflow_service.show_summary()
                self._log("[Summary] Dataset Summary:")
                for key, val in summary.items():
                    self._log(f"  {key:<38} {val}")
                self._log("[Summary] Done.\n")
                self._set_status("Summary loaded.")
            except Exception as error:
                self._log(f"[Error] {error}")
                self._set_status("Error loading summary.")

        threading.Thread(target=task, daemon=True).start()

    def _run_eda(self) -> None:
        """Run Stage 1 EDA in a background thread to keep the UI responsive."""
        self._log("\n[EDA] Starting Stage 1 EDA. This may take a moment...")
        self._set_status("Running EDA...")
        self.progress.start(10)

        def task():
            try:
                self.workflow_service.generate_eda()
                self._log("[EDA] All charts saved to outputs/eda/")
                self._log("[EDA] Stage 1 complete.\n")
                self._set_status("EDA complete.")
            except Exception as error:
                self._log(f"[Error] {error}")
                self._set_status("EDA failed.")
            finally:
                self.progress.stop()

        threading.Thread(target=task, daemon=True).start()

    def _run_training(self) -> None:
        """Train the classifier in a background thread."""
        self._log("\n[Training] Starting Stage 2 classifier training...")
        self._log("[Training] This may take several minutes. Please wait...")
        self._set_status("Training classifier...")
        self.progress.start(10)

        def task():
            try:
                results = self.workflow_service.train_model()
                if results:
                    self._log(f"[Training] Test Accuracy: {results['accuracy']:.4f}")
                    self._log("[Training] Model and reports saved to outputs/")
                    self._log("[Training] Stage 2 complete.\n")
                    self._set_status(f"Training complete. Accuracy: {results['accuracy']:.4f}")
            except Exception as error:
                self._log(f"[Error] {error}")
                self._set_status("Training failed.")
            finally:
                self.progress.stop()

        threading.Thread(target=task, daemon=True).start()

    def _choose_image(self) -> None:
        """Open a file dialog and preview the selected image."""
        file_path = filedialog.askopenfilename(
            title="Select an image for prediction",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
                ("All Files", "*.*"),
            ],
        )
        if not file_path:
            return

        self.selected_file = file_path
        self._log(f"\n[Image] Selected: {Path(file_path).name}")
        self.result_var.set("")

        # I load and display a thumbnail so the user can confirm they
        # chose the right file before running prediction.
        try:
            image = Image.open(file_path)
            image.thumbnail((300, 280))
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # I keep a reference to stop GC
            self._set_status(f"Image loaded: {Path(file_path).name}")
        except Exception as error:
            self._log(f"[Error] Could not preview image: {error}")

    def _predict(self) -> None:
        """Predict the class of the currently selected image."""
        if not self.selected_file:
            messagebox.showwarning(
                "No Image Selected",
                "Please choose an image first using 'Choose Image'.",
            )
            return

        self._log(f"\n[Predict] Running prediction on {Path(self.selected_file).name}...")
        self._set_status("Predicting...")
        self.progress.start(10)

        def task():
            try:
                result = self.workflow_service.predict_image(self.selected_file)
                label = result["label"]
                confidence = result["confidence"]
                self.result_var.set(
                    f"Predicted: {label}   |   Confidence: {confidence:.2%}"
                )
                self._log(f"[Predict] Class      : {label}")
                self._log(f"[Predict] Confidence : {confidence:.2%}")
                self._log("[Predict] Prediction complete.\n")
                self._set_status(f"Predicted: {label} ({confidence:.2%})")
            except FileNotFoundError as error:
                messagebox.showerror(
                    "No Model Found",
                    str(error) + "\n\nPlease train the classifier first (Option 3).",
                )
                self._log(f"[Error] {error}")
                self._set_status("Prediction failed — no model.")
            except ValueError as error:
                self._log(f"[Error] {error}")
                self._set_status("Prediction failed — bad image.")
            except Exception as error:
                self._log(f"[Error] Unexpected: {error}")
                self._set_status("Prediction failed.")
            finally:
                self.progress.stop()

        threading.Thread(target=task, daemon=True).start()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    """Launch the Tkinter GUI application."""
    workflow = WorkflowService()
    app = MacroApp(workflow)
    app.mainloop()


if __name__ == "__main__":
    main()
