# src/services/eda_service.py
# I put all the chart-generation logic inside this class so Stage 1 is
# completely self-contained. The WorkflowService just calls these methods
# and the outputs land in the outputs/eda/ folder automatically.

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class EDAService:
    """Generate and save EDA outputs for the indexed image dataset.

    I separate EDA from indexing because a good design means each class
    has one clear job. This class only knows how to make charts and
    summaries — it never touches the file system scanner or the model.
    """

    def __init__(self, dataframe: pd.DataFrame, output_dir: Path) -> None:
        # I store the dataframe and output directory at construction time
        # so every method can access them without extra arguments.
        self.dataframe = dataframe
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Chart generation methods
    # ------------------------------------------------------------------

    def save_class_distribution(self) -> None:
        """Save a bar chart showing the number of images per class.

        I sort by count so the chart is easy to read and markers can
        immediately spot any class imbalance in the dataset.
        """
        plt.figure(figsize=(14, 6))
        order = self.dataframe["label"].value_counts().index
        ax = sns.countplot(
            data=self.dataframe,
            x="label",
            order=order,
            palette="viridis",
        )
        ax.set_title("Macroinvertebrate Images per Class", fontsize=14, pad=12)
        ax.set_xlabel("Class", fontsize=11)
        ax.set_ylabel("Image Count", fontsize=11)
        plt.xticks(rotation=45, ha="right", fontsize=9)

        # I annotate each bar with the count so the chart is readable
        # without hovering over it.
        for bar in ax.patches:
            ax.annotate(
                f"{int(bar.get_height())}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        output_path = self.output_dir / "class_distribution.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  [EDA] Class distribution chart saved to {output_path}")

    def save_image_size_distribution(self) -> None:
        """Save side-by-side histograms for image width and height.

        I include this because inconsistent image sizes are a common issue
        in Kaggle datasets and they affect our preprocessing decisions.
        """
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Image Dimension Distributions", fontsize=14)

        sns.histplot(
            self.dataframe["width"], bins=30, kde=True,
            ax=axes[0], color="steelblue",
        )
        axes[0].set_title("Image Width")
        axes[0].set_xlabel("Pixels")
        axes[0].set_ylabel("Count")

        sns.histplot(
            self.dataframe["height"], bins=30, kde=True,
            ax=axes[1], color="coral",
        )
        axes[1].set_title("Image Height")
        axes[1].set_xlabel("Pixels")
        axes[1].set_ylabel("Count")

        plt.tight_layout()
        output_path = self.output_dir / "image_size_distribution.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  [EDA] Image size distribution chart saved to {output_path}")

    def save_sample_grid(self, sample_count: int = 9) -> None:
        """Save a 3x3 grid of randomly sampled images with their labels.

        I use a fixed random seed so the grid is reproducible. This helps
        during the presentation because we can show the same examples
        every time we run the script.
        """
        sample_df = self.dataframe.sample(
            min(sample_count, len(self.dataframe)), random_state=42
        )

        cols = 3
        rows = (len(sample_df) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(11, 11))
        fig.suptitle("Sample Images from Dataset", fontsize=14, y=1.01)

        for ax, (_, row) in zip(axes.flat, sample_df.iterrows()):
            image = cv2.imread(row["file_path"])
            if image is None:
                ax.axis("off")
                continue
            # I convert from BGR (OpenCV default) to RGB for Matplotlib.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image)
            ax.set_title(row["label"], fontsize=9)
            ax.axis("off")

        # I turn off any unused grid cells if the sample count is not
        # a perfect multiple of 3.
        for ax in axes.flat[len(sample_df):]:
            ax.axis("off")

        plt.tight_layout()
        output_path = self.output_dir / "sample_grid.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [EDA] Sample grid saved to {output_path}")

    def save_channel_distribution(self) -> None:
        """Save a pie chart showing how many images are colour vs grayscale.

        I add this chart because a mix of grayscale and colour images
        is something we need to handle in preprocessing.
        """
        channel_counts = self.dataframe["channels"].value_counts()
        labels = {1: "Grayscale (1ch)", 3: "Colour (3ch)"}
        display_labels = [labels.get(c, f"{c} channels") for c in channel_counts.index]

        plt.figure(figsize=(6, 6))
        plt.pie(
            channel_counts.values,
            labels=display_labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=["#6baed6", "#fd8d3c"],
        )
        plt.title("Image Channel Distribution", fontsize=13)
        plt.tight_layout()
        output_path = self.output_dir / "channel_distribution.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  [EDA] Channel distribution chart saved to {output_path}")

    def save_aspect_ratio_scatter(self) -> None:
        """Save a scatter plot of width vs height coloured by class.

        I find this useful for seeing whether any class has an unusual
        aspect ratio that might affect our resizing strategy.
        """
        plt.figure(figsize=(10, 7))
        classes = self.dataframe["label"].unique()
        palette = sns.color_palette("tab20", n_colors=len(classes))

        for colour, cls in zip(palette, classes):
            subset = self.dataframe[self.dataframe["label"] == cls]
            plt.scatter(
                subset["width"],
                subset["height"],
                label=cls,
                color=colour,
                alpha=0.5,
                s=15,
            )

        plt.title("Image Width vs Height by Class", fontsize=13)
        plt.xlabel("Width (px)")
        plt.ylabel("Height (px)")
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=7,
            ncol=1,
        )
        plt.tight_layout()
        output_path = self.output_dir / "aspect_ratio_scatter.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [EDA] Aspect ratio scatter plot saved to {output_path}")

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def build_summary(self) -> dict:
        """Return key dataset summary statistics as a dictionary.

        I return a plain dict rather than printing directly so the caller
        (WorkflowService or the console menu) can decide how to display it.
        """
        class_counts = self.dataframe["label"].value_counts()
        summary = {
            "total_images": int(len(self.dataframe)),
            "total_classes": int(self.dataframe["label"].nunique()),
            "mean_width_px": round(float(self.dataframe["width"].mean()), 1),
            "mean_height_px": round(float(self.dataframe["height"].mean()), 1),
            "min_width_px": int(self.dataframe["width"].min()),
            "max_width_px": int(self.dataframe["width"].max()),
            "most_common_class": str(class_counts.idxmax()),
            "most_common_class_count": int(class_counts.max()),
            "least_common_class": str(class_counts.idxmin()),
            "least_common_class_count": int(class_counts.min()),
        }
        return summary

    def save_summary_text(self) -> None:
        """Write the summary statistics to a human-readable text file.

        I save this to the reports folder so we have a quick reference
        without needing to re-run the EDA every time.
        """
        summary = self.build_summary()
        report_dir = self.output_dir.parent.parent / "outputs" / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        output_path = report_dir / "eda_summary.txt"

        lines = ["=" * 50, "  EDA SUMMARY — Macroinvertebrate Dataset", "=" * 50]
        for key, value in summary.items():
            lines.append(f"  {key:<35} {value}")
        lines.append("=" * 50)

        output_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  [EDA] Summary text saved to {output_path}")

    def run_all(self) -> dict:
        """Run every EDA output in sequence and return the summary dict.

        I provide this convenience method so the WorkflowService can
        trigger the full Stage 1 pipeline in a single call.
        """
        print("\n[EDA] Generating all Stage 1 outputs...")
        self.save_class_distribution()
        self.save_image_size_distribution()
        self.save_sample_grid()
        self.save_channel_distribution()
        self.save_aspect_ratio_scatter()
        self.save_summary_text()
        summary = self.build_summary()
        print("[EDA] Stage 1 complete.\n")
        return summary
