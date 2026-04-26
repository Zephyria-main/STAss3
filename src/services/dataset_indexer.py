# src/services/dataset_indexer.py
# I wrote this class to handle all the file-scanning logic in one place.
# The rest of the application just calls build_dataframe() and gets a
# clean pandas DataFrame back — no file path logic leaks anywhere else.

from pathlib import Path

import cv2
import pandas as pd

from src.config import RAW_DATA_DIR, SUPPORTED_EXTENSIONS


class DatasetIndexer:
    """Scan the dataset folder and build a tabular image index.

    I walk the directory tree recursively, read each image with OpenCV
    to get its dimensions, and use the parent folder name as the class
    label. This makes the indexer robust to different subfolder depths
    as long as the immediate parent of each image is the class name.
    """

    def __init__(self, data_dir: Path = RAW_DATA_DIR) -> None:
        # I accept a custom path so the indexer is easy to test with a
        # small fixture dataset without touching the real data folder.
        self.data_dir = data_dir

    def build_dataframe(self) -> pd.DataFrame:
        """Return one row per image with file path, label, and dimensions.

        I use rglob to find images at any depth under data_dir. Each row
        stores the string path, class label, width, height, and channel
        count so EDA and training can use the same table.

        Returns:
            pd.DataFrame: Columns are file_path, label, width, height,
                          channels. Returns an empty DataFrame if no
                          valid images are found.
        """
        records = []

        for file_path in self.data_dir.rglob("*"):
            # I skip anything that is not a supported image extension.
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            # I use OpenCV to read the image and get real pixel dimensions.
            # cv2.imread returns None if the file is corrupt or unreadable.
            image = cv2.imread(str(file_path))
            if image is None:
                print(f"  [Warning] Could not read image, skipping: {file_path.name}")
                continue

            height, width = image.shape[:2]
            # I check for a third dimension because grayscale images only
            # have shape (H, W), not (H, W, C).
            channels = image.shape[2] if len(image.shape) == 3 else 1

            # I use the immediate parent folder name as the class label.
            # This convention matches the Kaggle dataset structure where
            # each class is its own subfolder.
            label = file_path.parent.name

            records.append(
                {
                    "file_path": str(file_path),
                    "label": label,
                    "width": width,
                    "height": height,
                    "channels": channels,
                }
            )

        if not records:
            print(
                f"  [Warning] No images found in {self.data_dir}. "
                "Make sure the dataset is placed inside data/raw/."
            )
            return pd.DataFrame(
                columns=["file_path", "label", "width", "height", "channels"]
            )

        df = pd.DataFrame(records)
        print(
            f"  [Indexer] Found {len(df)} images across "
            f"{df['label'].nunique()} classes."
        )
        return df

    def save_index(self, dataframe: pd.DataFrame, output_path: Path) -> None:
        """Persist the index to a CSV so we do not re-scan every run.

        I save a CSV rather than a pickle so the index is human-readable
        and can be inspected in a spreadsheet if needed.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_path, index=False)
        print(f"  [Indexer] Index saved to {output_path}")

    def load_index(self, csv_path: Path) -> pd.DataFrame:
        """Load a previously saved dataset index from disk.

        I use this to skip the slow rglob scan on repeated runs when
        the dataset has not changed.
        """
        return pd.read_csv(csv_path)
