# src/services/image_preprocessor.py
# I keep all image-to-feature conversion logic inside this class so the
# ClassifierService never has to know anything about pixels, resizing, or
# normalisation. Separating these concerns also makes it easy to swap in a
# different preprocessing strategy without rewriting the training code.

import cv2
import numpy as np

from src.config import IMAGE_SIZE


class ImagePreprocessor:
    """Convert raw images into model-ready numeric features.

    I read each image in grayscale, resize it to a fixed size, normalise
    pixel values to the [0, 1] range, and flatten it into a 1-D array.
    This gives the RandomForestClassifier a consistent feature vector
    regardless of the original image size.
    """

    def __init__(self, image_size: tuple = IMAGE_SIZE) -> None:
        # I store image_size so it can be changed at construction time
        # for experiments without editing the config file.
        self.image_size = image_size

    def transform(self, file_path: str) -> np.ndarray:
        """Load, resize, normalise, and flatten one image.

        I always read in grayscale because colour information is less
        important than shape and texture for the baseline classifier,
        and it keeps the feature vector size manageable.

        Args:
            file_path: Path to the image file as a string.

        Returns:
            np.ndarray: A 1-D float32 array of normalised pixel values.

        Raises:
            ValueError: If the image cannot be read from disk.
        """
        # IMREAD_GRAYSCALE reads the image as a single-channel array,
        # which reduces the feature vector size compared to colour.
        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(
                f"Could not read image at path: {file_path}. "
                "Check that the file exists and is a supported format."
            )

        # I resize to the standard IMAGE_SIZE so every feature vector
        # has exactly the same length — the classifier requires this.
        resized = cv2.resize(image, self.image_size)

        # I normalise to [0, 1] so pixel magnitude does not dominate
        # the feature space. astype float32 keeps memory usage lower
        # than float64 while maintaining enough precision.
        normalised = resized.astype("float32") / 255.0

        # I flatten the 2-D array into a 1-D vector because
        # RandomForestClassifier expects a flat feature array per sample.
        return normalised.flatten()

    def transform_batch(self, file_paths: list) -> np.ndarray:
        """Transform a list of image paths into a 2-D feature matrix.

        I add this method as a convenience wrapper so the ClassifierService
        can process an entire list in one call instead of looping manually.

        Args:
            file_paths: List of path strings to image files.

        Returns:
            np.ndarray: Shape (n_images, n_features), float32.
        """
        features = []
        failed = 0
        for path in file_paths:
            try:
                features.append(self.transform(path))
            except ValueError as error:
                # I log the failure but continue so one bad image does not
                # crash the whole batch.
                print(f"  [Preprocessor] Skipping: {error}")
                failed += 1

        if failed > 0:
            print(f"  [Preprocessor] {failed} image(s) could not be processed.")

        return np.array(features)
