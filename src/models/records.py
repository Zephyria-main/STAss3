# src/models/records.py
# I define the ImageRecord dataclass here to represent a single indexed image.
# Using a dataclass gives me a clean, typed structure without writing boilerplate
# __init__ and __repr__ methods manually.

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ImageRecord:
    """Store the core metadata for one indexed macroinvertebrate image.

    I use this as the unit of data that flows from the DatasetIndexer
    through the EDAService and into the ClassifierService. Keeping it
    as a dataclass means the fields are self-documenting and type-safe.
    """

    file_path: Path   # Absolute path to the image file on disk
    label: str        # Class name, derived from the parent folder name
    width: int        # Image width in pixels
    height: int       # Image height in pixels
    channels: int     # 1 for grayscale, 3 for colour (BGR/RGB)
