#*******************************
#Author: u323115
#Assessment 3 
#Programming: u3231515
#*******************************
#
# src/models/records.py
# I define the ImageRecord dataclass here to represent a single indexed image.
# Using a dataclass gives me a clean, typed structure without writing boilerplate
# __init__ and __repr__ methods manually.
#
# Unit tutorial / guidance — acknowledgement (Step 3: data model):
# Based on: Assignment 3 Full Guidance, Step 3 (ImageRecord dataclass with file_path,
# label, width, height, channels).
# How this project integrates it: same field semantics; file_path kept as Path type for
# type hints; used as the structured row concept alongside DatasetIndexer and pandas.
# See IMPLEMENTATION_SUMMARY.md.

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
