"""Project-specific dataset configuration shared by all notebooks."""

from pathlib import Path

RAW_DIR = Path("data/raw")

# Edit this mapping once; both 01_data.ipynb and 02_eda.ipynb import it.
DATASETS = {
    "train": {
        "filename": "train.csv",
        "role": "train",
        "required": True,
    },
    "test": {
        "filename": "test.csv",
        "role": "inference",
        "required": False,
    },
    "submission_example": {
        "filename": "sample_submission.csv",
        "role": "submission_example",
        "required": False,
    },
}

TRAIN_DATASET = "train"
INFERENCE_DATASET = "test"
KEY = None       # Example: "id"
TARGET = None    # Example: "target"

# Assign every train column except TARGET to exactly one group.
# numeric + count are used in numeric EDA;
# categorical + ordinal are used in categorical EDA.
FEATURE_GROUPS = {
    "numeric": [],
    "count": [],
    "categorical": [],
    "ordinal": [],
    "text": [],
    "datetime": [],
    "identifier": [],
    "ignored": [],
}

# Optional human-readable column descriptions used in docs/01_data.md.
FIELD_DESCRIPTIONS = {}
