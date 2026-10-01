"""Synchronize saved EXP-007 artifacts into experiments/EXP-007.md."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml_project.epic_gc_binned_report import sync_exp007_report


if __name__ == "__main__":
    print(sync_exp007_report(PROJECT_ROOT))
