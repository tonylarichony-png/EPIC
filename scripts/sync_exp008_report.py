"""Synchronize saved EXP-008 artifacts into experiments/EXP-008.md."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml_project.epic_gc_interaction_report import sync_exp008_report


if __name__ == "__main__":
    print(sync_exp008_report(PROJECT_ROOT))
