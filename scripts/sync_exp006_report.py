"""Refresh experiments/EXP-006.md from a completed run_001."""

from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "src/ml_project/epic_gc_report.py"
SPEC = importlib.util.spec_from_file_location("epic_gc_report", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load report module: {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> None:
    note = MODULE.sync_exp006_report(PROJECT_ROOT)
    print(f"Updated: {note}")


if __name__ == "__main__":
    main()
