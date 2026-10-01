"""Execute EXP-001 with its registered conda kernel and preserve outputs."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    ROOT / "notebooks/experiments/EXP-001_dinucleotide_baseline.ipynb"
)


def main() -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)

    def preserve_progress(cell, cell_index, **_kwargs):
        print(
            datetime.now().isoformat(timespec="seconds"),
            "cell",
            cell_index,
            "completed",
            flush=True,
        )
        nbformat.write(notebook, NOTEBOOK_PATH)

    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="epic-eda",
        resources={"metadata": {"path": str(ROOT)}},
        on_cell_executed=preserve_progress,
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, NOTEBOOK_PATH)
    metadata_path = (
        ROOT / "artifacts/experiments/EXP-001/run_001/metadata.json"
    )
    if metadata_path.is_file():
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        record["run"]["notebook_sha256"] = hashlib.sha256(
            NOTEBOOK_PATH.read_bytes()
        ).hexdigest()
        metadata_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(f"Notebook executed and saved: {NOTEBOOK_PATH}", flush=True)


if __name__ == "__main__":
    main()
