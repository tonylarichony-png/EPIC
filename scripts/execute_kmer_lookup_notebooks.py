"""Execute EXP-002/003/004 sequentially and preserve notebook outputs."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    ("EXP-002", ROOT / "notebooks/experiments/EXP-002_4mer_lookup.ipynb"),
    ("EXP-003", ROOT / "notebooks/experiments/EXP-003_6mer_lookup.ipynb"),
    ("EXP-004", ROOT / "notebooks/experiments/EXP-004_8mer_lookup.ipynb"),
)


def execute_notebook(experiment_id: str, notebook_path: Path) -> None:
    notebook = nbformat.read(notebook_path, as_version=4)

    def preserve_progress(cell, cell_index, **_kwargs):
        del cell
        print(
            datetime.now().isoformat(timespec="seconds"),
            experiment_id,
            "cell",
            cell_index,
            "completed",
            flush=True,
        )
        nbformat.write(notebook, notebook_path)

    client = NotebookClient(
        notebook,
        timeout=600,
        # The runner itself already executes inside epic-eda.  Its local
        # ``python3`` kernelspec points directly at that environment, whereas
        # the user-level ``epic-eda`` spec adds a nested ``conda run`` wrapper
        # that can exit early on Windows.
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        on_cell_executed=preserve_progress,
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, notebook_path)

    metadata_path = (
        ROOT / "artifacts/experiments" / experiment_id / "run_001/metadata.json"
    )
    if metadata_path.is_file():
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        record["run"]["notebook_sha256"] = hashlib.sha256(
            notebook_path.read_bytes()
        ).hexdigest()
        metadata_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(f"Notebook executed and saved: {notebook_path}", flush=True)


def main() -> None:
    for experiment_id, notebook_path in NOTEBOOKS:
        execute_notebook(experiment_id, notebook_path)


if __name__ == "__main__":
    main()
