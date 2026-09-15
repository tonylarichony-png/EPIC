"""File discovery, loading, inventory, and schema cataloguing."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .profiling import DatasetProfiler


class DataCatalog:
    """Load configured datasets and generate file-level metadata."""

    def __init__(
        self,
        project_root: Path,
        raw_dir: Path,
        dataset_specs: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.raw_dir = (
            Path(raw_dir)
            if Path(raw_dir).is_absolute()
            else self.project_root / raw_dir
        )
        self.dataset_specs = {
            name: dict(spec)
            for name, spec in dataset_specs.items()
        }
        self._frames: dict[str, pd.DataFrame] = {}

    def path(self, name: str) -> Path:
        spec = self.dataset_specs[name]
        return self.raw_dir / str(spec["filename"])

    def validate(self) -> None:
        missing = [
            self.path(name)
            for name, spec in self.dataset_specs.items()
            if bool(spec.get("required", True)) and not self.path(name).exists()
        ]
        if missing:
            formatted = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(f"Required dataset files are missing:\n{formatted}")

    def available_names(self) -> list[str]:
        return [
            name
            for name in self.dataset_specs
            if self.path(name).exists()
        ]

    def load(self, name: str) -> pd.DataFrame:
        if name not in self.dataset_specs:
            raise KeyError(f"Unknown dataset {name!r}.")
        if name not in self._frames:
            path = self.path(name)
            if not path.exists():
                raise FileNotFoundError(path)
            self._frames[name] = pd.read_csv(path)
        return self._frames[name]

    def load_all(self) -> dict[str, pd.DataFrame]:
        self.validate()
        return {
            name: self.load(name)
            for name in self.available_names()
        }

    def file_report(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for name in self.available_names():
            path = self.path(name)
            frame = self.load(name)
            rows.append(
                {
                    "dataset": name,
                    "file": path.relative_to(self.project_root).as_posix(),
                    "role": self.dataset_specs[name].get("role", "unspecified"),
                    "rows": len(frame),
                    "columns": frame.shape[1],
                    "disk_kib": path.stat().st_size / 1024,
                    "memory_mib": frame.memory_usage(deep=True).sum() / 1024**2,
                    "sha256": self._sha256(path),
                }
            )
        return pd.DataFrame(rows)

    def schema_report(
        self,
        *,
        key: str | None,
        target: str | None,
        inference_dataset: str | None,
        field_descriptions: Mapping[str, str] | None = None,
    ) -> pd.DataFrame:
        field_descriptions = field_descriptions or {}
        inference_columns = (
            set(self.load(inference_dataset).columns)
            if inference_dataset in self.available_names()
            else set()
        )
        rows: list[dict[str, object]] = []

        for name in self.available_names():
            frame = self.load(name)
            spec_role = self.dataset_specs[name].get("role", "unspecified")
            schema = DatasetProfiler(frame, name=name).schema_report()
            for record in schema.to_dict(orient="records"):
                field = str(record["field"])
                if field == key:
                    field_role = "id"
                elif spec_role == "submission_example":
                    field_role = "output"
                elif field == target:
                    field_role = "target"
                else:
                    field_role = "feature"

                if spec_role == "submission_example":
                    available = "not applicable"
                elif field_role in {"target", "output"}:
                    available = "no"
                else:
                    available = "yes" if field in inference_columns else "no"

                rows.append(
                    {
                        "dataset": name,
                        "field": field,
                        "dtype": record["dtype"],
                        "description": field_descriptions.get(field, ""),
                        "role": field_role,
                        "available_at_inference": available,
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
