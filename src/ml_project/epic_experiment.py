"""Человекочитаемая запись запусков геномных экспериментов EPIC."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
from typing import Mapping

import numpy as np
import pandas as pd

from .epic_data import PreparedSplit, sha256_file


EXPERIMENT_ID_PATTERN = re.compile(r"EXP-\d{3,}")
RUN_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")
DECISIONS = {"pending", "adopt", "reject", "iterate", "inconclusive"}


@dataclass(frozen=True)
class ExperimentSpec:
    """То, что необходимо зафиксировать до обучения модели."""

    experiment_id: str
    title: str
    hypothesis: str
    changed_variable: str
    success_criterion: str
    primary_metric: str = "average_precision"
    secondary_metric: str = "epic_spearman"
    seed: int = 42

    def __post_init__(self) -> None:
        if not EXPERIMENT_ID_PATTERN.fullmatch(self.experiment_id):
            raise ValueError("experiment_id должен выглядеть как EXP-006")
        for field_name in (
            "title",
            "hypothesis",
            "changed_variable",
            "success_criterion",
            "primary_metric",
            "secondary_metric",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} не должен быть пустым")


def _manifest_hash(manifest: Mapping[str, object]) -> str:
    payload = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_value(value):
    if value is pd.NA:
        return None
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def build_run_record(
    spec: ExperimentSpec,
    split: PreparedSplit,
    *,
    run_name: str,
    notebook_path: Path | str,
    parameters: Mapping[str, object],
    metrics: Mapping[str, object],
    interpretation: str,
    decision: str = "pending",
) -> dict[str, object]:
    """Собрать самодостаточный JSON-отчёт о конкретном запуске."""

    if not RUN_NAME_PATTERN.fullmatch(run_name):
        raise ValueError("run_name: только строчные буквы, цифры, _ и -")
    if decision not in DECISIONS:
        raise ValueError(f"decision должен быть одним из {sorted(DECISIONS)}")
    required_metrics = {
        f"{role}_{metric}"
        for metric in (spec.primary_metric, spec.secondary_metric)
        for role in ("reference", "candidate", "delta")
    }
    missing_metrics = sorted(required_metrics - set(metrics))
    if missing_metrics:
        raise ValueError(
            "Не записаны обе обязательные метрики: "
            + ", ".join(missing_metrics)
        )
    notebook = Path(notebook_path).resolve()
    project_root = split.paths.project_root.resolve()
    try:
        notebook_relative = notebook.relative_to(project_root).as_posix()
    except ValueError:
        notebook_relative = notebook.as_posix()

    record: dict[str, object] = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": asdict(spec),
        "run": {
            "name": run_name,
            "decision": decision,
            "interpretation": interpretation.strip(),
            "notebook": notebook_relative,
            "notebook_sha256": (
                sha256_file(notebook) if notebook.is_file() else None
            ),
        },
        "data_contract": {
            "assembly": split.config.assembly,
            "split_version": split.config.split_version,
            "split_seed": split.config.seed,
            "validation_contig_fraction": (
                split.config.validation_contig_fraction
            ),
            "manifest_sha256": _manifest_hash(split.manifest),
            "source_sha256": split.manifest.get("source_sha256", {}),
            "artifact_sha256": split.manifest.get("artifact_sha256", {}),
            "summary": split.summary.to_dict(orient="records"),
        },
        "parameters": dict(parameters),
        "metrics": dict(metrics),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
    }
    return _json_value(record)


def save_run_record(
    project_root: Path | str,
    record: Mapping[str, object],
    *,
    allow_overwrite: bool = False,
) -> Path:
    """Сохранить metadata.json и плоскую metrics.csv в artifacts/."""

    root = Path(project_root).resolve()
    experiment = record.get("experiment", {})
    run = record.get("run", {})
    experiment_id = str(experiment.get("experiment_id", ""))
    run_name = str(run.get("name", ""))
    if not EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        raise ValueError("В record нет корректного experiment_id")
    if not RUN_NAME_PATTERN.fullmatch(run_name):
        raise ValueError("В record нет корректного run.name")

    output = root / "artifacts/experiments" / experiment_id / run_name
    metadata_path = output / "metadata.json"
    metrics_path = output / "metrics.csv"
    if not allow_overwrite and (metadata_path.exists() or metrics_path.exists()):
        raise FileExistsError(
            f"Запуск уже записан: {output}. Используйте новое run_name."
        )
    output.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metric_rows = [
        {"metric": name, "value": value}
        for name, value in record.get("metrics", {}).items()
    ]
    pd.DataFrame(metric_rows, columns=["metric", "value"]).to_csv(
        metrics_path,
        index=False,
        lineterminator="\n",
    )
    return output


__all__ = [
    "DECISIONS",
    "EXPERIMENT_ID_PATTERN",
    "ExperimentSpec",
    "build_run_record",
    "save_run_record",
]
