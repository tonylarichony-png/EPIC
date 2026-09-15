"""Persist local run data and Git-tracked experiment metric figures."""

from __future__ import annotations

import inspect
import json
import platform
import re
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping

from ._utils import _display_value, _json_safe, _sklearn_import_error
from .contracts import (
    ModelingSettings,
    CVEvaluation,
    FeaturePlan,
    PreparedData,
    SavedBaselineRun,
    ScoringPlan,
    TRACKED_METRIC_FIGURE_ROOT,
)
from .estimators import resolved_model_name


def _environment_snapshot() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "joblib",
    ):
        try:
            packages[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            packages[distribution] = "not-installed"
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        **packages,
    }


def metric_figure_filename(metric_key: str, metric_label: str) -> str:
    """Return a stable, filesystem-safe filename for one metric chart."""

    slug = re.sub(r"[^a-z0-9]+", "-", metric_label.lower()).strip("-")
    slug = slug or "metric"
    return f"metric-{metric_key}-{slug}.png"


def build_metric_figures(
    evaluation: CVEvaluation,
    scoring: ScoringPlan,
) -> dict[str, Any]:
    """Build one fold-by-fold validation chart for every configured metric."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "Metric figures require matplotlib. Install project dependencies "
            "with: python -m pip install -r requirements.txt"
        ) from error

    validation = evaluation.fold_scores[
        evaluation.fold_scores["split"] == "validation"
    ]
    figures: dict[str, Any] = {}
    for metric_key in scoring.scorers:
        metric_rows = validation[validation["metric_key"] == metric_key]
        if metric_rows.empty:
            continue

        label = scoring.labels[metric_key]
        direction = scoring.directions[metric_key]
        figure, axis = plt.subplots(figsize=(8.5, 4.8))
        for model_name in metric_rows["model"].drop_duplicates():
            model_rows = metric_rows[
                metric_rows["model"] == model_name
            ].sort_values("fold")
            mean = float(model_rows["value"].mean())
            std = float(model_rows["value"].std())
            line, = axis.plot(
                model_rows["fold"],
                model_rows["value"],
                marker="o",
                linewidth=2,
                label=f"{model_name}: {mean:.4f} ± {std:.4f}",
            )
            axis.axhline(
                mean,
                color=line.get_color(),
                linestyle=":",
                linewidth=1.2,
                alpha=0.65,
            )

        folds = sorted(int(value) for value in metric_rows["fold"].unique())
        axis.set_xticks(folds)
        axis.set_xlabel("CV fold")
        axis.set_ylabel(label)
        axis.set_title(f"{label} по validation folds ({direction})")
        axis.grid(True, alpha=0.25)
        axis.legend(title="Модель: mean ± std", frameon=False)
        figure.tight_layout()
        figures[metric_key] = figure
    return figures


def _cleanup_generated_run_files(run_dir: Path) -> None:
    """Remove only files owned by the baseline generator, never manual files."""

    known_files = {
        "cv_fold_scores.csv",
        "cv_summary.csv",
        "metadata.json",
        "model.joblib",
    }
    for path in run_dir.iterdir():
        generated_metric_figure = (
            path.is_file()
            and path.name.startswith("metric-")
            and path.suffix.lower() == ".png"
        )
        if path.is_file() and (path.name in known_files or generated_metric_figure):
            path.unlink()


def _metric_figure_directory(root: Path, experiment_id: str) -> Path:
    """Resolve the Git-tracked figure directory for one experiment."""

    figure_root = (root / TRACKED_METRIC_FIGURE_ROOT).resolve()
    figure_dir = (figure_root / experiment_id).resolve()
    try:
        figure_dir.relative_to(figure_root)
    except ValueError as error:
        raise ValueError(
            "settings.experiment_id resolves outside the tracked metric figure "
            "directory"
        ) from error
    return figure_dir


def _cleanup_generated_metric_figures(figure_dir: Path) -> None:
    """Refresh generated metric PNGs while preserving manual experiment assets."""

    for path in figure_dir.iterdir():
        if (
            path.is_file()
            and path.name.startswith("metric-")
            and path.suffix.lower() == ".png"
        ):
            path.unlink()


def save_baseline_run(
    project_root: Path,
    settings: ModelingSettings,
    evaluation: CVEvaluation,
    plan: FeaturePlan,
    scoring: ScoringPlan,
    *,
    dataset_version: str,
    cv_description: str,
    final_pipeline: Any | None = None,
    data: PreparedData | None = None,
    metric_figures: Mapping[str, Any] | None = None,
) -> SavedBaselineRun:
    """Save local run data, tracked figures and optionally a final pipeline.

    With ``settings.allow_overwrite=True`` repeated execution refreshes only files owned
    by this generator. Unknown files in run and figure directories are preserved.
    """

    root = Path(project_root).resolve()
    base_dir = (root / settings.artifact_dir).resolve()
    try:
        base_dir.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "settings.artifact_dir resolves outside the project root"
        ) from error

    run_dir = (base_dir / settings.run_name).resolve()
    try:
        run_dir.relative_to(base_dir)
    except ValueError as error:
        raise ValueError(
            "settings.run_name resolves outside settings.artifact_dir"
        ) from error

    figure_dir = (
        _metric_figure_directory(root, settings.experiment_id)
        if settings.save_metric_figures
        else None
    )
    if run_dir.exists() and any(run_dir.iterdir()) and not settings.allow_overwrite:
        raise FileExistsError(
            f"Modeling run already exists: {run_dir}. Change settings.run_name "
            "or set settings.allow_overwrite=True deliberately."
        )
    if (
        figure_dir is not None
        and figure_dir.exists()
        and any(
            path.is_file()
            and path.name.startswith("metric-")
            and path.suffix.lower() == ".png"
            for path in figure_dir.iterdir()
        )
        and not settings.allow_overwrite
    ):
        raise FileExistsError(
            f"Metric figures already exist: {figure_dir}. Change "
            "settings.experiment_id or set settings.allow_overwrite=True "
            "deliberately."
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    if settings.allow_overwrite:
        _cleanup_generated_run_files(run_dir)
    if figure_dir is not None:
        figure_dir.mkdir(parents=True, exist_ok=True)
        if settings.allow_overwrite:
            _cleanup_generated_metric_figures(figure_dir)

    fold_scores_path = run_dir / "cv_fold_scores.csv"
    summary_path = run_dir / "cv_summary.csv"
    metadata_path = run_dir / "metadata.json"
    evaluation.fold_scores.to_csv(fold_scores_path, index=False)
    evaluation.summary.to_csv(summary_path, index=False)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": settings.experiment_id,
        "experiment_title": settings.experiment_title,
        "experiment_note": settings.experiment_note.as_posix(),
        "run_name": settings.run_name,
        "task_type": settings.task_type,
        "dataset_version": dataset_version,
        "cv": cv_description,
        "primary_metric": scoring.contract_metric,
        "primary_scorer": _display_value(scoring.scorers["primary"]),
        "model": resolved_model_name(settings),
        "model_params": dict(settings.model_params),
        "numeric_features": list(plan.numeric),
        "categorical_features": list(plan.categorical),
        "excluded_features": list(plan.excluded),
        "metric_figures": {
            key: (
                TRACKED_METRIC_FIGURE_ROOT
                / settings.experiment_id
                / metric_figure_filename(key, scoring.labels[key])
            ).as_posix()
            for key in scoring.scorers
        } if settings.save_metric_figures else {},
        "environment": _environment_snapshot(),
        "settings": _json_safe(asdict(settings)),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    figure_paths: dict[str, Path] = {}
    if settings.save_metric_figures:
        if figure_dir is None:  # pragma: no cover - guarded by the same setting
            raise RuntimeError("Metric figure directory was not initialized")
        figures = dict(metric_figures or build_metric_figures(evaluation, scoring))
        missing = [key for key in scoring.scorers if key not in figures]
        if missing:
            raise ValueError(
                "Metric figures are missing for: " + ", ".join(missing)
            )
        for metric_key in scoring.scorers:
            figure_path = figure_dir / metric_figure_filename(
                metric_key, scoring.labels[metric_key]
            )
            figures[metric_key].savefig(
                figure_path,
                dpi=settings.metric_figure_dpi,
                bbox_inches="tight",
            )
            figure_paths[metric_key] = figure_path

    model_path: Path | None = None
    if settings.save_final_model:
        if final_pipeline is None or data is None:
            raise ValueError(
                "final_pipeline and data are required when "
                "settings.save_final_model=True"
            )
        try:
            from joblib import dump
            from sklearn.base import clone
        except ImportError as error:  # pragma: no cover - environment dependent
            raise _sklearn_import_error() from error
        fitted = clone(final_pipeline).fit(data.X, data.y)
        model_path = run_dir / "model.joblib"
        dump(fitted, model_path)

    return SavedBaselineRun(
        run_dir=run_dir,
        fold_scores_path=fold_scores_path,
        summary_path=summary_path,
        metadata_path=metadata_path,
        model_path=model_path,
        metric_figure_paths=figure_paths,
    )
