"""Figures, artifacts, cards and registries for grouped model screening."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..docsync import MarkdownDocument, dataframe_to_markdown
from .artifacts import build_metric_figures, metric_figure_filename
from .contracts import (
    ModelScreeningResult,
    ModelScreeningSettings,
    SavedModelScreening,
)
from .screening import validate_screening_settings
from .screening_diagnostics import build_staged_log_loss_diagnostics


TRACKED_SCREENING_FIGURE_ROOT = Path("assets/model-screening")


def build_screening_figures(
    result: ModelScreeningResult,
    settings: ModelScreeningSettings,
) -> dict[str, Any]:
    """Build metric, paired-fold, OOF and importance figures for the card."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError("Screening figures require matplotlib") from error

    figures = {
        metric_figure_filename(key, result.scoring.labels[key]): figure
        for key, figure in build_metric_figures(
            result.evaluation,
            result.scoring,
        ).items()
    }

    primary = result.leaderboard.sort_values("mean")
    figure, axis = plt.subplots(figsize=(9, max(4.5, 0.55 * len(primary))))
    axis.barh(
        primary["model"],
        primary["mean"],
        xerr=primary["std"].fillna(0.0),
        alpha=0.85,
    )
    axis.set_xlabel(str(primary.iloc[0]["metric"]))
    axis.set_title("Primary metric: mean ± std по одинаковым folds")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figures["ranking-primary.png"] = figure

    paired_primary = result.paired_deltas[
        result.paired_deltas["metric_key"].eq("primary")
    ]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    for model, rows in paired_primary.groupby("model", sort=False):
        axis.plot(
            rows["fold"],
            rows["improvement"],
            marker="o",
            linewidth=1.8,
            label=model,
        )
    axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axis.set_xlabel("CV fold")
    axis.set_ylabel("Improvement vs feature champion")
    axis.set_title("Парный Δ: положительное значение означает улучшение")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figures["paired-primary-delta.png"] = figure

    diagnostic_model = settings.diagnostic_model_id
    if diagnostic_model is None and result.shortlist:
        diagnostic_model = result.shortlist[0]
    importance = result.feature_importance[
        result.feature_importance["model"].eq(diagnostic_model)
    ].head(20)
    if not importance.empty:
        importance = importance.sort_values("importance_mean")
        figure, axis = plt.subplots(figsize=(9, max(4.5, 0.38 * len(importance))))
        axis.barh(
            importance["source_feature"],
            importance["importance_mean"],
            xerr=importance["importance_std"].fillna(0.0),
            alpha=0.85,
        )
        axis.set_xlabel(str(importance.iloc[0]["importance_kind"]))
        axis.set_title(f"Диагностика признаков: {diagnostic_model}")
        axis.grid(axis="x", alpha=0.25)
        figure.tight_layout()
        figures[f"importance-{diagnostic_model}.png"] = figure

    loss_rows = build_staged_log_loss_diagnostics(
        result.evaluation,
        result.built.context.data,
    )
    loss_models = loss_rows["model"].drop_duplicates().tolist()
    if loss_models:
        figure, axes = plt.subplots(
            1,
            len(loss_models),
            figsize=(6.2 * len(loss_models), 4.8),
            squeeze=False,
            sharey=True,
        )
        colors = {"train": "#64748b", "validation": "#2563eb"}
        for axis, model_name in zip(axes.ravel(), loss_models):
            model_rows = loss_rows[loss_rows["model"].eq(model_name)]
            summary = (
                model_rows.groupby(["split", "iteration"], as_index=False)
                .agg(
                    loss_mean=("log_loss", "mean"),
                    loss_std=("log_loss", "std"),
                )
                .fillna({"loss_std": 0.0})
            )
            for split in ("train", "validation"):
                split_rows = summary[summary["split"].eq(split)]
                axis.plot(
                    split_rows["iteration"],
                    split_rows["loss_mean"],
                    label=split,
                    color=colors[split],
                    linewidth=2.0,
                )
                axis.fill_between(
                    split_rows["iteration"],
                    split_rows["loss_mean"] - split_rows["loss_std"],
                    split_rows["loss_mean"] + split_rows["loss_std"],
                    color=colors[split],
                    alpha=0.10,
                )
            validation = summary[summary["split"].eq("validation")]
            best = validation.loc[validation["loss_mean"].idxmin()]
            best_iteration = int(best["iteration"])
            axis.axvline(
                best_iteration,
                color="#dc2626",
                linestyle="--",
                linewidth=1.2,
                label=f"best validation: {best_iteration}",
            )
            axis.set_title(str(model_name))
            axis.set_xlabel("Boosting iteration")
            axis.grid(alpha=0.22)
            axis.legend(frameon=False, fontsize=9)
        axes[0, 0].set_ylabel("Log loss (mean ± std across CV folds)")
        figure.suptitle("Train and validation loss", fontsize=13)
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        figures["boosting-log-loss.png"] = figure
    return figures


def _safe_project_path(root: Path, relative: Path, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} resolves outside the project root") from error
    return path


def _environment_snapshot() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in (
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
    ):
        try:
            packages[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            packages[distribution] = "not-installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **packages,
    }


def _screening_report_markdown(
    result: ModelScreeningResult,
    settings: ModelScreeningSettings,
    figure_paths: Mapping[str, Path],
    *,
    dataset_version: str,
) -> str:
    leaderboard = result.leaderboard[
        [
            "rank",
            "model",
            "metric",
            "mean ± std",
            "improvement_vs_reference",
            "fold_wins",
            "fold_ties",
            "fold_losses",
            "shortlisted",
        ]
    ].rename(
        columns={
            "rank": "Rank",
            "model": "Model",
            "metric": "Metric",
            "mean ± std": "Mean ± std",
            "improvement_vs_reference": "Δ vs champion",
            "fold_wins": "Wins",
            "fold_ties": "Ties",
            "fold_losses": "Losses",
            "shortlisted": "Shortlist",
        }
    )
    parts = [
        "## Контракт screening",
        "",
        dataframe_to_markdown(
            pd.DataFrame(
                [
                    ("Feature reference", result.built.context.feature_reference_id),
                    ("Feature module", result.built.context.feature_reference_module or "EXP-001 baseline"),
                    ("Dataset SHA-256", dataset_version),
                    ("Группа", result.built.group.title),
                    ("Preprocessing", result.built.group.preprocessing_profile),
                    ("Validation", result.cv_description),
                    ("Primary metric", result.scoring.contract_metric),
                    ("Shortlist", ", ".join(result.shortlist) or "—"),
                ],
                columns=["Поле", "Значение"],
            )
        ),
        "",
        "## Результат группы",
        "",
        dataframe_to_markdown(leaderboard, float_digits=4),
        "",
        "> [!important]",
        "> Screening ранжирует стартовые конфигурации. Он не доказывает, что",
        "> первое место — окончательно лучшая модель: shortlist сначала проходит",
        "> отдельный coarse tuning и диагностику на том же validation contract.",
        "",
        "## Параметры запуска",
        "",
        dataframe_to_markdown(result.built.parameters, float_digits=4),
        "",
        "## OOF-сравнение с feature champion",
        "",
        dataframe_to_markdown(result.prediction_comparison, float_digits=4),
        "",
        "## Графики",
        "",
    ]
    for path in figure_paths.values():
        parts.append(f"![[{path.as_posix()}]]")
        parts.append("")
    return "\n".join(parts).rstrip()


def _ensure_note(path: Path, settings: ModelScreeningSettings, feature_id: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "---",
                "type: model-screening",
                f"id: {settings.screening_id}",
                "status: completed",
                "decision: pending",
                f"feature_reference: {feature_id}",
                f"model_group: {settings.active_group}",
                "selected_model:",
                "---",
                "",
                f"# {settings.screening_id} — {settings.screening_title}",
                "",
                "<!-- auto:model-screening-report:start -->",
                "Отчёт появится после notebook write-action.",
                "<!-- auto:model-screening-report:end -->",
                "",
                "## Интерпретация",
                "",
                "- Что устойчиво по folds:",
                "- Где модель исправляет / создаёт ошибки:",
                "- Какие параметры ещё не исследованы:",
                "",
                "## Решение",
                "",
                "- Какие модели переходят в coarse tuning:",
                "- Почему:",
                "- Что запускаем следующим:",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _screening_stage_blocks(registry: pd.DataFrame) -> dict[str, str]:
    """Build a concise one-row-per-screening overview for stage 05."""

    runs: list[dict[str, Any]] = []
    for screening_id, rows in registry.groupby("screening_id", sort=False):
        ranked = rows.assign(
            _rank=pd.to_numeric(rows["rank"], errors="coerce"),
            _mean=pd.to_numeric(rows["mean"], errors="coerce"),
            _std=pd.to_numeric(rows["std"], errors="coerce"),
            _delta=pd.to_numeric(
                rows["improvement_vs_reference"], errors="coerce"
            ),
        ).sort_values("_rank", kind="stable")
        winner = ranked.iloc[0]

        reference_id = "feature_champion"
        if "reference_model" in ranked:
            configured = ranked["reference_model"].dropna()
            if not configured.empty:
                reference_id = str(configured.iloc[0])
        reference = ranked[ranked["model"].astype(str).eq(reference_id)]
        if reference.empty:
            reference = ranked.loc[[ranked["_delta"].abs().idxmin()]]
            reference_id = str(reference.iloc[0]["model"])
        reference_row = reference.iloc[0]

        shortlisted = ranked[
            ranked["shortlisted"]
            .astype(str)
            .str.strip()
            .str.casefold()
            .isin({"true", "1", "yes"})
        ]["model"].astype(str).tolist()
        note = str(winner["note"])
        link = f"[[{note}|{screening_id}]]"
        runs.append(
            {
                "screening_id": str(screening_id),
                "link": link,
                "title": str(winner["title"]),
                "feature_reference": str(winner["feature_reference"]),
                "group": str(winner["group"]),
                "winner": str(winner["model"]),
                "metric": str(winner["primary_metric"]),
                "reference_model": reference_id,
                "reference_mean": float(reference_row["_mean"]),
                "winner_mean": float(winner["_mean"]),
                "winner_std": float(winner["_std"]),
                "delta": float(winner["_delta"]),
                "shortlist": ", ".join(shortlisted) or "—",
                "candidate_count": max(len(ranked) - 1, 0),
            }
        )

    latest = runs[-1]
    latest_frame = pd.DataFrame(
        [
            {"Поле": "Screening", "Значение": f"{latest['link']} — {latest['title']}"},
            {"Поле": "Feature set", "Значение": latest["feature_reference"]},
            {"Поле": "Группа", "Значение": latest["group"]},
            {"Поле": "Проверено кандидатов", "Значение": latest["candidate_count"]},
            {"Поле": "Метрика", "Значение": latest["metric"]},
            {
                "Поле": "Reference",
                "Значение": (
                    f"{latest['reference_model']}: "
                    f"{latest['reference_mean']:.4f}"
                ),
            },
            {
                "Поле": "Лидер",
                "Значение": (
                    f"{latest['winner']}: {latest['winner_mean']:.4f} "
                    f"± {latest['winner_std']:.4f}"
                ),
            },
            {"Поле": "Δ к reference", "Значение": f"{latest['delta']:+.4f}"},
            {"Поле": "Shortlist", "Значение": latest["shortlist"]},
        ]
    )
    overview = pd.DataFrame(
        [
            {
                "Screening": run["link"],
                "Feature set": run["feature_reference"],
                "Группа": run["group"],
                "Лидер": run["winner"],
                "Метрика": run["metric"],
                "Reference": run["reference_mean"],
                "Лучший": run["winner_mean"],
                "Δ": f"{run['delta']:+.4f}",
                "Shortlist": run["shortlist"],
            }
            for run in runs
        ]
    )
    return {
        "latest-model-screening": dataframe_to_markdown(
            latest_frame, float_digits=4
        ),
        "model-screening-summary": dataframe_to_markdown(
            overview, float_digits=4
        ),
    }


def _sync_screening_stage_summary(root: Path, registry: pd.DataFrame) -> None:
    """Update stage 05 when that project report is present."""

    stage_path = root / "docs/05_experiments.md"
    if not stage_path.exists() or registry.empty:
        return
    MarkdownDocument(stage_path).update_blocks(
        _screening_stage_blocks(registry)
    )


def _sync_registry(
    root: Path,
    result: ModelScreeningResult,
    settings: ModelScreeningSettings,
    *,
    dataset_version: str,
) -> Path:
    path = _safe_project_path(root, settings.results_registry, "results_registry")
    path.parent.mkdir(parents=True, exist_ok=True)
    parameters = result.built.parameters.set_index("model_id")["params"].to_dict()
    rows = []
    for item in result.leaderboard.to_dict("records"):
        rows.append(
            {
                "screening_id": settings.screening_id,
                "title": settings.screening_title,
                "note": settings.screening_note.as_posix(),
                "feature_reference": result.built.context.feature_reference_id,
                "feature_reference_module": (
                    result.built.context.feature_reference_module or ""
                ),
                "feature_module_sha256": result.built.context.feature_reference_sha256 or "",
                "group": settings.active_group,
                "preprocessing_profile": result.built.group.preprocessing_profile,
                "reference_model": settings.reference_model_id,
                "model": item["model"],
                "primary_metric": item["metric"],
                "direction": item["direction"],
                "mean": item["mean"],
                "std": item["std"],
                "improvement_vs_reference": item["improvement_vs_reference"],
                "rank": item["rank"],
                "shortlisted": item["shortlisted"],
                "params": parameters.get(item["model"], "{}"),
                "run_name": settings.run_name,
                "dataset_version": dataset_version,
            }
        )
    current = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not current.empty:
        current = current[
            ~current["screening_id"].astype(str).eq(settings.screening_id)
        ]
    registry = pd.concat([current, pd.DataFrame(rows)], ignore_index=True)
    registry.to_csv(path, index=False)

    index_path = root / "model-screening/_index.md"
    if not index_path.exists():
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(
            "---\ntype: registry\nentity: model-screening\n---\n\n"
            "# Реестр группового screening моделей\n\n"
            "Запуски выполняются только через [[notebooks/06_model_screening.ipynb]].\n\n"
            "<!-- auto:model-screening-registry:start -->\n"
            "Реестр появится после первого сохранения.\n"
            "<!-- auto:model-screening-registry:end -->\n",
            encoding="utf-8",
        )
    summary = registry.copy()
    summary["Screening"] = summary.apply(
        lambda row: f"[[{row['note']}|{row['screening_id']}]]", axis=1
    )
    summary = summary[
        [
            "Screening",
            "feature_reference",
            "group",
            "model",
            "primary_metric",
            "mean",
            "std",
            "improvement_vs_reference",
            "rank",
            "shortlisted",
        ]
    ].rename(
        columns={
            "feature_reference": "Feature set",
            "group": "Group",
            "model": "Model",
            "primary_metric": "Metric",
            "mean": "Mean",
            "std": "Std",
            "improvement_vs_reference": "Δ vs champion",
            "rank": "Rank",
            "shortlisted": "Shortlist",
        }
    )
    MarkdownDocument(index_path).update_blocks(
        {"model-screening-registry": dataframe_to_markdown(summary, float_digits=4)}
    )
    _sync_screening_stage_summary(root, registry)
    return path


def save_model_screening(
    project_root: str | Path,
    result: ModelScreeningResult,
    settings: ModelScreeningSettings,
    *,
    dataset_version: str,
    figures: Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
) -> SavedModelScreening:
    """Explicitly persist local diagnostics and tracked screening evidence."""

    root = Path(project_root).resolve()
    validate_screening_settings(settings)
    run_dir = _safe_project_path(
        root,
        settings.artifact_dir / settings.run_name,
        "screening run directory",
    )
    figure_dir = _safe_project_path(
        root,
        TRACKED_SCREENING_FIGURE_ROOT / settings.screening_id,
        "screening figure directory",
    )
    note_path = _safe_project_path(root, settings.screening_note, "screening_note")
    config_sha256 = None
    if config_path is not None:
        resolved_config = Path(config_path).resolve()
        if resolved_config.is_file():
            config_sha256 = hashlib.sha256(resolved_config.read_bytes()).hexdigest()
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "screening_id": settings.screening_id,
        "screening_title": settings.screening_title,
        "run_name": settings.run_name,
        "dataset_version": dataset_version,
        "feature_reference_id": result.built.context.feature_reference_id,
        "feature_reference_module": result.built.context.feature_reference_module,
        "feature_reference_sha256": result.built.context.feature_reference_sha256,
        "config_sha256": config_sha256,
        "group": settings.active_group,
        "preprocessing_profile": result.built.group.preprocessing_profile,
        "cv": result.cv_description,
        "primary_metric": result.scoring.contract_metric,
        "shortlist": list(result.shortlist),
        "models": result.built.parameters.to_dict("records"),
        "environment": _environment_snapshot(),
    }
    metadata_path = run_dir / "metadata.json"
    if run_dir.exists() and any(run_dir.iterdir()) and not settings.allow_overwrite:
        raise FileExistsError(
            f"Screening run already exists: {run_dir}; change run_name or allow overwrite"
        )
    if (
        figure_dir.exists()
        and any(figure_dir.glob("*.png"))
        and not settings.allow_overwrite
    ):
        raise FileExistsError(
            f"Screening figures already exist: {figure_dir}; use a new MS-ID"
        )
    if metadata_path.exists() and settings.allow_overwrite:
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        contract_keys = {
            "screening_id",
            "run_name",
            "dataset_version",
            "feature_reference_id",
            "feature_reference_module",
            "feature_reference_sha256",
            "group",
            "preprocessing_profile",
            "cv",
            "primary_metric",
            "models",
        }
        previous_contract = {key: previous.get(key) for key in contract_keys}
        current_contract = {key: metadata.get(key) for key in contract_keys}
        if previous_contract != current_contract:
            raise FileExistsError(
                "Existing run has another screening contract. Use a new MS-ID "
                "and run_name instead of overwriting changed models or data."
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated_names = {
        "cv_fold_scores.csv",
        "cv_summary.csv",
        "leaderboard.csv",
        "paired_deltas.csv",
        "oof_predictions.csv",
        "prediction_comparison.csv",
        "feature_importance.csv",
        "model_parameters.csv",
        "metadata.json",
    }
    if settings.allow_overwrite:
        for path in run_dir.iterdir():
            if path.is_file() and path.name in generated_names:
                path.unlink()
        for path in figure_dir.glob("*.png"):
            path.unlink()

    fold_scores_path = run_dir / "cv_fold_scores.csv"
    summary_path = run_dir / "cv_summary.csv"
    leaderboard_path = run_dir / "leaderboard.csv"
    result.evaluation.fold_scores.to_csv(fold_scores_path, index=False)
    result.evaluation.summary.to_csv(summary_path, index=False)
    result.leaderboard.to_csv(leaderboard_path, index=False)
    result.paired_deltas.to_csv(run_dir / "paired_deltas.csv", index=False)
    result.oof_predictions.to_csv(run_dir / "oof_predictions.csv", index=False)
    result.prediction_comparison.to_csv(
        run_dir / "prediction_comparison.csv", index=False
    )
    result.feature_importance.to_csv(run_dir / "feature_importance.csv", index=False)
    result.built.parameters.to_csv(run_dir / "model_parameters.csv", index=False)

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    figure_paths: dict[str, Path] = {}
    if settings.save_figures:
        for filename, figure in (figures or {}).items():
            safe_name = Path(filename).name
            if not safe_name.lower().endswith(".png"):
                safe_name += ".png"
            path = figure_dir / safe_name
            figure.savefig(path, dpi=settings.figure_dpi, bbox_inches="tight")
            figure_paths[safe_name] = (
                TRACKED_SCREENING_FIGURE_ROOT / settings.screening_id / safe_name
            )

    if settings.sync_screening_note:
        _ensure_note(note_path, settings, result.built.context.feature_reference_id)
        report = _screening_report_markdown(
            result,
            settings,
            figure_paths,
            dataset_version=dataset_version,
        )
        MarkdownDocument(note_path).update_blocks({"model-screening-report": report})

    registry_path = _safe_project_path(
        root, settings.results_registry, "results_registry"
    )
    if settings.sync_registry:
        registry_path = _sync_registry(
            root,
            result,
            settings,
            dataset_version=dataset_version,
        )
    return SavedModelScreening(
        run_dir=run_dir,
        figure_dir=figure_dir,
        note_path=note_path,
        fold_scores_path=fold_scores_path,
        summary_path=summary_path,
        leaderboard_path=leaderboard_path,
        metadata_path=metadata_path,
        registry_path=registry_path,
        figure_paths=figure_paths,
    )

__all__ = [
    "TRACKED_SCREENING_FIGURE_ROOT",
    "build_screening_figures",
    "save_model_screening",
]
