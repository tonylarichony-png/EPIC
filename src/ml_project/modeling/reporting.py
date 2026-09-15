"""Markdown reports and auto-synchronization for modeling results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..docsync import MarkdownDocument, dataframe_to_markdown
from .contracts import (
    ModelingSettings,
    CVEvaluation,
    FeaturePlan,
    PreparedData,
    SavedBaselineRun,
    ScoringPlan,
)
from .estimators import resolved_model_label
from .report_blocks import (
    build_best_result_block,
    build_experiment_registry_block,
    build_feature_registry_block,
    build_key_results_block,
    build_model_ready_block,
    build_preprocessing_block,
    build_reproducibility_block,
    build_secondary_metrics_block,
    build_validation_protocol_block,
)


def _vault_relative(project_root: Path, path: Path) -> str:
    root = Path(project_root).resolve()
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"Path resolves outside the project root: {path}") from error


def _wikilink(project_root: Path, path: Path, alias: str | None = None) -> str:
    relative = _vault_relative(project_root, path)
    return f"[[{relative}|{alias}]]" if alias else f"[[{relative}]]"


def build_baseline_experiment_report(
    project_root: Path,
    settings: ModelingSettings,
    evaluation: CVEvaluation,
    scoring: ScoringPlan,
    saved_run: SavedBaselineRun,
    *,
    dataset_version: str,
    cv_description: str,
    model_name: str = "simple_model",
) -> str:
    """Render an Obsidian-native report with charts and readable CV tables."""

    overview = pd.DataFrame(
        [
            ("Эксперимент", f"{settings.experiment_id} — {settings.experiment_title}"),
            ("Run", settings.run_name),
            ("Версия данных", dataset_version),
            ("Validation", cv_description),
            ("Основная метрика", scoring.contract_metric),
            ("Основная модель", resolved_model_label(model_name)),
        ],
        columns=["Поле", "Значение"],
    )

    validation_summary = evaluation.summary[
        evaluation.summary["split"] == "validation"
    ].copy()
    validation_summary["mean ± std"] = validation_summary.apply(
        lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}", axis=1
    )
    comparison = validation_summary[
        ["model", "metric", "direction", "mean ± std", "min", "max", "folds"]
    ].rename(
        columns={
            "model": "Модель",
            "metric": "Метрика",
            "direction": "Направление",
            "min": "Min",
            "max": "Max",
            "folds": "Folds",
        }
    )

    sections = [
        "## Сводка запуска\n\n" + dataframe_to_markdown(overview),
        "## Сравнение всех метрик\n\n" + dataframe_to_markdown(comparison, float_digits=4),
    ]

    figure_paths = dict(saved_run.metric_figure_paths or {})
    validation_folds = evaluation.fold_scores[
        evaluation.fold_scores["split"] == "validation"
    ]
    for metric_key in scoring.scorers:
        label = scoring.labels[metric_key]
        metric_summary = validation_summary[
            validation_summary["metric_key"] == metric_key
        ][["model", "mean", "std", "min", "max", "folds", "direction"]].rename(
            columns={
                "model": "Модель",
                "mean": "Mean",
                "std": "Std",
                "min": "Min",
                "max": "Max",
                "folds": "Folds",
                "direction": "Направление",
            }
        )
        metric_folds = validation_folds[
            validation_folds["metric_key"] == metric_key
        ].pivot(index="fold", columns="model", values="value")
        metric_folds = metric_folds.reset_index().rename(columns={"fold": "Fold"})

        metric_parts = [f"## Метрика: {label}"]
        if metric_key in figure_paths:
            figure_link = _vault_relative(project_root, figure_paths[metric_key])
            metric_parts.append(f"![[{figure_link}]]")
        metric_parts.extend(
            [
                "### Сводка по моделям\n\n"
                + dataframe_to_markdown(metric_summary, float_digits=4),
                "### Значения по folds\n\n"
                + dataframe_to_markdown(metric_folds, float_digits=4),
            ]
        )
        sections.append("\n\n".join(metric_parts))

    artifact_lines = [
        f"- Полная таблица folds: {_wikilink(project_root, saved_run.fold_scores_path, 'cv_fold_scores.csv')}",
        f"- Полная сводка: {_wikilink(project_root, saved_run.summary_path, 'cv_summary.csv')}",
        f"- Конфигурация запуска: {_wikilink(project_root, saved_run.metadata_path, 'metadata.json')}",
    ]
    if saved_run.model_path is not None:
        artifact_lines.append(
            f"- Финальная модель: {_wikilink(project_root, saved_run.model_path, 'model.joblib')}"
        )
    sections.append("## Артефакты\n\n" + "\n".join(artifact_lines))
    return "\n\n".join(sections)


def sync_baseline_experiment_note(
    project_root: Path,
    settings: ModelingSettings,
    evaluation: CVEvaluation,
    scoring: ScoringPlan,
    saved_run: SavedBaselineRun,
    *,
    dataset_version: str,
    cv_description: str,
    model_name: str = "simple_model",
) -> list[str]:
    """Create/update the generated block while preserving manual analysis."""

    root = Path(project_root).resolve()
    note_path = (root / settings.experiment_note).resolve()
    _vault_relative(root, note_path)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    if not note_path.exists():
        note_path.write_text(
            "---\n"
            f"id: {settings.experiment_id}\n"
            "type: experiment\n"
            "experiment_type: baseline\n"
            "status: completed\n"
            "---\n\n"
            f"# {settings.experiment_id} — {settings.experiment_title}\n\n"
            "> [!info] Автоматический отчёт\n"
            "> Перезапуск notebook обновляет только блок ниже. Ручные выводы "
            "после него сохраняются.\n\n"
            "<!-- auto:baseline-experiment-report:start -->\n\n"
            "Отчёт появится после сохранения baseline.\n\n"
            "<!-- auto:baseline-experiment-report:end -->\n\n"
            "## Анализ и выводы\n\n"
            "- Что показало сравнение с dummy:\n"
            "- Насколько результат стабилен между folds:\n"
            "- Какие метрики расходятся и почему это важно:\n"
            "- Что проверить следующим экспериментом:\n",
            encoding="utf-8",
        )

    report = build_baseline_experiment_report(
        root,
        settings,
        evaluation,
        scoring,
        saved_run,
        dataset_version=dataset_version,
        cv_description=cv_description,
        model_name=model_name,
    )
    return MarkdownDocument(note_path).update_blocks(
        {"baseline-experiment-report": report}
    )


def _selected_primary_record(
    evaluation: CVEvaluation,
    model_name: str,
) -> pd.Series:
    primary = evaluation.primary_summary()
    row = primary[primary["model"] == model_name]
    if row.empty:
        row = primary.tail(1)
    return row.iloc[0]


def build_validation_baseline_block(
    evaluation: CVEvaluation,
    scoring: ScoringPlan,
    *,
    dataset_version: str,
    cv_description: str,
) -> str:
    """Render the comparable primary scores for Validation."""

    primary = evaluation.primary_summary().copy()
    report = pd.DataFrame(
        {
            "Baseline": primary["model"],
            "Версия данных": dataset_version,
            "Протокол": cv_description,
            "Метрика": scoring.contract_metric,
            "Значение": primary.apply(
                lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}", axis=1
            ),
        }
    )
    return dataframe_to_markdown(report)


def build_current_baseline_block(
    evaluation: CVEvaluation,
    scoring: ScoringPlan,
    *,
    dataset_version: str,
    cv_description: str,
    model_name: str = "simple_model",
    experiment_note: Path | None = None,
) -> str:
    """Render the selected simple reference model for Experiments."""

    record = _selected_primary_record(evaluation, model_name)
    rows: list[tuple[str, Any]] = [
            ("Эксперимент / версия", "Baseline / " + model_name),
            ("Данные", dataset_version),
            ("Модель", resolved_model_label(model_name)),
            ("Validation", cv_description),
            ("Основная метрика", scoring.contract_metric),
            ("Значение", f"{record['mean']:.4f} ± {record['std']:.4f}"),
            ("Стоимость / latency", "см. cv_summary.csv"),
    ]
    if experiment_note is not None:
        rows.insert(
            1,
            (
                "Карточка",
                f"[[{experiment_note.as_posix()}|подробный baseline-отчёт]]",
            ),
        )
    report = pd.DataFrame(
        rows,
        columns=["Поле", "Значение"],
    )
    return dataframe_to_markdown(report)


def sync_baseline_docs(
    project_root: Path,
    evaluation: CVEvaluation,
    scoring: ScoringPlan,
    *,
    dataset_version: str,
    cv_description: str,
    model_name: str = "simple_model",
    experiment_note: Path | None = None,
    settings: ModelingSettings | None = None,
    feature_plan: FeaturePlan | None = None,
    prepared_data: PreparedData | None = None,
    saved_run: SavedBaselineRun | None = None,
) -> dict[str, list[str]]:
    """Update explicit baseline, metric and dashboard auto-blocks."""

    root = Path(project_root)
    validation_payload = {
        "secondary-metrics": build_secondary_metrics_block(scoring),
        "baseline-results": build_validation_baseline_block(
            evaluation,
            scoring,
            dataset_version=dataset_version,
            cv_description=cv_description,
        ),
    }
    if settings is not None:
        validation_payload["validation-protocol"] = build_validation_protocol_block(
            settings,
            scoring,
            cv_description=cv_description,
        )
        validation_payload["reproducibility"] = build_reproducibility_block(
            root,
            settings,
            dataset_version=dataset_version,
            cv_description=cv_description,
            saved_run=saved_run,
        )
    validation_blocks = MarkdownDocument(root / "docs/03_validation.md").update_blocks(
        validation_payload
    )
    feature_blocks: list[str] = []
    if settings is not None and feature_plan is not None:
        feature_blocks = MarkdownDocument(root / "docs/04_features.md").update_blocks(
            {
                "model-ready-contract": build_model_ready_block(
                    settings,
                    feature_plan,
                    dataset_version=dataset_version,
                    data=prepared_data,
                ),
                "feature-registry": build_feature_registry_block(feature_plan),
                "preprocessing-contract": build_preprocessing_block(
                    settings,
                    feature_plan,
                ),
            }
        )
    experiment_blocks = MarkdownDocument(root / "docs/05_experiments.md").update_blocks(
        {
            "current-baseline": build_current_baseline_block(
                evaluation,
                scoring,
                dataset_version=dataset_version,
                cv_description=cv_description,
                model_name=model_name,
                experiment_note=experiment_note,
            )
        }
    )
    registry_path = root / "experiments/results.csv"
    registry = (
        pd.read_csv(registry_path)
        if registry_path.exists() and registry_path.stat().st_size
        else None
    )
    baseline_record = _selected_primary_record(evaluation, model_name)
    best_result_block = build_best_result_block(
        baseline_metric=scoring.contract_metric,
        baseline_score=float(baseline_record["mean"]),
        baseline_std=float(baseline_record["std"]),
        baseline_direction=scoring.directions["primary"],
        baseline_note=experiment_note,
        experiments=registry,
    )
    experiment_blocks.extend(
        MarkdownDocument(root / "docs/05_experiments.md").update_blocks(
            {"best-measured-result": best_result_block}
        )
    )
    registry_blocks = MarkdownDocument(root / "experiments/_index.md").update_blocks(
        {
            "experiment-registry": build_experiment_registry_block(
                experiment_note,
                registry,
                baseline_run=settings.run_name if settings is not None else "baseline",
                baseline_metric=scoring.contract_metric,
                baseline_score=float(baseline_record["mean"]),
            )
        }
    )
    readme_blocks = MarkdownDocument(root / "README.md").update_blocks(
        {
            "key-results": build_key_results_block(
                baseline_metric=scoring.contract_metric,
                baseline_score=float(baseline_record["mean"]),
                baseline_note=experiment_note,
                experiments=registry,
            )
        }
    )
    return {
        "README.md": readme_blocks,
        "docs/03_validation.md": validation_blocks,
        "docs/04_features.md": feature_blocks,
        "docs/05_experiments.md": experiment_blocks,
        "experiments/_index.md": registry_blocks,
    }
