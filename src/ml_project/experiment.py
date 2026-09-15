"""Reusable orchestration for controlled experiments after the baseline."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import re
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import modeling as modeling_tools
from .modeling import (
    ModelingSettings,
    CVEvaluation,
    ExperimentData,
    ExperimentDefinition,
    ExperimentDiagnostics,
    ExperimentSettings,
    FeaturePlan,
    PreparedData,
    SavedBaselineRun,
    SavedDiagnostics,
    ScoringPlan,
)
from .docsync import MarkdownDocument, dataframe_to_markdown
from .experiment_relations import sync_experiment_eda_relations
from .experiment_state import (
    DECISIONS,
    settings_with_card_decision,
    sync_experiment_state,
)


EXPERIMENT_MODULE_PREFIX = "ml_project.experiments."


def load_experiment(
    module_name: str,
    *,
    reload_module: bool = True,
) -> ExperimentDefinition:
    """Load one versioned experiment module and capture its source hash."""

    if not module_name.startswith(EXPERIMENT_MODULE_PREFIX):
        raise ValueError(
            "EXPERIMENT_MODULE must start with "
            f"{EXPERIMENT_MODULE_PREFIX!r}"
        )
    module = importlib.import_module(module_name)
    if reload_module:
        module = importlib.reload(module)
    source_value = getattr(module, "__file__", None)
    if not source_value:
        raise ValueError(f"Experiment module has no source file: {module_name}")
    source_path = Path(source_value).resolve()
    if source_path.suffix.lower() != ".py":
        raise ValueError(
            f"Experiment module must resolve to a .py source file: {source_path}"
        )
    settings = getattr(module, "EXPERIMENT", None)
    prepare_data = getattr(module, "prepare_candidate_data", None)
    build_models = getattr(module, "build_candidate_models", None)
    if not isinstance(settings, ExperimentSettings):
        raise TypeError(
            f"{module_name}.EXPERIMENT must be an ExperimentSettings object"
        )
    if not callable(prepare_data):
        raise TypeError(f"{module_name}.prepare_candidate_data must be callable")
    if not callable(build_models):
        raise TypeError(f"{module_name}.build_candidate_models must be callable")
    project_root = next(
        (
            parent
            for parent in source_path.parents
            if (parent / "src/ml_project").is_dir()
        ),
        None,
    )
    if project_root is not None:
        settings = settings_with_card_decision(project_root, settings)
    return ExperimentDefinition(
        module_name=module_name,
        source_path=source_path,
        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        settings=settings,
        prepare_data=prepare_data,
        build_models=build_models,
    )


def resolve_experiment_lineage(
    definition: ExperimentDefinition,
    *,
    require_adopted: bool = True,
) -> tuple[ExperimentDefinition, ...]:
    """Return parents from immediate to oldest and reject broken lineages."""

    lineage: list[ExperimentDefinition] = []
    path = [definition.module_name]
    seen = set(path)
    current = definition
    while current.settings.parent_experiment_module:
        module_name = current.settings.parent_experiment_module
        if module_name in seen:
            chain = " -> ".join([*path, module_name])
            raise ValueError(f"Experiment lineage contains a cycle: {chain}")
        parent = load_experiment(module_name, reload_module=False)
        validate_settings(parent.settings)
        if require_adopted and parent.settings.decision != "adopt":
            raise ValueError(
                f"{current.settings.experiment_id} cannot use "
                f"{parent.settings.experiment_id} as champion: parent decision "
                f"is {parent.settings.decision!r}, expected 'adopt'"
            )
        lineage.append(parent)
        path.append(module_name)
        seen.add(module_name)
        current = parent
    return tuple(lineage)


def validate_settings(settings: ExperimentSettings) -> None:
    errors: list[str] = []
    name_pattern = r"[A-Za-z0-9][A-Za-z0-9._-]*"
    if not re.fullmatch(name_pattern, settings.experiment_id):
        errors.append("EXPERIMENT.experiment_id contains unsupported characters")
    if not re.fullmatch(name_pattern, settings.run_name):
        errors.append("EXPERIMENT.run_name contains unsupported characters")
    for label, value in (
        ("experiment_title", settings.experiment_title),
        ("hypothesis", settings.hypothesis),
        ("change_description", settings.change_description),
        ("success_criterion", settings.success_criterion),
        ("reference_model", settings.reference_model),
        ("primary_candidate", settings.primary_candidate),
    ):
        if not value.strip() or "CHANGE ME" in value.upper():
            errors.append(f"Fill EXPERIMENT.{label} before running the experiment")
    if not np.isfinite(settings.primary_improvement_min):
        errors.append("EXPERIMENT.primary_improvement_min must be finite")
    for metric, threshold in settings.metric_guardrails.items():
        if not str(metric).strip():
            errors.append("EXPERIMENT.metric_guardrails contains an empty metric")
        if not np.isfinite(threshold):
            errors.append(
                f"EXPERIMENT.metric_guardrails[{metric!r}] must be finite"
            )
    if settings.reference_model == settings.primary_candidate:
        errors.append(
            "EXPERIMENT.reference_model and EXPERIMENT.primary_candidate "
            "must be different"
        )
    if (
        settings.parent_experiment_module is not None
        and not settings.parent_experiment_module.startswith(
            EXPERIMENT_MODULE_PREFIX
        )
    ):
        errors.append(
            "EXPERIMENT.parent_experiment_module must start with "
            f"{EXPERIMENT_MODULE_PREFIX!r}"
        )
    if settings.decision not in DECISIONS:
        errors.append(
            "EXPERIMENT.decision must be one of: "
            + ", ".join(sorted(DECISIONS))
        )
    for label, path, suffix in (
        ("experiment_note", settings.experiment_note, ".md"),
        ("results_registry", settings.results_registry, ".csv"),
    ):
        if path.is_absolute():
            errors.append(f"EXPERIMENT.{label} must be relative to the project root")
        if path.suffix.lower() != suffix:
            errors.append(f"EXPERIMENT.{label} must end with {suffix}")
    if settings.artifact_dir.is_absolute():
        errors.append("EXPERIMENT.artifact_dir must be relative to the project root")
    if settings.metric_figure_dpi < 72:
        errors.append("EXPERIMENT.metric_figure_dpi must be at least 72")
    if errors:
        raise ValueError(
            "Invalid selected experiment module:\n"
            + "\n".join(f"- {error}" for error in errors)
        )


def settings_report(settings: ExperimentSettings) -> pd.DataFrame:
    rows = [
        ("identity", "experiment_id", settings.experiment_id),
        ("identity", "experiment_title", settings.experiment_title),
        ("pre-registration", "hypothesis", settings.hypothesis),
        ("pre-registration", "change_description", settings.change_description),
        ("pre-registration", "success_criterion", settings.success_criterion),
        (
            "pre-registration",
            "primary_improvement_min",
            settings.primary_improvement_min,
        ),
        (
            "pre-registration",
            "metric_guardrails",
            settings.metric_guardrails or "none",
        ),
        ("comparison", "reference_model", settings.reference_model),
        (
            "comparison",
            "parent_experiment_module",
            settings.parent_experiment_module or "baseline",
        ),
        ("comparison", "primary_candidate", settings.primary_candidate),
        ("comparison", "decision", settings.decision),
        ("write", "run_name", settings.run_name),
        ("write", "save_artifacts", settings.save_artifacts),
        ("write", "save_metric_figures", settings.save_metric_figures),
        ("write", "save_final_model", settings.save_final_model),
        ("write", "sync_experiment_note", settings.sync_experiment_note),
        ("write", "sync_docs", settings.sync_docs),
        ("write", "allow_overwrite", settings.allow_overwrite),
    ]
    return pd.DataFrame(rows, columns=["section", "parameter", "value"])


def _run_prepare_hook(
    definition: ExperimentDefinition,
    frame: pd.DataFrame,
    feature_groups: Mapping[str, Any],
    settings: ModelingSettings,
) -> ExperimentData:
    prepared = definition.prepare_data(
        frame.copy(deep=True),
        copy.deepcopy(feature_groups),
        settings,
    )
    if not isinstance(prepared, ExperimentData):
        raise TypeError(
            f"{definition.module_name}.prepare_candidate_data must return "
            "ExperimentData"
        )
    return prepared


def prepare_experiment_parent(
    definition: ExperimentDefinition,
    train: pd.DataFrame,
    feature_groups: Mapping[str, Any],
    initial_settings: ModelingSettings,
) -> ExperimentData:
    """Собрать reference из EXP-001 и всех принятых parent-экспериментов."""

    return prepare_reference_experiment_data(
        definition.settings.parent_experiment_module,
        train,
        feature_groups,
        initial_settings,
    )


def prepare_reference_experiment_data(
    parent_experiment_module: str | None,
    train: pd.DataFrame,
    feature_groups: Mapping[str, Any],
    initial_settings: ModelingSettings,
) -> ExperimentData:
    """Вернуть данные и ModelingSettings эффективного reference."""

    if parent_experiment_module is None:
        return ExperimentData(
            frame=train.copy(deep=True),
            feature_groups=copy.deepcopy(feature_groups),
            settings=initial_settings,
            diagnostics={},
        )
    parent = load_experiment(parent_experiment_module, reload_module=False)
    validate_settings(parent.settings)
    if parent.settings.decision != "adopt":
        raise ValueError(
            f"Champion parent {parent.settings.experiment_id} has decision "
            f"{parent.settings.decision!r}; expected 'adopt'"
        )
    context = ExperimentData(
        frame=train.copy(deep=True),
        feature_groups=copy.deepcopy(feature_groups),
        settings=initial_settings,
        diagnostics={},
    )
    stack = [*reversed(resolve_experiment_lineage(parent)), parent]
    for item in stack:
        context = _run_prepare_hook(
            item,
            context.frame,
            context.feature_groups,
            context.settings,
        )
    return context


def prepare_experiment_candidate(
    definition: ExperimentDefinition,
    train: pd.DataFrame,
    feature_groups: Mapping[str, Any],
    initial_settings: ModelingSettings,
) -> ExperimentData:
    """Применить reference-цепочку, затем hook выбранного candidate."""

    parent = prepare_experiment_parent(
        definition,
        train,
        feature_groups,
        initial_settings,
    )
    return _run_prepare_hook(
        definition,
        parent.frame,
        parent.feature_groups,
        parent.settings,
    )


def build_reference_pipeline(
    parent_experiment_module: str | None,
    preprocessor: Any,
    settings: ModelingSettings,
) -> Any:
    """Build baseline or the exact primary candidate of an adopted parent."""

    if parent_experiment_module is None:
        return modeling_tools.build_model_pipeline(
            preprocessor,
            modeling_tools.build_simple_estimator(settings),
        )
    parent = load_experiment(parent_experiment_module, reload_module=False)
    validate_settings(parent.settings)
    if parent.settings.decision != "adopt":
        raise ValueError(
            f"Champion parent {parent.settings.experiment_id} has decision "
            f"{parent.settings.decision!r}; expected 'adopt'"
        )
    resolve_experiment_lineage(parent)
    models = build_experiment_candidates(parent, preprocessor, settings)
    return models[parent.settings.primary_candidate]


def build_experiment_reference(
    definition: ExperimentDefinition,
    preprocessor: Any,
    settings: ModelingSettings,
) -> Any:
    """Build the selected experiment's explicit baseline/champion reference."""

    return build_reference_pipeline(
        definition.settings.parent_experiment_module,
        preprocessor,
        settings,
    )


def build_experiment_candidates(
    definition: ExperimentDefinition,
    preprocessor: Any,
    candidate_settings: ModelingSettings,
) -> dict[str, Any]:
    """Run the experiment-owned model hook and normalize its mapping."""

    models = definition.build_models(
        preprocessor,
        candidate_settings,
        definition.settings,
    )
    if not isinstance(models, Mapping):
        raise TypeError(
            f"{definition.module_name}.build_candidate_models must return a mapping"
        )
    normalized = dict(models)
    validate_model_contract(normalized, definition.settings)
    return normalized


def prepare_experiment_data(
    reference_data: PreparedData,
    experiment_frame: pd.DataFrame,
    *,
    target: str,
) -> PreparedData:
    """Align candidate columns to exactly the rows/target used by the reference."""

    missing_rows = reference_data.row_index.difference(experiment_frame.index)
    if len(missing_rows):
        raise ValueError("experiment_frame lost rows from the reference dataset")
    if target not in experiment_frame:
        raise KeyError(f"TARGET {target!r} is absent from experiment_frame")
    aligned = experiment_frame.loc[reference_data.row_index]
    candidate_y = aligned[target]
    if not np.array_equal(candidate_y.to_numpy(), reference_data.y.to_numpy()):
        raise ValueError(
            "The experiment changed TARGET values or row order. Start a new "
            "validation contract instead of comparing this run to the baseline."
        )
    changed_reference_features = [
        column
        for column in reference_data.X.columns
        if column not in aligned
        or not aligned[column].equals(reference_data.X[column])
    ]
    if changed_reference_features:
        raise ValueError(
            "Do not overwrite raw baseline/champion features in experiment_frame. "
            "Create a new feature or transform it inside the candidate Pipeline: "
            + ", ".join(changed_reference_features)
        )
    return PreparedData(
        X=aligned.drop(columns=[target]),
        y=reference_data.y,
        groups=reference_data.groups,
        row_index=reference_data.row_index,
    )


def validate_model_contract(
    candidate_models: Mapping[str, Any],
    settings: ExperimentSettings,
) -> None:
    if not candidate_models:
        raise ValueError(
            "The candidate_models dictionary is empty. Fill the editable "
            "experiment cell before continuing."
        )
    if settings.reference_model in candidate_models:
        raise ValueError(
            "candidate_models must not overwrite EXPERIMENT.reference_model"
        )
    if settings.primary_candidate not in candidate_models:
        raise ValueError(
            f"EXPERIMENT.primary_candidate={settings.primary_candidate!r} "
            "is absent from "
            "candidate_models"
        )


def comparison_summary(
    evaluation: CVEvaluation,
    scoring: ScoringPlan,
    *,
    reference_model: str,
) -> pd.DataFrame:
    """Add deltas where positive always means improvement over the reference."""

    summary = evaluation.summary[
        evaluation.summary["split"] == "validation"
    ].copy()
    reference = summary[summary["model"] == reference_model][
        ["metric_key", "mean"]
    ].rename(columns={"mean": "reference_mean"})
    if set(reference["metric_key"]) != set(scoring.scorers):
        raise ValueError(f"Reference model {reference_model!r} has incomplete metrics")
    result = summary.merge(reference, on="metric_key", how="left")
    raw_delta = result["mean"] - result["reference_mean"]
    result["improvement"] = np.where(
        result["direction"].eq("maximize"), raw_delta, -raw_delta
    )
    result["mean ± std"] = result.apply(
        lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}", axis=1
    )
    return result


def success_criteria_report(
    evaluation: CVEvaluation,
    scoring: ScoringPlan,
    settings: ExperimentSettings,
) -> pd.DataFrame:
    """Evaluate pre-registered primary and guardrail delta thresholds."""

    comparison = comparison_summary(
        evaluation,
        scoring,
        reference_model=settings.reference_model,
    )
    candidate = comparison[
        comparison["model"].eq(settings.primary_candidate)
    ].copy()
    lookup: dict[str, pd.Series] = {}
    for _, row in candidate.iterrows():
        lookup[str(row["metric_key"]).casefold()] = row
        lookup[str(row["metric"]).casefold()] = row

    primary = lookup.get("primary")
    if primary is None:
        raise ValueError("Primary candidate has no primary metric result")
    rows = [
        {
            "role": "primary",
            "metric": primary["metric"],
            "observed_improvement": float(primary["improvement"]),
            "minimum_improvement": float(settings.primary_improvement_min),
            "passed": bool(
                float(primary["improvement"])
                >= float(settings.primary_improvement_min)
            ),
        }
    ]
    for metric, threshold in settings.metric_guardrails.items():
        record = lookup.get(str(metric).casefold())
        if record is None:
            available = sorted(
                {str(row["metric"]) for _, row in candidate.iterrows()}
            )
            raise ValueError(
                f"Unknown metric guardrail {metric!r}; available: {available}"
            )
        rows.append(
            {
                "role": "guardrail",
                "metric": record["metric"],
                "observed_improvement": float(record["improvement"]),
                "minimum_improvement": float(threshold),
                "passed": bool(float(record["improvement"]) >= float(threshold)),
            }
        )
    return pd.DataFrame(rows)


def _effective_run_settings(
    candidate_settings: ModelingSettings,
    experiment_settings: ExperimentSettings,
) -> ModelingSettings:
    return replace(
        candidate_settings,
        experiment_id=experiment_settings.experiment_id,
        experiment_title=experiment_settings.experiment_title,
        experiment_note=experiment_settings.experiment_note,
        run_name=experiment_settings.run_name,
        artifact_dir=experiment_settings.artifact_dir,
        save_artifacts=experiment_settings.save_artifacts,
        save_metric_figures=experiment_settings.save_metric_figures,
        metric_figure_dpi=experiment_settings.metric_figure_dpi,
        save_final_model=experiment_settings.save_final_model,
        sync_docs=experiment_settings.sync_docs,
        sync_experiment_note=experiment_settings.sync_experiment_note,
        allow_overwrite=experiment_settings.allow_overwrite,
    )


def save_experiment_run(
    project_root: Path,
    experiment_settings: ExperimentSettings,
    candidate_settings: ModelingSettings,
    evaluation: CVEvaluation,
    candidate_plan: FeaturePlan,
    scoring: ScoringPlan,
    *,
    dataset_version: str,
    cv_description: str,
    models: Mapping[str, Any],
    data: PreparedData,
    metric_figures: Mapping[str, Any] | None = None,
    definition: ExperimentDefinition | None = None,
) -> SavedBaselineRun:
    """Сохранить результат с фактическими ModelingSettings candidate."""

    experiment_settings = settings_with_card_decision(
        project_root, experiment_settings
    )

    validate_model_contract(
        {key: value for key, value in models.items() if key != experiment_settings.reference_model},
        experiment_settings,
    )
    effective = _effective_run_settings(candidate_settings, experiment_settings)
    saved = modeling_tools.save_baseline_run(
        project_root,
        effective,
        evaluation,
        candidate_plan,
        scoring,
        dataset_version=dataset_version,
        cv_description=cv_description,
        final_pipeline=models[experiment_settings.primary_candidate],
        data=data,
        metric_figures=metric_figures,
    )
    metadata = json.loads(saved.metadata_path.read_text(encoding="utf-8"))
    criteria = success_criteria_report(evaluation, scoring, experiment_settings)
    metadata.update(
        {
            "run_kind": "controlled_experiment",
            "hypothesis": experiment_settings.hypothesis,
            "change_description": experiment_settings.change_description,
            "success_criterion": experiment_settings.success_criterion,
            "success_criteria": criteria.to_dict(orient="records"),
            "success_criteria_passed": bool(criteria["passed"].all()),
            "reference_model": experiment_settings.reference_model,
            "primary_candidate": experiment_settings.primary_candidate,
            "evaluated_models": list(models),
            "experiment_parameters": dict(experiment_settings.experiment_parameters),
            "decision": experiment_settings.decision,
        }
    )
    if definition is not None:
        metadata.update(
            {
                "implementation_module": definition.module_name,
                "implementation_path": _vault_relative(
                    project_root, definition.source_path
                ),
                "implementation_sha256": definition.source_sha256,
                "experiment_lineage": _lineage_provenance(
                    project_root,
                    definition,
                ),
            }
        )
    baseline_config_path = (
        Path(project_root).resolve() / "src/ml_project/baseline_config.py"
    )
    if baseline_config_path.exists():
        metadata["baseline_config_sha256"] = hashlib.sha256(
            baseline_config_path.read_bytes()
        ).hexdigest()
    saved.metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return saved


def _vault_relative(project_root: Path, path: Path) -> str:
    root = Path(project_root).resolve()
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"Path resolves outside project root: {path}") from error


def _short_version(value: str) -> str:
    return value if len(value) <= 16 else value[:12] + "…"


def _lineage_provenance(
    project_root: Path,
    definition: ExperimentDefinition,
) -> list[dict[str, Any]]:
    definitions = [
        *reversed(resolve_experiment_lineage(definition)),
        definition,
    ]
    return [
        {
            "experiment_id": item.settings.experiment_id,
            "module": item.module_name,
            "path": _vault_relative(project_root, item.source_path),
            "sha256": item.source_sha256,
            "candidate": item.settings.primary_candidate,
            "decision": item.settings.decision,
        }
        for item in definitions
    ]


def _lineage_markdown(
    definition: ExperimentDefinition,
) -> str:
    parents = list(reversed(resolve_experiment_lineage(definition)))
    links = ["EXP-001"]
    links.extend(
        f"[[{parent.settings.experiment_note.as_posix()}|"
        f"{parent.settings.experiment_id}]]"
        for parent in parents
    )
    links.append(definition.settings.experiment_id)
    return " → ".join(links)


def _build_diagnostics_report(
    project_root: Path,
    diagnostics: ExperimentDiagnostics,
    saved: SavedDiagnostics,
) -> str:
    focus = pd.DataFrame(
        [
            {
                "Изменение": "добавлен в candidate",
                "Признак": feature,
            }
            for feature in diagnostics.focus_features
        ]
        + [
            {
                "Изменение": "исключён относительно reference",
                "Признак": feature,
            }
            for feature in diagnostics.removed_features
        ]
    )
    if focus.empty:
        focus = pd.DataFrame(
            [{"Изменение": "feature plan не изменился", "Признак": "—"}]
        )

    paired = (
        diagnostics.paired_fold_deltas.groupby(
            ["metric", "direction"], as_index=False, sort=False
        )
        .agg(
            mean_improvement=("improvement", "mean"),
            std_improvement=("improvement", "std"),
            min_improvement=("improvement", "min"),
            max_improvement=("improvement", "max"),
        )
        .rename(
            columns={
                "metric": "Метрика",
                "direction": "Направление",
                "mean_improvement": "Средний paired Δ",
                "std_improvement": "Std paired Δ",
                "min_improvement": "Min paired Δ",
                "max_improvement": "Max paired Δ",
            }
        )
    )
    changes = diagnostics.prediction_changes.rename(
        columns={
            "outcome_change": "Переход",
            "rows": "Строк",
            "share": "Доля",
        }
    )
    permutation = diagnostics.permutation_importance
    if permutation.empty:
        permutation_summary = pd.DataFrame(
            [{"Признак": "—", "Mean importance": np.nan, "Std": np.nan}]
        )
    else:
        permutation_summary = (
            permutation.groupby("feature", as_index=False)
            .agg(
                mean_importance=("importance_mean", "mean"),
                std_importance=("importance_mean", "std"),
            )
            .sort_values("mean_importance", ascending=False)
            .head(15)
            .rename(
                columns={
                    "feature": "Признак",
                    "mean_importance": "Mean importance",
                    "std_importance": "Std",
                }
            )
        )

    parts = [
        "## Диагностика candidate",
        "> [!info] Граница интерпретации\n> Диагностика использует fitted-модели тех же CV-folds. Importance описывает предсказания модели, а не причинный эффект признака.",
        "### Контролируемое изменение\n\n" + dataframe_to_markdown(focus),
        "### Путь данных по candidate pipeline\n\n"
        + dataframe_to_markdown(
            diagnostics.pipeline_stages.rename(
                columns={
                    "stage": "Этап",
                    "transformer": "Transformer",
                    "rows": "Строк",
                    "columns": "Колонок",
                    "sparse": "Sparse",
                    "density": "Плотность",
                    "missing_cells": "Пропусков",
                }
            ),
            float_digits=4,
        ),
        "### Paired Δ на одинаковых folds\n\n"
        + dataframe_to_markdown(paired, float_digits=4),
        "### Изменение OOF-ошибок\n\n"
        + dataframe_to_markdown(changes, float_digits=4),
        "### Validation permutation importance candidate\n\n"
        + dataframe_to_markdown(permutation_summary, float_digits=4),
    ]
    for figure_path in saved.figure_paths.values():
        parts.append(f"![[{_vault_relative(project_root, figure_path)}]]")
    if diagnostics.warnings:
        parts.append(
            "> [!warning] Ограничения диагностики\n> "
            + "\n> ".join(diagnostics.warnings)
        )
    diagnostic_rel = _vault_relative(project_root, saved.artifact_dir)
    parts.append(
        "### Диагностические таблицы\n\n"
        + "\n".join(
            f"- [[{diagnostic_rel}/{path.name}|{path.name}]]"
            for path in saved.table_paths.values()
        )
    )
    parts.append(
        "Подробный интерактивный разбор: [[notebooks/05_diagnostics.ipynb|05_diagnostics.ipynb]]."
    )
    return "\n\n".join(parts)


def build_experiment_report(
    project_root: Path,
    settings: ExperimentSettings,
    evaluation: CVEvaluation,
    scoring: ScoringPlan,
    saved: SavedBaselineRun,
    *,
    dataset_version: str,
    cv_description: str,
    definition: ExperimentDefinition | None = None,
    diagnostics: ExperimentDiagnostics | None = None,
    saved_diagnostics: SavedDiagnostics | None = None,
) -> str:
    comparison = comparison_summary(
        evaluation, scoring, reference_model=settings.reference_model
    )
    criteria = success_criteria_report(evaluation, scoring, settings)
    overview_rows = [
        ("Эксперимент", f"{settings.experiment_id} — {settings.experiment_title}"),
        ("Гипотеза", settings.hypothesis),
        ("Одно изменение", settings.change_description),
        ("Критерий успеха", settings.success_criterion),
        (
            "Формальные критерии",
            "passed" if bool(criteria["passed"].all()) else "failed",
        ),
        ("Решение", settings.decision),
        ("Run", settings.run_name),
        ("Версия данных", _short_version(dataset_version)),
        ("Validation", cv_description),
        ("Reference", settings.reference_model),
        ("Основной кандидат", settings.primary_candidate),
        ("Основная метрика", scoring.contract_metric),
    ]
    if definition is not None:
        implementation = _vault_relative(project_root, definition.source_path)
        overview_rows.extend(
            [
                ("Цепочка", _lineage_markdown(definition)),
                ("Код эксперимента", f"[[{implementation}|{definition.module_name}]]"),
                ("Hash кода", definition.source_sha256[:12] + "…"),
            ]
        )
    overview = pd.DataFrame(overview_rows, columns=["Поле", "Значение"])
    compact = comparison[
        ["model", "metric", "direction", "mean ± std", "reference_mean", "improvement"]
    ].rename(
        columns={
            "model": "Модель",
            "metric": "Метрика",
            "direction": "Направление",
            "reference_mean": "Reference",
            "improvement": "Δ к reference",
        }
    )
    sections = [
        "## Контракт эксперимента\n\n" + dataframe_to_markdown(overview),
        "> [!note] Как читать Δ\n> Положительное значение означает улучшение — и для maximize, и для minimize-метрик.",
        "## Проверка pre-registered criteria\n\n"
        + dataframe_to_markdown(
            criteria.rename(
                columns={
                    "role": "Роль",
                    "metric": "Метрика",
                    "observed_improvement": "Наблюдаемый Δ",
                    "minimum_improvement": "Минимальный Δ",
                    "passed": "Пройден",
                }
            ),
            float_digits=4,
        ),
        "## Сравнение всех метрик\n\n" + dataframe_to_markdown(compact, float_digits=4),
    ]
    fold_scores = evaluation.fold_scores[evaluation.fold_scores["split"] == "validation"]
    figures = dict(saved.metric_figure_paths or {})
    for metric_key in scoring.scorers:
        label = scoring.labels[metric_key]
        metric_summary = comparison[comparison["metric_key"] == metric_key][
            ["model", "mean", "std", "min", "max", "reference_mean", "improvement"]
        ].rename(
            columns={
                "model": "Модель", "mean": "Mean", "std": "Std",
                "min": "Min", "max": "Max", "reference_mean": "Reference",
                "improvement": "Δ к reference",
            }
        )
        folds = fold_scores[fold_scores["metric_key"] == metric_key].pivot(
            index="fold", columns="model", values="value"
        ).reset_index().rename(columns={"fold": "Fold"})
        parts = [f"## Метрика: {label}"]
        if metric_key in figures:
            parts.append(f"![[{_vault_relative(project_root, figures[metric_key])}]]")
        parts.extend(
            [
                "### Сводка\n\n" + dataframe_to_markdown(metric_summary, float_digits=4),
                "### Значения по folds\n\n" + dataframe_to_markdown(folds, float_digits=4),
            ]
        )
        sections.append("\n\n".join(parts))
    if diagnostics is not None and saved_diagnostics is not None:
        sections.append(
            _build_diagnostics_report(
                project_root,
                diagnostics,
                saved_diagnostics,
            )
        )
    run_rel = _vault_relative(project_root, saved.run_dir)
    artifact_lines = [
        f"- [[{run_rel}/cv_fold_scores.csv|cv_fold_scores.csv]]",
        f"- [[{run_rel}/cv_summary.csv|cv_summary.csv]]",
        f"- [[{run_rel}/metadata.json|metadata.json]]",
    ]
    if saved.model_path is not None:
        artifact_lines.append(f"- [[{run_rel}/model.joblib|model.joblib]]")
    sections.append("## Артефакты\n\n" + "\n".join(artifact_lines))
    return "\n\n".join(sections)


def _update_note_frontmatter(path: Path, fields: Mapping[str, str]) -> None:
    """Update generated scalar fields without touching the note body."""

    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"Experiment note has no YAML frontmatter: {path}")
    closing = text.find("\n---\n", 4)
    if closing < 0:
        raise ValueError(f"Experiment note has invalid YAML frontmatter: {path}")
    frontmatter = text[4:closing]
    for key, value in fields.items():
        rendered = f"{key}: {value}"
        pattern = re.compile(rf"^{re.escape(key)}:.*$", re.MULTILINE)
        if pattern.search(frontmatter):
            frontmatter = pattern.sub(rendered, frontmatter, count=1)
        else:
            frontmatter = frontmatter.rstrip() + "\n" + rendered
    path.write_text(
        "---\n" + frontmatter.rstrip() + text[closing:], encoding="utf-8"
    )


def sync_experiment_note(
    project_root: Path,
    settings: ExperimentSettings,
    evaluation: CVEvaluation,
    scoring: ScoringPlan,
    saved: SavedBaselineRun,
    *,
    dataset_version: str,
    cv_description: str,
    definition: ExperimentDefinition | None = None,
    diagnostics: ExperimentDiagnostics | None = None,
    saved_diagnostics: SavedDiagnostics | None = None,
) -> list[str]:
    root = Path(project_root).resolve()
    settings = settings_with_card_decision(root, settings)
    note_path = (root / settings.experiment_note).resolve()
    _vault_relative(root, note_path)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    if not note_path.exists():
        yaml_string = lambda value: json.dumps(value, ensure_ascii=False)
        note_path.write_text(
            "---\n"
            f"id: {settings.experiment_id}\n"
            "type: experiment\n"
            "experiment_type: hypothesis-test\n"
            "status: completed\n"
            f"created: {date.today().isoformat()}\n"
            f"hypothesis: {yaml_string(settings.hypothesis)}\n"
            f"primary_metric: {yaml_string(scoring.contract_metric)}\n"
            f"decision: {settings.decision}\n"
            "eda_findings: []\n"
            "---\n\n"
            f"# {settings.experiment_id} — {settings.experiment_title}\n\n"
            "← [[experiments/_index.md|Реестр]] · [[docs/05_experiments.md|Эксперименты]]\n\n"
            "> [!info] Автоматическая часть\n"
            "> Повторный запуск заменяет только отчёт между маркерами. Ручной анализ ниже сохраняется.\n\n"
            "<!-- auto:experiment-report:start -->\n\n"
            "Отчёт появится после сохранения запуска.\n\n"
            "<!-- auto:experiment-report:end -->\n\n"
            "## EDA-основания\n\n"
            "<!-- auto:experiment-eda-links:start -->\n\n"
            "> Добавьте EDA ID в `eda_findings` во frontmatter карточки.\n\n"
            "<!-- auto:experiment-eda-links:end -->\n\n"
            "## Анализ результата — заполнить вручную\n\n"
            "- **Что произошло:**\n"
            "- **Подтвердилась ли гипотеза:**\n"
            "- **Почему мог получиться такой результат:**\n"
            "- **Стабильность по folds / seeds:**\n"
            "- **Ограничения и возможный leakage:**\n\n"
            "## Обоснование решения — заполнить вручную\n\n"
            "> Source of truth для `decision` — поле во frontmatter этой "
            "карточки. После изменения запустите "
            "`sync-experiment-state.cmd`; переобучение не требуется.\n\n"
            "- **Почему выбрано это решение:**\n"
            "- **Следующий шаг:**\n",
            encoding="utf-8",
        )
    _update_note_frontmatter(
        note_path,
        {
            "status": "completed",
            "hypothesis": json.dumps(settings.hypothesis, ensure_ascii=False),
            "primary_metric": json.dumps(scoring.contract_metric, ensure_ascii=False),
        },
    )
    report = build_experiment_report(
        root, settings, evaluation, scoring, saved,
        dataset_version=dataset_version, cv_description=cv_description,
        definition=definition,
        diagnostics=diagnostics,
        saved_diagnostics=saved_diagnostics,
    )
    updated = MarkdownDocument(note_path).update_blocks(
        {"experiment-report": report}
    )
    sync_experiment_eda_relations(root)
    return updated


def _registry_row(
    project_root: Path,
    settings: ExperimentSettings,
    evaluation: CVEvaluation,
    scoring: ScoringPlan,
    *,
    dataset_version: str,
    definition: ExperimentDefinition | None = None,
) -> dict[str, Any]:
    comparison = comparison_summary(
        evaluation, scoring, reference_model=settings.reference_model
    )
    row = comparison[
        (comparison["metric_key"] == "primary")
        & (comparison["model"] == settings.primary_candidate)
    ]
    if row.empty:
        raise ValueError("Primary candidate has no primary metric result")
    record = row.iloc[0]
    reference_record = evaluation.primary_summary()
    reference_record = reference_record[
        reference_record["model"].eq(settings.reference_model)
    ]
    if reference_record.empty:
        raise ValueError("Reference model has no primary metric result")
    reference_primary = reference_record.iloc[0]
    criteria = success_criteria_report(evaluation, scoring, settings)
    result = {
        "experiment_id": settings.experiment_id,
        "title": settings.experiment_title,
        "note": settings.experiment_note.as_posix(),
        "hypothesis": settings.hypothesis,
        "change": settings.change_description,
        "dataset_version": dataset_version,
        "run_name": settings.run_name,
        "candidate": settings.primary_candidate,
        "primary_metric": scoring.contract_metric,
        "direction": record["direction"],
        "reference_score": float(record["reference_mean"]),
        "reference_std": float(reference_primary["std"]),
        "candidate_score": float(record["mean"]),
        "candidate_std": float(record["std"]),
        "improvement": float(record["improvement"]),
        "criteria_passed": bool(criteria["passed"].all()),
        "primary_improvement_min": float(settings.primary_improvement_min),
        "metric_guardrails": json.dumps(
            dict(settings.metric_guardrails),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "decision": settings.decision,
        "parent_experiment_id": "EXP-001",
        "parent_experiment_module": "",
        "parent_implementation_sha256": "",
    }
    if definition is not None:
        lineage = resolve_experiment_lineage(definition)
        if lineage:
            parent = lineage[0]
            result.update(
                {
                    "parent_experiment_id": parent.settings.experiment_id,
                    "parent_experiment_module": parent.module_name,
                    "parent_implementation_sha256": parent.source_sha256,
                }
            )
        result.update(
            {
                "implementation_module": definition.module_name,
                "implementation_path": _vault_relative(
                    project_root, definition.source_path
                ),
                "implementation_sha256": definition.source_sha256,
            }
        )
    return result


def sync_experiment_docs(
    project_root: Path,
    settings: ExperimentSettings,
    evaluation: CVEvaluation,
    scoring: ScoringPlan,
    *,
    dataset_version: str,
    initial_settings: ModelingSettings | None = None,
    definition: ExperimentDefinition | None = None,
) -> dict[str, Any]:
    """Upsert the registry and refresh experiment and README leaderboards."""

    root = Path(project_root).resolve()
    settings = settings_with_card_decision(root, settings)
    registry_path = (root / settings.results_registry).resolve()
    _vault_relative(root, registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    row = _registry_row(
        root,
        settings,
        evaluation,
        scoring,
        dataset_version=dataset_version,
        definition=definition,
    )
    if registry_path.exists():
        registry = pd.read_csv(registry_path)
        registry = registry[
            ~(
                registry["experiment_id"].astype(str).eq(settings.experiment_id)
                & registry["run_name"].astype(str).eq(settings.run_name)
            )
        ]
    else:
        registry = pd.DataFrame()
    registry = pd.concat([registry, pd.DataFrame([row])], ignore_index=True)
    registry.to_csv(registry_path, index=False)

    latest = pd.DataFrame(
        [
            ("Эксперимент", f"[[{row['note']}|{row['experiment_id']} — {row['title']}]]"),
            ("Родитель", row["parent_experiment_id"]),
            ("Гипотеза", row["hypothesis"]),
            ("Изменение", row["change"]),
            ("Метрика", row["primary_metric"]),
            ("Reference", f"{row['reference_score']:.4f}"),
            ("Кандидат", f"{row['candidate_score']:.4f}"),
            ("Δ к reference", f"{row['improvement']:+.4f}"),
            (
                "Формальные критерии",
                "passed" if bool(row["criteria_passed"]) else "failed",
            ),
            ("Решение", row["decision"]),
        ],
        columns=["Поле", "Значение"],
    )
    leaderboard = registry.copy()
    leaderboard["Experiment"] = leaderboard.apply(
        lambda item: f"[[{item['note']}|{item['experiment_id']}]]", axis=1
    )
    leaderboard = leaderboard[
        [
            "Experiment",
            "parent_experiment_id",
            "hypothesis",
            "change",
            "primary_metric",
            "reference_score",
            "candidate_score",
            "improvement",
            "criteria_passed",
            "decision",
        ]
    ].rename(
        columns={
            "hypothesis": "Hypothesis", "change": "Change",
            "parent_experiment_id": "Parent",
            "primary_metric": "Metric", "reference_score": "Reference",
            "candidate_score": "Result", "improvement": "Δ",
            "criteria_passed": "Criteria",
            "decision": "Decision",
        }
    )
    baseline_note = (
        initial_settings.experiment_note
        if initial_settings is not None
        else Path("experiments/EXP-001 Baseline.md")
    )
    baseline_rows = registry[
        registry["parent_experiment_id"].astype(str).eq("EXP-001")
    ]
    baseline_record = (
        baseline_rows.sort_values("experiment_id").iloc[0]
        if not baseline_rows.empty
        else registry.sort_values("experiment_id").iloc[0]
    )
    baseline_score = float(baseline_record["reference_score"])
    baseline_std = float(baseline_record["reference_std"])
    best_measured = modeling_tools.build_best_result_block(
        baseline_metric=str(row["primary_metric"]),
        baseline_score=baseline_score,
        baseline_std=baseline_std,
        baseline_direction=str(row["direction"]),
        baseline_note=baseline_note,
        experiments=registry,
    )
    blocks = MarkdownDocument(root / "docs/05_experiments.md").update_blocks(
        {
            "latest-experiment": dataframe_to_markdown(latest, float_digits=4),
            "experiment-leaderboard": dataframe_to_markdown(leaderboard, float_digits=4),
            "best-measured-result": best_measured,
        }
    )
    registry_blocks = MarkdownDocument(root / "experiments/_index.md").update_blocks(
        {
            "experiment-registry": modeling_tools.build_experiment_registry_block(
                baseline_note,
                registry,
                baseline_run=(
                    initial_settings.run_name
                    if initial_settings is not None
                    else "baseline"
                ),
                baseline_metric=str(row["primary_metric"]),
                baseline_score=baseline_score,
            )
        }
    )
    readme_blocks = MarkdownDocument(root / "README.md").update_blocks(
        {
            "key-results": modeling_tools.build_key_results_block(
                baseline_metric=str(row["primary_metric"]),
                baseline_score=baseline_score,
                baseline_note=baseline_note,
                experiments=registry,
            )
        }
    )
    state = sync_experiment_state(root)
    return {
        "registry": settings.results_registry.as_posix(),
        "blocks": blocks,
        "registry_blocks": registry_blocks,
        "readme_blocks": readme_blocks,
        "state": state,
    }


__all__ = [
    "ExperimentData",
    "ExperimentDefinition",
    "ExperimentSettings",
    "build_experiment_candidates",
    "build_experiment_reference",
    "build_experiment_report",
    "build_reference_pipeline",
    "comparison_summary",
    "load_experiment",
    "prepare_experiment_candidate",
    "prepare_experiment_data",
    "prepare_experiment_parent",
    "prepare_reference_experiment_data",
    "resolve_experiment_lineage",
    "save_experiment_run",
    "settings_report",
    "success_criteria_report",
    "sync_experiment_docs",
    "sync_experiment_eda_relations",
    "sync_experiment_note",
    "sync_experiment_state",
    "validate_model_contract",
    "validate_settings",
]
