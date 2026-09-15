"""Validate and display typed modeling settings."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..profiling import FEATURE_GROUP_NAMES
from .contracts import (
    ModelingSettings,
    CLASSIFICATION_TASKS,
    SUPPORTED_CV_STRATEGIES,
    SUPPORTED_MODEL_FEATURE_GROUPS,
    SUPPORTED_TASKS,
    TRACKED_METRIC_FIGURE_ROOT,
)


def validate_modeling_settings(settings: ModelingSettings) -> list[str]:
    """Проверить настройки baseline, reference или candidate модели."""

    errors: list[str] = []
    warnings: list[str] = []

    if settings.task_type not in SUPPORTED_TASKS:
        errors.append(
            "settings.task_type must be one of: "
            + ", ".join(sorted(SUPPORTED_TASKS))
        )
    if settings.cv_strategy not in SUPPORTED_CV_STRATEGIES:
        errors.append(
            "settings.cv_strategy must be one of: "
            + ", ".join(sorted(SUPPORTED_CV_STRATEGIES))
        )
    if settings.n_splits < 2:
        errors.append("settings.n_splits must be at least 2")
    if settings.n_jobs == 0:
        errors.append("settings.n_jobs cannot be 0")
    if settings.cv_strategy == "group_kfold" and not settings.group_column:
        errors.append("settings.group_column is required for group_kfold")
    if settings.cv_strategy == "time_series" and not settings.time_column:
        errors.append("settings.time_column is required for time_series")
    if settings.cv_strategy == "auto":
        warnings.append(
            "CV_STRATEGY='auto' удобна для первого запуска; после описания "
            "production-сценария зафиксируйте явную стратегию."
        )

    unknown_feature_groups = sorted(
        set(settings.model_feature_groups) - set(FEATURE_GROUP_NAMES)
    )
    unsupported_feature_groups = sorted(
        set(settings.model_feature_groups) - SUPPORTED_MODEL_FEATURE_GROUPS
    )
    if unknown_feature_groups:
        errors.append(
            "Unknown settings.model_feature_groups: "
            + ", ".join(unknown_feature_groups)
        )
    if unsupported_feature_groups:
        errors.append(
            "The first tabular baseline has no transformer for groups: "
            + ", ".join(unsupported_feature_groups)
        )
    overlap = sorted(set(settings.include_features) & set(settings.exclude_features))
    if overlap:
        errors.append(
            "Features cannot be both included and excluded: " + ", ".join(overlap)
        )

    if settings.numeric_imputer not in {
        "median",
        "mean",
        "most_frequent",
        "constant",
    }:
        errors.append("Unsupported settings.numeric_imputer")
    if settings.numeric_scaler not in {"standard", "robust", "minmax", "none"}:
        errors.append("Unsupported settings.numeric_scaler")
    if settings.categorical_imputer not in {"most_frequent", "constant"}:
        errors.append("Unsupported settings.categorical_imputer")
    if not 0 <= settings.column_transformer_sparse_threshold <= 1:
        errors.append(
            "settings.column_transformer_sparse_threshold must be between 0 and 1"
        )

    if settings.model_name not in {"auto", "logistic_regression", "ridge"}:
        errors.append(
            "settings.model_name must be auto, logistic_regression or ridge for the "
            "reference estimator"
        )
    if (
        settings.task_type in CLASSIFICATION_TASKS
        and settings.model_name == "ridge"
    ):
        errors.append("ridge is a regression baseline, not a classifier")
    if settings.task_type == "regression" and settings.model_name == "logistic_regression":
        errors.append("logistic_regression cannot be used for regression")

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", settings.run_name):
        errors.append(
            "settings.run_name must contain only letters, digits, dot, "
            "underscore or dash"
        )
    if settings.artifact_dir.is_absolute():
        errors.append("settings.artifact_dir must be relative to the project root")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", settings.experiment_id):
        errors.append(
            "settings.experiment_id must contain only letters, digits, dot, "
            "underscore or dash"
        )
    if not settings.experiment_title.strip():
        errors.append("settings.experiment_title cannot be empty")
    if settings.experiment_note.is_absolute():
        errors.append("settings.experiment_note must be relative to the project root")
    if settings.experiment_note.suffix.lower() != ".md":
        errors.append("settings.experiment_note must point to a Markdown (.md) file")
    if settings.metric_figure_dpi < 72:
        errors.append("settings.metric_figure_dpi must be at least 72")

    if errors:
        raise ValueError(
            "Invalid ModelingSettings:\n"
            + "\n".join(f"- {error}" for error in errors)
        )
    return warnings


# Совместимость со старым публичным API. Новый код вызывает
# validate_modeling_settings независимо от роли объекта.
validate_baseline_settings = validate_modeling_settings


def settings_report(settings: ModelingSettings) -> pd.DataFrame:
    """Compact, human-readable report of the settings that change results."""

    rows = [
        ("task", "task_type", settings.task_type),
        ("features", "model_feature_groups", settings.model_feature_groups),
        ("features", "include_features", settings.include_features or "all"),
        ("features", "exclude_features", settings.exclude_features or "none"),
        ("metric", "primary_scorer", settings.primary_scorer or "from 00_problem"),
        ("metric", "secondary_scorers", settings.secondary_scorers or "none"),
        ("validation", "cv_strategy", settings.cv_strategy),
        ("validation", "n_splits", settings.n_splits),
        ("validation", "shuffle", settings.shuffle),
        ("validation", "random_state", settings.random_state),
        ("preprocessing", "numeric_imputer", settings.numeric_imputer),
        ("preprocessing", "numeric_scaler", settings.numeric_scaler),
        ("preprocessing", "categorical_imputer", settings.categorical_imputer),
        ("preprocessing", "onehot_handle_unknown", settings.onehot_handle_unknown),
        ("model", "run_dummy_baseline", settings.run_dummy_baseline),
        ("model", "model_name", settings.model_name),
        ("model", "model_params", settings.model_params or "defaults"),
        ("write", "experiment_id", settings.experiment_id),
        ("write", "experiment_note", settings.experiment_note),
        ("write", "run_name", settings.run_name),
        (
            "write",
            "metric_figure_dir",
            TRACKED_METRIC_FIGURE_ROOT / settings.experiment_id,
        ),
        ("write", "save_artifacts", settings.save_artifacts),
        ("write", "save_metric_figures", settings.save_metric_figures),
        ("write", "save_final_model", settings.save_final_model),
        ("write", "sync_docs", settings.sync_docs),
        ("write", "sync_experiment_note", settings.sync_experiment_note),
        ("write", "allow_overwrite", settings.allow_overwrite),
    ]
    return pd.DataFrame(rows, columns=["section", "parameter", "value"])
