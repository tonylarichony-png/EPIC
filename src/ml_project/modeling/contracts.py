"""Stable data contracts shared by baseline and controlled experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from ._utils import _display_value


CLASSIFICATION_TASKS = {
    "binary_classification",
    "multiclass_classification",
}
SUPPORTED_TASKS = {*CLASSIFICATION_TASKS, "regression"}
SUPPORTED_CV_STRATEGIES = {
    "auto",
    "stratified_kfold",
    "kfold",
    "group_kfold",
    "time_series",
}
SUPPORTED_MODEL_FEATURE_GROUPS = {
    "numeric",
    "count",
    "categorical",
    "ordinal",
}
TRACKED_METRIC_FIGURE_ROOT = Path("assets/experiments")


@dataclass(frozen=True)
class ModelingSettings:
    """Настройки подготовки, обучения, валидации и сохранения модели.

    Это нейтральный контракт: объект может описывать исходный baseline,
    текущий reference-чемпион или candidate после контролируемого изменения.
    Роль конкретного экземпляра определяется именем переменной
    ``reference_settings`` / ``candidate_settings``, а не типом.
    """

    # Задача и состав признаков, которые разрешено передать модели.
    task_type: str | None
    model_feature_groups: tuple[str, ...]
    include_features: tuple[str, ...]
    exclude_features: tuple[str, ...]
    require_inference_features: bool

    # Метрики, схема cross-validation и воспроизводимость вычислений.
    primary_scorer: Any
    secondary_scorers: Mapping[str, Any]
    cv_strategy: str
    n_splits: int
    shuffle: bool
    random_state: int | None
    group_column: str | None
    time_column: str | None
    n_jobs: int | None
    error_score: str | float
    return_train_score: bool

    # Fold-safe preprocessing: параметры обучаются отдельно внутри каждого fold.
    numeric_imputer: str
    numeric_fill_value: Any
    add_numeric_missing_indicator: bool
    numeric_scaler: str
    categorical_imputer: str
    categorical_fill_value: Any
    onehot_handle_unknown: str
    onehot_min_frequency: int | float | None
    onehot_max_categories: int | None
    onehot_sparse_output: bool
    column_transformer_sparse_threshold: float

    # Reference estimator и no-skill модель для проверки здравого смысла.
    run_dummy_baseline: bool
    dummy_strategy: str
    dummy_params: Mapping[str, Any]
    model_name: str
    model_params: Mapping[str, Any]

    # Идентичность запуска и управляемые действия записи результатов.
    experiment_id: str
    experiment_title: str
    experiment_note: Path
    run_name: str
    artifact_dir: Path
    save_artifacts: bool
    save_metric_figures: bool
    metric_figure_dpi: int
    save_final_model: bool
    sync_docs: bool
    sync_experiment_note: bool
    allow_overwrite: bool


# Временная обратная совместимость со старыми модулями и сохранёнными объектами.
# Новый код должен импортировать ModelingSettings.
BaselineSettings = ModelingSettings


@dataclass(frozen=True)
class ExperimentSettings:
    """Описание гипотезы, сравнения, решения и provenance эксперимента."""

    experiment_id: str
    experiment_title: str
    experiment_note: Path
    hypothesis: str
    change_description: str
    success_criterion: str
    primary_improvement_min: float
    metric_guardrails: Mapping[str, float]
    reference_model: str
    primary_candidate: str
    experiment_parameters: Mapping[str, Any]
    run_name: str
    artifact_dir: Path
    results_registry: Path
    save_artifacts: bool
    save_metric_figures: bool
    metric_figure_dpi: int
    save_final_model: bool
    sync_experiment_note: bool
    sync_docs: bool
    allow_overwrite: bool
    parent_experiment_module: str | None = None
    # Mutable lifecycle state is read from the Markdown experiment card.
    # This fallback is used only before the card exists (the first run).
    decision: str = "pending"


@dataclass(frozen=True)
class ExperimentData:
    """Данные candidate и его эффективные ModelingSettings."""

    frame: pd.DataFrame
    feature_groups: Mapping[str, Sequence[str]]
    settings: ModelingSettings
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentDefinition:
    """Loaded immutable experiment module plus its source provenance."""

    module_name: str
    source_path: Path
    source_sha256: str
    settings: ExperimentSettings
    prepare_data: Callable[..., ExperimentData]
    build_models: Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class FeaturePlan:
    """Explicit model columns and the reason every other column is excluded."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    excluded: tuple[str, ...]
    group_by_feature: Mapping[str, str]
    exclusion_reason: Mapping[str, str]
    warnings: tuple[str, ...] = ()

    @property
    def model_features(self) -> tuple[str, ...]:
        return (*self.numeric, *self.categorical)

    def to_frame(self) -> pd.DataFrame:
        rows: list[dict[str, str]] = []
        selected = set(self.model_features)
        for feature, group in self.group_by_feature.items():
            if feature in selected:
                role = "numeric" if feature in self.numeric else "categorical"
                rows.append(
                    {
                        "feature": feature,
                        "config_group": group,
                        "model_role": role,
                        "status": "used",
                        "reason": "",
                    }
                )
            else:
                rows.append(
                    {
                        "feature": feature,
                        "config_group": group,
                        "model_role": "—",
                        "status": "excluded",
                        "reason": self.exclusion_reason.get(
                            feature, "not selected for the baseline"
                        ),
                    }
                )
        return pd.DataFrame(rows)


@dataclass(frozen=True)
class PreparedData:
    """Train matrix, target and optional split-control groups."""

    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series | None
    row_index: pd.Index


@dataclass(frozen=True)
class ScoringPlan:
    """Contract metric plus sklearn scoring implementations."""

    contract_metric: str
    scorers: Mapping[str, Any]
    labels: Mapping[str, str]
    directions: Mapping[str, str]
    negated: Mapping[str, bool]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "role": "primary" if key == "primary" else "secondary",
                    "report_name": self.labels[key],
                    "sklearn_scorer": _display_value(scorer),
                    "direction": self.directions[key],
                }
                for key, scorer in self.scorers.items()
            ]
        )


@dataclass(frozen=True)
class CVEvaluation:
    """Fold-level and aggregated results for comparable model pipelines."""

    fold_scores: pd.DataFrame
    summary: pd.DataFrame
    raw_results: Mapping[str, Mapping[str, Any]]
    cv_splits: tuple[tuple[Any, Any], ...] = ()

    def primary_summary(self) -> pd.DataFrame:
        return self.summary[
            (self.summary["metric_key"] == "primary")
            & (self.summary["split"] == "validation")
        ].reset_index(drop=True)


@dataclass(frozen=True)
class SavedBaselineRun:
    """Paths created by an explicit baseline save action."""

    run_dir: Path
    fold_scores_path: Path
    summary_path: Path
    metadata_path: Path
    model_path: Path | None = None
    metric_figure_paths: Mapping[str, Path] | None = None


@dataclass(frozen=True)
class ScreeningModelSpec:
    """Одна модель и её явно зафиксированные стартовые параметры."""

    model_id: str
    label: str
    params: Mapping[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass(frozen=True)
class ModelGroupSettings:
    """Группа моделей, сравнимая в одном screening-прогоне."""

    group_id: str
    title: str
    preprocessing_profile: str
    models: tuple[ScreeningModelSpec, ...]
    purpose: str = ""


@dataclass(frozen=True)
class ModelScreeningSettings:
    """Notebook-first контракт группового выбора семейства моделей."""

    screening_id: str
    screening_title: str
    screening_note: Path
    feature_reference_module: str | None
    active_group: str
    groups: Mapping[str, ModelGroupSettings]
    reference_model_id: str
    diagnostic_model_id: str | None
    shortlist_size: int
    run_name: str
    artifact_dir: Path
    results_registry: Path
    save_artifacts: bool
    save_figures: bool
    figure_dpi: int
    sync_screening_note: bool
    sync_registry: bool
    allow_overwrite: bool


@dataclass(frozen=True)
class ModelScreeningContext:
    """Замороженный feature set и pipeline-чемпион для screening."""

    feature_reference_id: str
    feature_reference_module: str | None
    feature_reference_sha256: str | None
    frame: pd.DataFrame
    feature_groups: Mapping[str, Sequence[str]]
    settings: ModelingSettings
    plan: FeaturePlan
    data: PreparedData
    reference_pipeline: Any


@dataclass(frozen=True)
class BuiltModelGroup:
    """Собранные pipelines одной группы до запуска cross-validation."""

    group: ModelGroupSettings
    context: ModelScreeningContext
    effective_settings: ModelingSettings
    models: Mapping[str, Any]
    parameters: pd.DataFrame


@dataclass(frozen=True)
class ModelScreeningResult:
    """Таблицы и fitted-fold результаты одного группового screening."""

    built: BuiltModelGroup
    scoring: ScoringPlan
    cv_description: str
    evaluation: CVEvaluation
    leaderboard: pd.DataFrame
    paired_deltas: pd.DataFrame
    oof_predictions: pd.DataFrame
    prediction_comparison: pd.DataFrame
    feature_importance: pd.DataFrame
    shortlist: tuple[str, ...]


@dataclass(frozen=True)
class SavedModelScreening:
    """Пути локальных таблиц и Git-tracked карточки/графиков screening."""

    run_dir: Path
    figure_dir: Path
    note_path: Path
    fold_scores_path: Path
    summary_path: Path
    leaderboard_path: Path
    metadata_path: Path
    registry_path: Path
    figure_paths: Mapping[str, Path]
