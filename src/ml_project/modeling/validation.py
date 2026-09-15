"""Cross-validation, scorer contracts and comparable model evaluation."""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ._utils import _sklearn_import_error
from .contracts import (
    ModelingSettings,
    CLASSIFICATION_TASKS,
    CVEvaluation,
    PreparedData,
    ScoringPlan,
)
def resolve_cv_strategy(settings: ModelingSettings) -> str:
    """Resolve the convenient ``auto`` strategy deterministically."""

    if settings.cv_strategy != "auto":
        return settings.cv_strategy
    return (
        "stratified_kfold"
        if settings.task_type in CLASSIFICATION_TASKS
        else "kfold"
    )



def build_cv_splitter(settings: ModelingSettings, y: pd.Series) -> tuple[Any, str]:
    """Construct the configured sklearn splitter."""

    try:
        from sklearn.model_selection import (
            GroupKFold,
            KFold,
            StratifiedKFold,
            TimeSeriesSplit,
        )
    except ImportError as error:  # pragma: no cover - environment dependent
        raise _sklearn_import_error() from error

    strategy = resolve_cv_strategy(settings)
    if strategy == "stratified_kfold":
        class_counts = y.value_counts()
        if class_counts.empty or int(class_counts.min()) < settings.n_splits:
            raise ValueError("Each class must contain at least N_SPLITS rows")
        splitter = StratifiedKFold(
            n_splits=settings.n_splits,
            shuffle=settings.shuffle,
            random_state=settings.random_state if settings.shuffle else None,
        )
    elif strategy == "kfold":
        splitter = KFold(
            n_splits=settings.n_splits,
            shuffle=settings.shuffle,
            random_state=settings.random_state if settings.shuffle else None,
        )
    elif strategy == "group_kfold":
        splitter = GroupKFold(n_splits=settings.n_splits)
    elif strategy == "time_series":
        splitter = TimeSeriesSplit(n_splits=settings.n_splits)
    else:  # guarded by validate_modeling_settings
        raise ValueError(f"Unsupported CV strategy: {strategy}")
    return splitter, strategy


def cv_protocol_description(settings: ModelingSettings, strategy: str) -> str:
    """Create a compact protocol label for notebook and Markdown reports."""

    if strategy in {"group_kfold", "time_series"}:
        detail = (
            f", group={settings.group_column}"
            if strategy == "group_kfold"
            else f", time={settings.time_column}"
        )
        return f"{strategy}(n_splits={settings.n_splits}{detail})"
    return (
        f"{strategy}(n_splits={settings.n_splits}, "
        f"shuffle={settings.shuffle}, seed={settings.random_state})"
    )


def read_inline_field(path: Path, field: str) -> str:
    """Read one Dataview inline field such as ``(primary_metric:: accuracy)``."""

    text = Path(path).read_text(encoding="utf-8")
    pattern = re.compile(
        rf"[\[(]\s*{re.escape(field)}\s*::\s*(.*?)[\])]",
        flags=re.IGNORECASE,
    )
    matches = [match.strip() for match in pattern.findall(text)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one ({field}:: value) field in {path}; "
            f"found {len(matches)}"
        )
    value = matches[0].strip().strip("`")
    if not value or value.upper() in {"TBD", "TODO", "NONE", "N/A"}:
        raise ValueError(f"Fill ({field}:: ...) in {path} before baseline")
    return value


def resolve_scoring_plan(
    project_root: Path,
    settings: ModelingSettings,
    *,
    problem_doc: Path = Path("docs/00_problem.md"),
) -> ScoringPlan:
    """Resolve the Problem metric contract into validated sklearn scorers."""

    try:
        from sklearn.metrics import get_scorer
    except ImportError as error:  # pragma: no cover - environment dependent
        raise _sklearn_import_error() from error

    contract = read_inline_field(Path(project_root) / problem_doc, "primary_metric")
    primary = settings.primary_scorer or _metric_alias(contract)
    scorer_items: list[tuple[str, str, Any]] = [("primary", contract, primary)]
    scorer_items.extend(
        (f"secondary_{index}", str(label), scorer)
        for index, (label, scorer) in enumerate(
            settings.secondary_scorers.items(), start=1
        )
    )

    scorers: dict[str, Any] = {}
    labels: dict[str, str] = {}
    directions: dict[str, str] = {}
    negated: dict[str, bool] = {}
    for key, label, scorer in scorer_items:
        if isinstance(scorer, str):
            try:
                get_scorer(scorer)
            except ValueError as error:
                raise ValueError(
                    f"{scorer!r} is not a valid scikit-learn scorer. "
                    "Use settings.primary_scorer/secondary_scorers for the technical "
                    "implementation while keeping the metric contract in "
                    "docs/00_problem.md."
                ) from error
        elif not callable(scorer):
            raise TypeError(f"Scorer for {label!r} must be a name or callable")
        is_negated = isinstance(scorer, str) and scorer.startswith("neg_")
        scorers[key] = scorer
        labels[key] = label
        directions[key] = "minimize" if is_negated else "maximize"
        negated[key] = is_negated

    return ScoringPlan(
        contract_metric=contract,
        scorers=scorers,
        labels=labels,
        directions=directions,
        negated=negated,
    )



def evaluate_models_cv(
    models: Mapping[str, Any],
    data: PreparedData,
    *,
    cv: Any,
    scoring: ScoringPlan,
    settings: ModelingSettings,
) -> CVEvaluation:
    """Evaluate every full pipeline on the same folds and scoring contract."""

    try:
        from sklearn.model_selection import cross_validate
    except ImportError as error:  # pragma: no cover - environment dependent
        raise _sklearn_import_error() from error

    if not models:
        raise ValueError("At least one model pipeline is required")

    if hasattr(cv, "split"):
        split_iterator = cv.split(
            data.X,
            data.y,
            groups=data.groups,
        )
    else:
        split_iterator = iter(cv)
    cv_splits = tuple(
        (
            np.asarray(train_indices, dtype=int),
            np.asarray(validation_indices, dtype=int),
        )
        for train_indices, validation_indices in split_iterator
    )
    if not cv_splits:
        raise ValueError("Cross-validation produced no folds")

    fold_rows: list[dict[str, Any]] = []
    raw_results: dict[str, Mapping[str, Any]] = {}
    for model_name, pipeline in models.items():
        result = cross_validate(
            pipeline,
            data.X,
            data.y,
            groups=data.groups,
            cv=cv_splits,
            scoring=dict(scoring.scorers),
            n_jobs=settings.n_jobs,
            return_estimator=True,
            return_train_score=settings.return_train_score,
            error_score=settings.error_score,
        )
        raw_results[model_name] = result
        n_folds = len(result["fit_time"])
        for metric_key in scoring.scorers:
            for split, result_prefix in (("validation", "test"), ("train", "train")):
                score_key = f"{result_prefix}_{metric_key}"
                if score_key not in result:
                    continue
                values = np.asarray(result[score_key], dtype=float)
                if scoring.negated[metric_key]:
                    values = -values
                for fold_index in range(n_folds):
                    fold_rows.append(
                        {
                            "model": model_name,
                            "fold": fold_index + 1,
                            "split": split,
                            "metric_key": metric_key,
                            "metric": scoring.labels[metric_key],
                            "direction": scoring.directions[metric_key],
                            "value": float(values[fold_index]),
                            "fit_seconds": float(result["fit_time"][fold_index]),
                            "score_seconds": float(result["score_time"][fold_index]),
                        }
                    )

    fold_scores = pd.DataFrame(fold_rows)
    summary = (
        fold_scores.groupby(
            ["model", "split", "metric_key", "metric", "direction"],
            sort=False,
            as_index=False,
        )
        .agg(
            mean=("value", "mean"),
            std=("value", "std"),
            min=("value", "min"),
            max=("value", "max"),
            folds=("value", "size"),
            fit_seconds_mean=("fit_seconds", "mean"),
            score_seconds_mean=("score_seconds", "mean"),
        )
    )
    return CVEvaluation(
        fold_scores=fold_scores,
        summary=summary,
        raw_results=raw_results,
        cv_splits=cv_splits,
    )



def _metric_alias(metric: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", metric.lower()).strip("_")
    aliases = {
        "acc": "accuracy",
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_score": "f1",
        "f1": "f1",
        "f1_macro": "f1_macro",
        "f1_micro": "f1_micro",
        "roc_auc": "roc_auc",
        "auc_roc": "roc_auc",
        "pr_auc": "average_precision",
        "average_precision": "average_precision",
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "mean_absolute_error": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "mean_squared_error": "neg_mean_squared_error",
        "rmse": "neg_root_mean_squared_error",
        "root_mean_squared_error": "neg_root_mean_squared_error",
        "log_loss": "neg_log_loss",
        "logloss": "neg_log_loss",
    }
    return aliases.get(normalized, normalized)
