"""Leakage-safe diagnostics for one reference and one primary candidate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ._utils import _sklearn_import_error
from .contracts import (
    CLASSIFICATION_TASKS,
    CVEvaluation,
    ExperimentSettings,
    FeaturePlan,
    ModelingSettings,
    PreparedData,
    ScoringPlan,
    TRACKED_METRIC_FIGURE_ROOT,
)


@dataclass(frozen=True)
class ExperimentDiagnostics:
    """Tables explaining how one controlled candidate changed the reference."""

    reference_model: str
    candidate_model: str
    focus_features: tuple[str, ...]
    removed_features: tuple[str, ...]
    pipeline_stages: pd.DataFrame
    transformed_features: pd.DataFrame
    transformed_preview: pd.DataFrame
    paired_fold_deltas: pd.DataFrame
    oof_predictions: pd.DataFrame
    prediction_changes: pd.DataFrame
    confusion: pd.DataFrame
    permutation_importance: pd.DataFrame
    native_importance: pd.DataFrame
    threshold_metrics: pd.DataFrame
    slice_metrics: pd.DataFrame
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SavedDiagnostics:
    """Paths written for one experiment diagnostic report."""

    artifact_dir: Path
    figure_dir: Path
    table_paths: Mapping[str, Path]
    figure_paths: Mapping[str, Path]
    summary_path: Path


def _source_feature(name: str, raw_features: Sequence[str]) -> str:
    base = str(name).split("__", maxsplit=1)[-1]
    for feature in sorted(map(str, raw_features), key=len, reverse=True):
        if base == feature or base.startswith(feature + "_"):
            return feature
    return ""


def _is_sparse(matrix: Any) -> bool:
    return hasattr(matrix, "nnz") and hasattr(matrix, "tocsr")


def _density(matrix: Any) -> float:
    rows, columns = matrix.shape
    cells = int(rows) * int(columns)
    if not cells:
        return 0.0
    if _is_sparse(matrix):
        return float(matrix.nnz / cells)
    values = np.asarray(matrix)
    return float(np.count_nonzero(values) / cells)


def _missing_cells(matrix: Any) -> int | None:
    if isinstance(matrix, pd.DataFrame):
        return int(matrix.isna().sum().sum())
    if _is_sparse(matrix):
        data = np.asarray(matrix.data)
        return int(np.isnan(data).sum()) if np.issubdtype(data.dtype, np.number) else 0
    values = np.asarray(matrix)
    if not np.issubdtype(values.dtype, np.number):
        return None
    return int(np.isnan(values).sum())


def _output_names(
    transformer: Any,
    transformed: Any,
    input_names: Sequence[str],
) -> list[str]:
    if isinstance(transformed, pd.DataFrame):
        return [str(column) for column in transformed.columns]
    getter = getattr(transformer, "get_feature_names_out", None)
    if callable(getter):
        for args in ((), (np.asarray(input_names, dtype=object),)):
            try:
                names = getter(*args)
            except (TypeError, ValueError, AttributeError):
                continue
            if len(names) == transformed.shape[1]:
                return [str(name) for name in names]
    return [f"feature_{index}" for index in range(transformed.shape[1])]


def _trace_pipeline(
    fitted_pipeline: Any,
    X: pd.DataFrame,
    *,
    focus_features: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stages = [
        {
            "stage": "input",
            "transformer": "DataFrame",
            "rows": int(X.shape[0]),
            "columns": int(X.shape[1]),
            "sparse": False,
            "density": _density(X),
            "missing_cells": _missing_cells(X),
        }
    ]
    current: Any = X.copy()
    names = [str(column) for column in X.columns]
    steps = list(getattr(fitted_pipeline, "steps", []))
    transform_steps = steps[:-1] if steps else []
    for step_name, transformer in transform_steps:
        current = transformer.transform(current)
        names = _output_names(transformer, current, names)
        stages.append(
            {
                "stage": step_name,
                "transformer": type(transformer).__name__,
                "rows": int(current.shape[0]),
                "columns": int(current.shape[1]),
                "sparse": _is_sparse(current),
                "density": _density(current),
                "missing_cells": _missing_cells(current),
            }
        )

    raw_features = [str(column) for column in X.columns]
    if _is_sparse(current):
        matrix = current.tocsr().astype(float)
        means = np.asarray(matrix.mean(axis=0)).ravel()
        squares = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel()
        stds = np.sqrt(np.maximum(squares - means**2, 0.0))
        nonzero = np.asarray(matrix.getnnz(axis=0)).ravel() / max(matrix.shape[0], 1)
    else:
        matrix = np.asarray(current, dtype=float)
        means = np.nanmean(matrix, axis=0)
        stds = np.nanstd(matrix, axis=0)
        nonzero = np.count_nonzero(np.nan_to_num(matrix), axis=0) / max(
            matrix.shape[0], 1
        )

    feature_rows = []
    for index, name in enumerate(names):
        source = _source_feature(name, raw_features)
        feature_rows.append(
            {
                "transformed_feature": name,
                "source_feature": source,
                "focus_feature": source in focus_features,
                "mean": float(means[index]),
                "std": float(stds[index]),
                "nonzero_share": float(nonzero[index]),
                "constant": bool(np.isclose(stds[index], 0.0)),
            }
        )
    transformed_features = pd.DataFrame(feature_rows)

    focus_indices = [
        index
        for index, row in enumerate(feature_rows)
        if row["focus_feature"]
    ]
    preview_indices = focus_indices or list(range(min(20, len(names))))
    preview_names = [names[index] for index in preview_indices]
    if _is_sparse(current):
        preview_values = current[:30, preview_indices].toarray()
    else:
        preview_values = np.asarray(current)[:30, preview_indices]
    preview = pd.DataFrame(
        preview_values,
        columns=preview_names,
        index=X.index[:30],
    )
    preview.index.name = "row_index"
    return pd.DataFrame(stages), transformed_features, preview.reset_index()


def _probabilities(estimator: Any, X: pd.DataFrame) -> tuple[np.ndarray | None, Any]:
    predictor = getattr(estimator, "predict_proba", None)
    classes = np.asarray(getattr(estimator, "classes_", []), dtype=object)
    if not callable(predictor) or len(classes) != 2:
        return None, None
    values = np.asarray(predictor(X))
    return values[:, 1].astype(float), classes[1]


def _paired_fold_deltas(
    evaluation: CVEvaluation,
    reference_model: str,
    candidate_model: str,
) -> pd.DataFrame:
    validation = evaluation.fold_scores[
        evaluation.fold_scores["split"].eq("validation")
    ]
    pivot = validation.pivot_table(
        index=["fold", "metric_key", "metric", "direction"],
        columns="model",
        values="value",
        aggfunc="first",
    ).reset_index()
    required = {reference_model, candidate_model}
    if not required.issubset(pivot.columns):
        missing = sorted(required.difference(pivot.columns))
        raise KeyError("Diagnostics models are absent from CV results: " + ", ".join(missing))
    pivot = pivot.rename(
        columns={
            reference_model: "reference_value",
            candidate_model: "candidate_value",
        }
    )
    raw_delta = pivot["candidate_value"] - pivot["reference_value"]
    pivot["improvement"] = np.where(
        pivot["direction"].eq("minimize"),
        -raw_delta,
        raw_delta,
    )
    return pivot


def _prediction_tables(
    evaluation: CVEvaluation,
    data: PreparedData,
    reference_model: str,
    candidate_model: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    reference_estimators = evaluation.raw_results[reference_model].get("estimator")
    candidate_estimators = evaluation.raw_results[candidate_model].get("estimator")
    if reference_estimators is None or candidate_estimators is None:
        raise ValueError("CV evaluation did not retain fitted fold estimators")
    if len(evaluation.cv_splits) != len(reference_estimators):
        raise ValueError("Fitted estimator count does not match CV split count")

    rows: list[pd.DataFrame] = []
    positive_label: Any = None
    for fold, ((_, validation_indices), reference, candidate) in enumerate(
        zip(
            evaluation.cv_splits,
            reference_estimators,
            candidate_estimators,
        ),
        start=1,
    ):
        positions = np.asarray(validation_indices, dtype=int)
        X_validation = data.X.iloc[positions]
        y_validation = data.y.iloc[positions]
        reference_prediction = np.asarray(reference.predict(X_validation))
        candidate_prediction = np.asarray(candidate.predict(X_validation))
        reference_probability, reference_positive = _probabilities(
            reference, X_validation
        )
        candidate_probability, candidate_positive = _probabilities(
            candidate, X_validation
        )
        if candidate_positive is not None:
            positive_label = candidate_positive
        elif reference_positive is not None:
            positive_label = reference_positive
        reference_correct = reference_prediction == y_validation.to_numpy()
        candidate_correct = candidate_prediction == y_validation.to_numpy()
        outcome = np.select(
            [
                ~reference_correct & candidate_correct,
                reference_correct & ~candidate_correct,
                reference_correct & candidate_correct,
            ],
            ["fixed", "broken", "both_correct"],
            default="both_wrong",
        )
        fold_frame = pd.DataFrame(
            {
                "row_position": positions,
                "row_index": data.row_index.take(positions),
                "fold": fold,
                "y_true": y_validation.to_numpy(),
                "reference_prediction": reference_prediction,
                "candidate_prediction": candidate_prediction,
                "reference_correct": reference_correct,
                "candidate_correct": candidate_correct,
                "outcome_change": outcome,
            }
        )
        if reference_probability is not None:
            fold_frame["reference_probability"] = reference_probability
        if candidate_probability is not None:
            fold_frame["candidate_probability"] = candidate_probability
        if reference_probability is not None and candidate_probability is not None:
            fold_frame["probability_delta"] = (
                candidate_probability - reference_probability
            )
        rows.append(fold_frame)
    oof = pd.concat(rows, ignore_index=True).sort_values("row_position")
    changes = (
        oof.groupby("outcome_change", as_index=False, sort=False)
        .size()
        .rename(columns={"size": "rows"})
    )
    changes["share"] = changes["rows"] / len(oof)

    confusion_rows = []
    for model_name, column in (
        (reference_model, "reference_prediction"),
        (candidate_model, "candidate_prediction"),
    ):
        counts = (
            oof.groupby(["y_true", column], dropna=False)
            .size()
            .reset_index(name="rows")
            .rename(columns={column: "prediction"})
        )
        counts.insert(0, "model", model_name)
        confusion_rows.append(counts)
    confusion = pd.concat(confusion_rows, ignore_index=True)
    return oof, changes, confusion, positive_label


def _permutation_importance(
    evaluation: CVEvaluation,
    data: PreparedData,
    scoring: ScoringPlan,
    candidate_model: str,
    settings: ModelingSettings,
    repeats: int,
) -> pd.DataFrame:
    try:
        from sklearn.inspection import permutation_importance
    except ImportError as error:  # pragma: no cover - environment dependent
        raise _sklearn_import_error() from error

    estimators = evaluation.raw_results[candidate_model]["estimator"]
    rows = []
    seed = settings.random_state or 0
    for fold, ((_, validation_indices), estimator) in enumerate(
        zip(evaluation.cv_splits, estimators), start=1
    ):
        positions = np.asarray(validation_indices, dtype=int)
        X_validation = data.X.iloc[positions]
        y_validation = data.y.iloc[positions]
        result = permutation_importance(
            estimator,
            X_validation,
            y_validation,
            scoring=scoring.scorers["primary"],
            n_repeats=repeats,
            random_state=seed + fold,
            n_jobs=1,
        )
        for feature, mean, std in zip(
            X_validation.columns,
            result.importances_mean,
            result.importances_std,
        ):
            rows.append(
                {
                    "fold": fold,
                    "feature": str(feature),
                    "importance_mean": float(mean),
                    "importance_std": float(std),
                }
            )
    return pd.DataFrame(rows)


def _native_importance(
    evaluation: CVEvaluation,
    data: PreparedData,
    candidate_model: str,
) -> pd.DataFrame:
    rows = []
    estimators = evaluation.raw_results[candidate_model]["estimator"]
    raw_features = [str(column) for column in data.X.columns]
    for fold, ((_, validation_indices), pipeline) in enumerate(
        zip(evaluation.cv_splits, estimators), start=1
    ):
        final_estimator = (
            pipeline.steps[-1][1]
            if getattr(pipeline, "steps", None)
            else pipeline
        )
        preprocessor = getattr(pipeline, "named_steps", {}).get("preprocess")
        if preprocessor is None or not hasattr(preprocessor, "get_feature_names_out"):
            continue
        names = [str(name) for name in preprocessor.get_feature_names_out()]
        values: np.ndarray | None = None
        kind = ""
        if hasattr(final_estimator, "coef_"):
            coefficients = np.asarray(final_estimator.coef_, dtype=float)
            values = coefficients[0] if coefficients.shape[0] == 1 else coefficients.mean(axis=0)
            kind = "coefficient"
        elif hasattr(final_estimator, "feature_importances_"):
            values = np.asarray(final_estimator.feature_importances_, dtype=float)
            kind = "feature_importance"
        if values is None or len(values) != len(names):
            continue
        for name, value in zip(names, values):
            rows.append(
                {
                    "fold": fold,
                    "transformed_feature": name,
                    "source_feature": _source_feature(name, raw_features),
                    "importance_kind": kind,
                    "value": float(value),
                    "absolute_value": float(abs(value)),
                }
            )
    return pd.DataFrame(rows)


def _threshold_metrics(
    oof: pd.DataFrame,
    reference_model: str,
    candidate_model: str,
    positive_label: Any,
) -> pd.DataFrame:
    required = {"reference_probability", "candidate_probability"}
    if positive_label is None or not required.issubset(oof.columns):
        return pd.DataFrame()
    try:
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )
    except ImportError as error:  # pragma: no cover - environment dependent
        raise _sklearn_import_error() from error
    classes = list(pd.unique(oof["y_true"]))
    if len(classes) != 2:
        return pd.DataFrame()
    negative_label = next(value for value in classes if value != positive_label)
    rows = []
    for model_name, probability_column in (
        (reference_model, "reference_probability"),
        (candidate_model, "candidate_probability"),
    ):
        probabilities = oof[probability_column].to_numpy(dtype=float)
        for threshold in np.linspace(0.05, 0.95, 19):
            prediction = np.where(
                probabilities >= threshold,
                positive_label,
                negative_label,
            )
            rows.append(
                {
                    "model": model_name,
                    "threshold": float(threshold),
                    "accuracy": float(accuracy_score(oof["y_true"], prediction)),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(oof["y_true"], prediction)
                    ),
                    "precision": float(
                        precision_score(
                            oof["y_true"], prediction,
                            pos_label=positive_label, zero_division=0,
                        )
                    ),
                    "recall": float(
                        recall_score(
                            oof["y_true"], prediction,
                            pos_label=positive_label, zero_division=0,
                        )
                    ),
                    "f1": float(
                        f1_score(
                            oof["y_true"], prediction,
                            pos_label=positive_label, zero_division=0,
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def _slice_metrics(
    oof: pd.DataFrame,
    data: PreparedData,
    focus_features: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for feature in focus_features:
        if feature not in data.X:
            continue
        values = data.X.iloc[oof["row_position"].to_numpy(dtype=int)][feature].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(values) and values.nunique(dropna=True) > 10:
            try:
                slices = pd.qcut(values, q=5, duplicates="drop").astype("string")
            except ValueError:
                continue
        elif values.nunique(dropna=False) <= 20:
            slices = values.astype("string").fillna("<missing>")
        else:
            continue
        frame = oof[["reference_correct", "candidate_correct"]].copy()
        frame["slice"] = slices
        grouped = frame.groupby("slice", dropna=False)
        for slice_name, part in grouped:
            if len(part) < 10:
                continue
            reference_accuracy = float(part["reference_correct"].mean())
            candidate_accuracy = float(part["candidate_correct"].mean())
            rows.append(
                {
                    "feature": feature,
                    "slice": str(slice_name),
                    "rows": int(len(part)),
                    "reference_accuracy": reference_accuracy,
                    "candidate_accuracy": candidate_accuracy,
                    "accuracy_delta": candidate_accuracy - reference_accuracy,
                }
            )
    return pd.DataFrame(rows)


def diagnose_experiment(
    evaluation: CVEvaluation,
    data: PreparedData,
    scoring: ScoringPlan,
    settings: ModelingSettings,
    reference_plan: FeaturePlan,
    candidate_plan: FeaturePlan,
    *,
    reference_model: str,
    candidate_model: str,
    permutation_repeats: int = 8,
) -> ExperimentDiagnostics:
    """Diagnose one primary candidate against its reference on fitted CV folds."""

    if permutation_repeats < 1:
        raise ValueError("permutation_repeats must be positive")
    if not evaluation.cv_splits:
        raise ValueError("CV evaluation has no retained fold indices")
    for model_name in (reference_model, candidate_model):
        if model_name not in evaluation.raw_results:
            raise KeyError(f"Model {model_name!r} is absent from CV evaluation")

    reference_features = set(reference_plan.model_features)
    candidate_features = set(candidate_plan.model_features)
    focus_features = tuple(
        feature
        for feature in candidate_plan.model_features
        if feature not in reference_features
    )
    removed_features = tuple(
        feature
        for feature in reference_plan.model_features
        if feature not in candidate_features
    )
    warnings = []
    if not focus_features and not removed_features:
        warnings.append(
            "Feature plan не изменился; диагностика описывает pipeline/model change."
        )

    candidate_estimators = evaluation.raw_results[candidate_model]["estimator"]
    first_validation = np.asarray(evaluation.cv_splits[0][1], dtype=int)
    pipeline_stages, transformed_features, transformed_preview = _trace_pipeline(
        candidate_estimators[0],
        data.X.iloc[first_validation],
        focus_features=focus_features,
    )
    paired = _paired_fold_deltas(evaluation, reference_model, candidate_model)
    oof, changes, confusion, positive_label = _prediction_tables(
        evaluation, data, reference_model, candidate_model
    )
    permutation = _permutation_importance(
        evaluation,
        data,
        scoring,
        candidate_model,
        settings,
        permutation_repeats,
    )
    native = _native_importance(evaluation, data, candidate_model)
    threshold = (
        _threshold_metrics(oof, reference_model, candidate_model, positive_label)
        if settings.task_type in CLASSIFICATION_TASKS
        else pd.DataFrame()
    )
    slices = _slice_metrics(oof, data, focus_features)
    return ExperimentDiagnostics(
        reference_model=reference_model,
        candidate_model=candidate_model,
        focus_features=focus_features,
        removed_features=removed_features,
        pipeline_stages=pipeline_stages,
        transformed_features=transformed_features,
        transformed_preview=transformed_preview,
        paired_fold_deltas=paired,
        oof_predictions=oof,
        prediction_changes=changes,
        confusion=confusion,
        permutation_importance=permutation,
        native_importance=native,
        threshold_metrics=threshold,
        slice_metrics=slices,
        warnings=tuple(warnings),
    )


def _save_figures(
    diagnostics: ExperimentDiagnostics,
    figure_dir: Path,
    dpi: int,
) -> dict[str, Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError("Diagnostics figures require matplotlib") from error

    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    changes = diagnostics.prediction_changes.set_index("outcome_change")["rows"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    changes.reindex(
        ["fixed", "broken", "both_correct", "both_wrong"], fill_value=0
    ).plot.bar(ax=axis, color=["#2ca02c", "#d62728", "#4c78a8", "#9e9e9e"])
    axis.set_title("Изменение OOF-ошибок candidate относительно reference")
    axis.set_xlabel("")
    axis.set_ylabel("Строк")
    axis.tick_params(axis="x", rotation=0)
    figure.tight_layout()
    path = figure_dir / "diagnostic-prediction-changes.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    paths["prediction_changes"] = path

    permutation = diagnostics.permutation_importance
    if not permutation.empty:
        summary = (
            permutation.groupby("feature", as_index=False)
            .agg(mean=("importance_mean", "mean"), std=("importance_mean", "std"))
            .sort_values("mean", ascending=False)
            .head(15)
            .sort_values("mean")
        )
        figure, axis = plt.subplots(figsize=(8, 6))
        axis.barh(summary["feature"], summary["mean"], xerr=summary["std"].fillna(0))
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_title("Validation permutation importance candidate")
        axis.set_xlabel("Падение primary score после перестановки")
        figure.tight_layout()
        path = figure_dir / "diagnostic-permutation-importance.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        paths["permutation_importance"] = path

    threshold = diagnostics.threshold_metrics
    if not threshold.empty:
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for model_name, part in threshold.groupby("model", sort=False):
            axes[0].plot(part["threshold"], part["accuracy"], label=model_name)
            axes[1].plot(part["threshold"], part["f1"], label=model_name)
        axes[0].set_title("Accuracy по порогу")
        axes[1].set_title("F1 по порогу")
        for axis in axes:
            axis.axvline(0.5, color="black", linestyle="--", linewidth=0.8)
            axis.set_xlabel("Threshold")
            axis.set_ylabel("Score")
            axis.legend()
        figure.tight_layout()
        path = figure_dir / "diagnostic-thresholds.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        paths["thresholds"] = path

    return paths


def save_experiment_diagnostics(
    project_root: Path,
    diagnostics: ExperimentDiagnostics,
    settings: ModelingSettings | ExperimentSettings,
) -> SavedDiagnostics:
    """Save diagnostic tables privately and compact figures in tracked assets."""

    root = Path(project_root).resolve()
    artifact_dir = (root / settings.artifact_dir / settings.run_name / "diagnostics").resolve()
    figure_dir = (
        root / TRACKED_METRIC_FIGURE_ROOT / settings.experiment_id / "diagnostics"
    ).resolve()
    for path in (artifact_dir, figure_dir):
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Diagnostics path resolves outside project root: {path}") from error
        path.mkdir(parents=True, exist_ok=True)

    tables = {
        "pipeline_stages": diagnostics.pipeline_stages,
        "transformed_features": diagnostics.transformed_features,
        "transformed_preview": diagnostics.transformed_preview,
        "paired_fold_deltas": diagnostics.paired_fold_deltas,
        "oof_predictions": diagnostics.oof_predictions,
        "prediction_changes": diagnostics.prediction_changes,
        "confusion": diagnostics.confusion,
        "permutation_importance": diagnostics.permutation_importance,
        "native_importance": diagnostics.native_importance,
        "threshold_metrics": diagnostics.threshold_metrics,
        "slice_metrics": diagnostics.slice_metrics,
    }
    table_paths = {}
    for name, frame in tables.items():
        path = artifact_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        table_paths[name] = path
    summary_path = artifact_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "reference_model": diagnostics.reference_model,
                "candidate_model": diagnostics.candidate_model,
                "focus_features": list(diagnostics.focus_features),
                "removed_features": list(diagnostics.removed_features),
                "warnings": list(diagnostics.warnings),
                "tables": {key: path.name for key, path in table_paths.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    figure_paths = _save_figures(
        diagnostics,
        figure_dir,
        settings.metric_figure_dpi,
    )
    return SavedDiagnostics(
        artifact_dir=artifact_dir,
        figure_dir=figure_dir,
        table_paths=table_paths,
        figure_paths=figure_paths,
        summary_path=summary_path,
    )


__all__ = [
    "ExperimentDiagnostics",
    "SavedDiagnostics",
    "diagnose_experiment",
    "save_experiment_diagnostics",
]
