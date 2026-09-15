"""OOF comparison and model-native importance for grouped screening."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


LOSS_CURVE_COLUMNS = [
    "model",
    "fold",
    "iteration",
    "split",
    "log_loss",
]


def build_staged_log_loss_diagnostics(
    evaluation: Any,
    data: Any,
) -> pd.DataFrame:
    """Рассчитать train/validation log loss по boosting-итерациям и CV-folds."""

    try:
        from sklearn.metrics import log_loss
    except ImportError:  # pragma: no cover - project dependency
        return pd.DataFrame(columns=LOSS_CURVE_COLUMNS)

    rows: list[dict[str, Any]] = []
    for model_name, raw in evaluation.raw_results.items():
        estimators = raw.get("estimator")
        if estimators is None:
            continue
        for fold, ((train_indices, validation_indices), pipeline) in enumerate(
            zip(evaluation.cv_splits, estimators),
            start=1,
        ):
            final_model = getattr(pipeline, "named_steps", {}).get("model")
            staged_predict_proba = getattr(
                final_model,
                "staged_predict_proba",
                None,
            )
            if final_model is None or not callable(staged_predict_proba):
                continue
            transformer = pipeline[:-1]
            X_train = transformer.transform(data.X.iloc[train_indices])
            X_validation = transformer.transform(
                data.X.iloc[validation_indices]
            )
            y_train = data.y.iloc[train_indices]
            y_validation = data.y.iloc[validation_indices]
            classes = getattr(final_model, "classes_", None)
            train_stages = final_model.staged_predict_proba(X_train)
            validation_stages = final_model.staged_predict_proba(X_validation)
            for iteration, (train_proba, validation_proba) in enumerate(
                zip(train_stages, validation_stages),
                start=1,
            ):
                rows.extend(
                    [
                        {
                            "model": model_name,
                            "fold": fold,
                            "iteration": iteration,
                            "split": "train",
                            "log_loss": float(
                                log_loss(
                                    y_train,
                                    np.asarray(train_proba),
                                    labels=classes,
                                )
                            ),
                        },
                        {
                            "model": model_name,
                            "fold": fold,
                            "iteration": iteration,
                            "split": "validation",
                            "log_loss": float(
                                log_loss(
                                    y_validation,
                                    np.asarray(validation_proba),
                                    labels=classes,
                                )
                            ),
                        },
                    ]
                )
    if not rows:
        return pd.DataFrame(columns=LOSS_CURVE_COLUMNS)
    return pd.DataFrame(rows, columns=LOSS_CURVE_COLUMNS)


def build_oof_diagnostics(
    evaluation: Any,
    data: Any,
    *,
    reference_model: str,
    classification: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    model_names = list(evaluation.raw_results)
    for model in model_names:
        estimators = evaluation.raw_results[model].get("estimator")
        if estimators is None:
            raise ValueError("CV evaluation did not retain fitted estimators")
        for fold, ((_, validation_indices), estimator) in enumerate(
            zip(evaluation.cv_splits, estimators), start=1
        ):
            X_valid = data.X.iloc[validation_indices]
            predictions = np.asarray(estimator.predict(X_valid))
            for offset, row_position in enumerate(validation_indices):
                rows.append(
                    {
                        "model": model,
                        "fold": fold,
                        "row_position": int(row_position),
                        "row_index": str(data.row_index[row_position]),
                        "actual": data.y.iloc[row_position],
                        "prediction": predictions[offset],
                    }
                )
    long = pd.DataFrame(rows)
    predictions = long.pivot_table(
        index=["fold", "row_position", "row_index", "actual"],
        columns="model",
        values="prediction",
        aggfunc="first",
    ).reset_index()
    comparisons: list[dict[str, Any]] = []
    for model in model_names:
        if model == reference_model:
            continue
        if classification:
            reference_correct = predictions[reference_model].eq(predictions["actual"])
            candidate_correct = predictions[model].eq(predictions["actual"])
            comparisons.append(
                {
                    "model": model,
                    "rows": len(predictions),
                    "agreement_share": float(
                        predictions[model].eq(predictions[reference_model]).mean()
                    ),
                    "candidate_accuracy": float(candidate_correct.mean()),
                    "reference_accuracy": float(reference_correct.mean()),
                    "corrected_errors": int((~reference_correct & candidate_correct).sum()),
                    "new_errors": int((reference_correct & ~candidate_correct).sum()),
                }
            )
        else:
            reference_error = (
                pd.to_numeric(predictions[reference_model])
                - pd.to_numeric(predictions["actual"])
            ).abs()
            candidate_error = (
                pd.to_numeric(predictions[model])
                - pd.to_numeric(predictions["actual"])
            ).abs()
            comparisons.append(
                {
                    "model": model,
                    "rows": len(predictions),
                    "candidate_better_share": float((candidate_error < reference_error).mean()),
                    "candidate_mae": float(candidate_error.mean()),
                    "reference_mae": float(reference_error.mean()),
                    "mae_improvement": float(reference_error.mean() - candidate_error.mean()),
                }
            )
    return predictions, pd.DataFrame(comparisons)


def _source_feature(name: str, raw_features: Sequence[str]) -> str:
    base = str(name).split("__", maxsplit=1)[-1]
    for feature in sorted(map(str, raw_features), key=len, reverse=True):
        if base == feature or base.startswith(feature + "_"):
            return feature
    return base


def build_screening_feature_importance(evaluation: Any, plan: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    raw_features = list(plan.model_features)
    for model, raw in evaluation.raw_results.items():
        for fold, pipeline in enumerate(raw.get("estimator", ()), start=1):
            estimator = pipeline.named_steps.get("model")
            preprocess = pipeline.named_steps.get("preprocess")
            if estimator is None or preprocess is None:
                continue
            if hasattr(estimator, "feature_importances_"):
                values = np.asarray(estimator.feature_importances_, dtype=float)
                kind = "feature_importances_"
            elif hasattr(estimator, "coef_"):
                coefficients = np.asarray(estimator.coef_, dtype=float)
                values = np.abs(coefficients)
                if values.ndim > 1:
                    values = values.mean(axis=0)
                kind = "abs(coef_)"
            else:
                continue
            getter = getattr(preprocess, "get_feature_names_out", None)
            names = list(getter()) if callable(getter) else raw_features
            if len(names) != len(values):
                continue
            fold_rows = pd.DataFrame(
                {
                    "source_feature": [
                        _source_feature(str(name), raw_features) for name in names
                    ],
                    "importance": values,
                }
            )
            aggregated = fold_rows.groupby("source_feature", as_index=False)[
                "importance"
            ].sum()
            for record in aggregated.to_dict("records"):
                rows.append(
                    {
                        "model": model,
                        "fold": fold,
                        "source_feature": record["source_feature"],
                        "importance": float(record["importance"]),
                        "importance_kind": kind,
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=[
                "model",
                "source_feature",
                "importance_mean",
                "importance_std",
                "importance_kind",
            ]
        )
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(
            ["model", "source_feature", "importance_kind"],
            as_index=False,
            sort=False,
        )
        .agg(
            importance_mean=("importance", "mean"),
            importance_std=("importance", "std"),
        )
        .sort_values(["model", "importance_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )



__all__ = [
    "build_oof_diagnostics",
    "build_screening_feature_importance",
    "build_staged_log_loss_diagnostics",
]
