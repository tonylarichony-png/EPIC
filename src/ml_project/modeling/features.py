"""Feature planning, model-ready data and leakage-safe preprocessing."""

from __future__ import annotations

import inspect
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..profiling import validate_feature_groups
from ._utils import _sklearn_import_error
from .contracts import (
    ModelingSettings,
    CLASSIFICATION_TASKS,
    FeaturePlan,
    PreparedData,
)
from .validation import resolve_cv_strategy
def resolve_feature_plan(
    frame: pd.DataFrame,
    feature_groups: Mapping[str, Sequence[str]],
    *,
    target: str | None,
    key: str | None,
    settings: ModelingSettings,
) -> FeaturePlan:
    """Resolve explicit feature groups into model columns without dtype guessing."""

    if target is None:
        raise ValueError("TARGET is not configured in src/ml_project/config.py")
    if target not in frame.columns:
        raise KeyError(f"TARGET {target!r} is absent from the train dataset")

    normalized = validate_feature_groups(frame, feature_groups, target=target)
    group_by_feature = {
        feature: group
        for group, features in normalized.items()
        for feature in features
    }

    unknown_include = sorted(set(settings.include_features) - set(frame.columns))
    unknown_exclude = sorted(set(settings.exclude_features) - set(frame.columns))
    if unknown_include or unknown_exclude:
        messages = []
        if unknown_include:
            messages.append(
                "settings.include_features absent from train: "
                + ", ".join(unknown_include)
            )
        if unknown_exclude:
            messages.append(
                "settings.exclude_features absent from train: "
                + ", ".join(unknown_exclude)
            )
        raise ValueError("\n".join(messages))

    forbidden_include = sorted(
        set(settings.include_features) & {value for value in (target, key) if value}
    )
    if forbidden_include:
        raise ValueError(
            "TARGET/KEY cannot be model features: " + ", ".join(forbidden_include)
        )

    enabled_groups = set(settings.model_feature_groups)
    group_candidates = {
        feature
        for group in enabled_groups
        for feature in normalized[group]
    }
    if settings.include_features:
        requested = set(settings.include_features)
        unsupported_requested = sorted(requested - group_candidates)
        if unsupported_requested:
            raise ValueError(
                "settings.include_features contains columns outside "
                "settings.model_feature_groups: "
                + ", ".join(unsupported_requested)
            )
        selected = requested
    else:
        selected = group_candidates

    selected -= set(settings.exclude_features)
    selected.discard(target)
    if key:
        selected.discard(key)

    numeric_groups = {"numeric", "count"}
    categorical_groups = {"categorical", "ordinal"}
    numeric = tuple(
        column
        for column in frame.columns
        if column in selected and group_by_feature.get(column) in numeric_groups
    )
    categorical = tuple(
        column
        for column in frame.columns
        if column in selected and group_by_feature.get(column) in categorical_groups
    )
    if not numeric and not categorical:
        raise ValueError(
            "The baseline has no model features. Fill FEATURE_GROUPS or adjust "
            "settings.model_feature_groups/settings.include_features."
        )

    exclusion_reason: dict[str, str] = {}
    warnings: list[str] = []
    for feature, group in group_by_feature.items():
        if feature in selected:
            continue
        if feature == key:
            reason = "project key / identifier"
        elif feature in settings.exclude_features:
            reason = "explicit settings.exclude_features"
        elif settings.include_features and feature not in settings.include_features:
            reason = "outside settings.include_features whitelist"
        elif group not in enabled_groups:
            reason = f"group {group!r} is not enabled for this baseline"
        else:
            reason = "not selected"
        exclusion_reason[feature] = reason

    unsupported = [
        feature
        for feature, group in group_by_feature.items()
        if group in {"text", "datetime"} and feature not in selected
    ]
    if unsupported:
        warnings.append(
            "Текстовые и временные признаки намеренно исключены из первого "
            "табличного baseline: " + ", ".join(unsupported)
        )

    excluded = tuple(
        column
        for column in frame.columns
        if column != target and column not in selected
    )
    return FeaturePlan(
        numeric=numeric,
        categorical=categorical,
        excluded=excluded,
        group_by_feature=group_by_feature,
        exclusion_reason=exclusion_reason,
        warnings=tuple(warnings),
    )


def validate_inference_schema(
    inference: pd.DataFrame,
    plan: FeaturePlan,
    *,
    strict: bool = True,
) -> list[str]:
    """Check that inference can receive the exact training feature contract."""

    missing = [feature for feature in plan.model_features if feature not in inference]
    if missing and strict:
        raise ValueError(
            "Inference dataset lacks baseline features: " + ", ".join(missing)
        )
    return [
        "Inference dataset lacks baseline features: " + ", ".join(missing)
    ] if missing else []



def prepare_training_data(
    frame: pd.DataFrame,
    *,
    target: str,
    plan: FeaturePlan,
    settings: ModelingSettings,
) -> PreparedData:
    """Prepare rows for CV without fitting any preprocessing statistic."""

    if frame[target].isna().any():
        raise ValueError("TARGET contains missing values; define an explicit policy")

    strategy = resolve_cv_strategy(settings)
    ordered = frame
    if strategy == "time_series":
        time_column = settings.time_column
        if not time_column or time_column not in frame:
            raise KeyError("Configured TIME_COLUMN is absent from train")
        if frame[time_column].isna().any():
            raise ValueError("TIME_COLUMN contains missing values")
        ordered = frame.sort_values(time_column, kind="stable")

    y = ordered[target].copy()
    if settings.task_type in CLASSIFICATION_TASKS:
        classes = y.value_counts(dropna=False)
        minimum_classes = 2 if settings.task_type == "binary_classification" else 3
        if len(classes) < minimum_classes:
            raise ValueError(
                f"{settings.task_type} requires at least {minimum_classes} target classes"
            )
        if strategy == "stratified_kfold" and int(classes.min()) < settings.n_splits:
            raise ValueError(
                "The rarest target class has fewer rows than N_SPLITS; reduce "
                "N_SPLITS or choose another validated protocol."
            )
    elif not pd.api.types.is_numeric_dtype(y):
        raise TypeError("Regression TARGET must be numeric")

    groups: pd.Series | None = None
    if strategy == "group_kfold":
        group_column = settings.group_column
        if not group_column or group_column not in ordered:
            raise KeyError("Configured GROUP_COLUMN is absent from train")
        if ordered[group_column].isna().any():
            raise ValueError("GROUP_COLUMN contains missing values")
        groups = ordered[group_column].copy()
        if groups.nunique() < settings.n_splits:
            raise ValueError("GROUP_COLUMN has fewer unique groups than N_SPLITS")

    return PreparedData(
        X=ordered.loc[:, list(plan.model_features)].copy(),
        y=y,
        groups=groups,
        row_index=ordered.index.copy(),
    )



def build_tabular_preprocessor(settings: ModelingSettings, plan: FeaturePlan) -> Any:
    """Build fold-fitted numeric/categorical preprocessing."""

    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import (
            MinMaxScaler,
            OneHotEncoder,
            RobustScaler,
            StandardScaler,
        )
    except ImportError as error:  # pragma: no cover - environment dependent
        raise _sklearn_import_error() from error

    transformers: list[tuple[str, Any, list[str]]] = []
    if plan.numeric:
        numeric_imputer_kwargs: dict[str, Any] = {
            "strategy": settings.numeric_imputer,
            "add_indicator": settings.add_numeric_missing_indicator,
        }
        if settings.numeric_imputer == "constant":
            numeric_imputer_kwargs["fill_value"] = settings.numeric_fill_value
        if "keep_empty_features" in inspect.signature(SimpleImputer).parameters:
            numeric_imputer_kwargs["keep_empty_features"] = True

        numeric_steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(**numeric_imputer_kwargs))
        ]
        scaler = {
            "standard": StandardScaler(),
            "robust": RobustScaler(),
            "minmax": MinMaxScaler(),
            "none": None,
        }[settings.numeric_scaler]
        if scaler is not None:
            numeric_steps.append(("scaler", scaler))
        transformers.append(
            ("numeric", Pipeline(numeric_steps), list(plan.numeric))
        )

    if plan.categorical:
        categorical_imputer_kwargs: dict[str, Any] = {
            "strategy": settings.categorical_imputer,
        }
        if settings.categorical_imputer == "constant":
            categorical_imputer_kwargs["fill_value"] = (
                settings.categorical_fill_value
            )
        if "keep_empty_features" in inspect.signature(SimpleImputer).parameters:
            categorical_imputer_kwargs["keep_empty_features"] = True

        encoder_kwargs: dict[str, Any] = {
            "handle_unknown": settings.onehot_handle_unknown,
        }
        encoder_signature = inspect.signature(OneHotEncoder).parameters
        if "sparse_output" in encoder_signature:
            encoder_kwargs["sparse_output"] = settings.onehot_sparse_output
        else:  # scikit-learn < 1.2
            encoder_kwargs["sparse"] = settings.onehot_sparse_output
        if settings.onehot_min_frequency is not None:
            encoder_kwargs["min_frequency"] = settings.onehot_min_frequency
        if settings.onehot_max_categories is not None:
            encoder_kwargs["max_categories"] = settings.onehot_max_categories

        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(**categorical_imputer_kwargs)),
                ("onehot", OneHotEncoder(**encoder_kwargs)),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, list(plan.categorical))
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=settings.column_transformer_sparse_threshold,
        verbose_feature_names_out=True,
    )


def preprocessing_report(
    settings: ModelingSettings,
    plan: FeaturePlan,
) -> pd.DataFrame:
    """Explain the transformations before any model is fitted."""

    rows: list[dict[str, Any]] = []
    if plan.numeric:
        rows.append(
            {
                "model_role": "numeric",
                "features": len(plan.numeric),
                "imputation": settings.numeric_imputer,
                "representation": settings.numeric_scaler,
                "unknown_policy": "—",
            }
        )
    if plan.categorical:
        rows.append(
            {
                "model_role": "categorical",
                "features": len(plan.categorical),
                "imputation": settings.categorical_imputer,
                "representation": "one-hot",
                "unknown_policy": settings.onehot_handle_unknown,
            }
        )
    return pd.DataFrame(rows)
