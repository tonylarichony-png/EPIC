"""Reference estimators and reusable sklearn pipeline assembly."""

from __future__ import annotations

from typing import Any

from ._utils import _sklearn_import_error
from .contracts import ModelingSettings, CLASSIFICATION_TASKS
def build_dummy_estimator(settings: ModelingSettings) -> Any:
    """Build a no-skill floor appropriate for the task."""

    try:
        from sklearn.dummy import DummyClassifier, DummyRegressor
    except ImportError as error:  # pragma: no cover - environment dependent
        raise _sklearn_import_error() from error

    params = dict(settings.dummy_params)
    if settings.task_type in CLASSIFICATION_TASKS:
        strategy = "prior" if settings.dummy_strategy == "auto" else settings.dummy_strategy
        params.setdefault("strategy", strategy)
        if strategy in {"stratified", "uniform"}:
            params.setdefault("random_state", settings.random_state)
        return DummyClassifier(**params)
    strategy = "mean" if settings.dummy_strategy == "auto" else settings.dummy_strategy
    params.setdefault("strategy", strategy)
    return DummyRegressor(**params)


def build_simple_estimator(settings: ModelingSettings) -> Any:
    """Build the intentionally simple, interpretable reference estimator."""

    try:
        from sklearn.linear_model import LogisticRegression, Ridge
    except ImportError as error:  # pragma: no cover - environment dependent
        raise _sklearn_import_error() from error

    model_name = settings.model_name
    if model_name == "auto":
        model_name = (
            "logistic_regression"
            if settings.task_type in CLASSIFICATION_TASKS
            else "ridge"
        )
    params = dict(settings.model_params)
    if model_name == "logistic_regression":
        params.setdefault("max_iter", 1000)
        params.setdefault("random_state", settings.random_state)
        return LogisticRegression(**params)
    if model_name == "ridge":
        return Ridge(**params)
    raise ValueError(f"Unsupported simple baseline model: {model_name}")


def resolved_model_name(settings: ModelingSettings) -> str:
    if settings.model_name != "auto":
        return settings.model_name
    return (
        "logistic_regression"
        if settings.task_type in CLASSIFICATION_TASKS
        else "ridge"
    )


def build_model_pipeline(preprocessor: Any, estimator: Any) -> Any:
    """Combine preprocessing and estimator into the leakage boundary."""

    try:
        from sklearn.base import clone
        from sklearn.pipeline import Pipeline
    except ImportError as error:  # pragma: no cover - environment dependent
        raise _sklearn_import_error() from error

    return Pipeline(
        [
            ("preprocess", clone(preprocessor)),
            ("model", estimator),
        ]
    )


def resolved_model_label(model_name: str) -> str:
    return model_name.replace("_", " ")
