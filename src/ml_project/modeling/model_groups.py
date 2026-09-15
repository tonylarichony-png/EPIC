"""Estimator registry and preprocessing profiles for grouped model screening."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from ._utils import _sklearn_import_error
from .contracts import CLASSIFICATION_TASKS, FeaturePlan, ModelingSettings


PREPROCESSING_PROFILES = {
    "scaled_dense",
    "unscaled_sparse",
    "unscaled_dense",
    "native_categorical",
}


def settings_for_preprocessing_profile(
    settings: ModelingSettings,
    profile: str,
) -> ModelingSettings:
    """Вернуть настройки preprocessing, подходящие всей группе моделей."""

    if profile not in PREPROCESSING_PROFILES:
        raise ValueError(
            f"Unknown preprocessing profile {profile!r}; expected one of: "
            + ", ".join(sorted(PREPROCESSING_PROFILES))
        )
    if profile == "scaled_dense":
        return replace(
            settings,
            numeric_scaler="standard",
            onehot_sparse_output=False,
            column_transformer_sparse_threshold=0.0,
        )
    if profile == "unscaled_sparse":
        return replace(
            settings,
            numeric_scaler="none",
            onehot_sparse_output=True,
            column_transformer_sparse_threshold=0.3,
        )
    if profile in {"unscaled_dense", "native_categorical"}:
        return replace(
            settings,
            numeric_scaler="none",
            onehot_sparse_output=False,
            column_transformer_sparse_threshold=0.0,
        )
    raise AssertionError("unreachable")


class NativeCategoricalPreprocessor:
    """Fold-safe imputation that keeps a pandas frame for CatBoost categories."""

    def __init__(
        self,
        *,
        numeric_features: Sequence[str],
        categorical_features: Sequence[str],
        numeric_strategy: str = "median",
        numeric_fill_value: Any = 0.0,
        categorical_strategy: str = "most_frequent",
        categorical_fill_value: Any = "__MISSING__",
    ) -> None:
        self.numeric_features = tuple(numeric_features)
        self.categorical_features = tuple(categorical_features)
        self.numeric_strategy = numeric_strategy
        self.numeric_fill_value = numeric_fill_value
        self.categorical_strategy = categorical_strategy
        self.categorical_fill_value = categorical_fill_value

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "numeric_strategy": self.numeric_strategy,
            "numeric_fill_value": self.numeric_fill_value,
            "categorical_strategy": self.categorical_strategy,
            "categorical_fill_value": self.categorical_fill_value,
        }

    def set_params(self, **params: Any) -> "NativeCategoricalPreprocessor":
        for name, value in params.items():
            if name not in self.get_params(deep=False):
                raise ValueError(f"Unknown parameter {name!r}")
            setattr(self, name, value)
        return self

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (*self.numeric_features, *self.categorical_features)

    def _validate_frame(self, X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("NativeCategoricalPreprocessor expects a pandas DataFrame")
        missing = sorted(set(self.feature_names).difference(X.columns))
        if missing:
            raise KeyError("Native categorical preprocessing lacks: " + ", ".join(missing))
        return X

    def fit(self, X: pd.DataFrame, y: Any = None) -> "NativeCategoricalPreprocessor":
        frame = self._validate_frame(X)
        numeric_fill: dict[str, Any] = {}
        for feature in self.numeric_features:
            values = pd.to_numeric(frame[feature], errors="coerce")
            if self.numeric_strategy == "median":
                fill = values.median()
            elif self.numeric_strategy == "mean":
                fill = values.mean()
            elif self.numeric_strategy == "most_frequent":
                modes = values.mode(dropna=True)
                fill = modes.iloc[0] if not modes.empty else self.numeric_fill_value
            elif self.numeric_strategy == "constant":
                fill = self.numeric_fill_value
            else:
                raise ValueError(f"Unsupported numeric strategy: {self.numeric_strategy}")
            numeric_fill[feature] = self.numeric_fill_value if pd.isna(fill) else fill

        categorical_fill: dict[str, str] = {}
        for feature in self.categorical_features:
            values = frame[feature].astype("string")
            if self.categorical_strategy == "most_frequent":
                modes = values.mode(dropna=True)
                fill = modes.iloc[0] if not modes.empty else self.categorical_fill_value
            elif self.categorical_strategy == "constant":
                fill = self.categorical_fill_value
            else:
                raise ValueError(
                    f"Unsupported categorical strategy: {self.categorical_strategy}"
                )
            categorical_fill[feature] = str(fill)

        self.numeric_fill_ = numeric_fill
        self.categorical_fill_ = categorical_fill
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "numeric_fill_"):
            raise ValueError("NativeCategoricalPreprocessor is not fitted")
        frame = self._validate_frame(X)
        result = pd.DataFrame(index=frame.index)
        for feature in self.numeric_features:
            result[feature] = pd.to_numeric(frame[feature], errors="coerce").fillna(
                self.numeric_fill_[feature]
            )
        for feature in self.categorical_features:
            result[feature] = (
                frame[feature]
                .astype("string")
                .fillna(self.categorical_fill_[feature])
                .astype(str)
            )
        return result.loc[:, list(self.feature_names)]

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Any = None,
        **fit_params: Any,
    ) -> pd.DataFrame:
        """Fit and transform while preserving the pandas column contract."""

        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.feature_names, dtype=object)


class CatBoostClassifierAdapter(ClassifierMixin, BaseEstimator):
    """Expose stable sklearn tags for CatBoost versions that lack them."""

    def __init__(self, params: Mapping[str, Any]) -> None:
        self.params = params

    def fit(self, X: Any, y: Any, **fit_params: Any) -> "CatBoostClassifierAdapter":
        from catboost import CatBoostClassifier

        self.model_ = CatBoostClassifier(**dict(self.params))
        self.model_.fit(X, y, **fit_params)
        self.classes_ = np.asarray(self.model_.classes_)
        self.n_features_in_ = X.shape[1]
        self.feature_importances_ = np.asarray(self.model_.feature_importances_)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(X)).reshape(-1)

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(X))


class CatBoostRegressorAdapter(RegressorMixin, BaseEstimator):
    """Expose stable sklearn tags for CatBoost versions that lack them."""

    def __init__(self, params: Mapping[str, Any]) -> None:
        self.params = params

    def fit(self, X: Any, y: Any, **fit_params: Any) -> "CatBoostRegressorAdapter":
        from catboost import CatBoostRegressor

        self.model_ = CatBoostRegressor(**dict(self.params))
        self.model_.fit(X, y, **fit_params)
        self.n_features_in_ = X.shape[1]
        self.feature_importances_ = np.asarray(self.model_.feature_importances_)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(X)).reshape(-1)


def build_native_categorical_preprocessor(
    settings: ModelingSettings,
    plan: FeaturePlan,
) -> NativeCategoricalPreprocessor:
    """Build the DataFrame-preserving preprocessing used only by CatBoost."""

    return NativeCategoricalPreprocessor(
        numeric_features=plan.numeric,
        categorical_features=plan.categorical,
        numeric_strategy=settings.numeric_imputer,
        numeric_fill_value=settings.numeric_fill_value,
        categorical_strategy=settings.categorical_imputer,
        categorical_fill_value=settings.categorical_fill_value,
    )


def build_screening_estimator(
    model_id: str,
    settings: ModelingSettings,
    params: Mapping[str, Any],
    *,
    categorical_features: Sequence[str] = (),
) -> Any:
    """Build a classification/regression estimator from a stable registry key."""

    classification = settings.task_type in CLASSIFICATION_TASKS
    if not classification and settings.task_type != "regression":
        raise ValueError(f"Unsupported screening task: {settings.task_type!r}")
    values = dict(params)
    task_key = "classification" if classification else "regression"
    if {"classification", "regression"}.intersection(values):
        task_values = values.get(task_key)
        if not isinstance(task_values, Mapping):
            raise ValueError(
                f"Model {model_id!r} has no parameter mapping for {task_key}"
            )
        values = dict(task_values)

    try:
        if model_id == "linear_reference":
            if classification:
                from sklearn.linear_model import LogisticRegression

                values.setdefault("max_iter", 2000)
                values.setdefault("random_state", settings.random_state)
                return LogisticRegression(**values)
            from sklearn.linear_model import Ridge

            return Ridge(**values)

        if model_id == "knn":
            if classification:
                from sklearn.neighbors import KNeighborsClassifier

                values.setdefault("n_jobs", 1)
                return KNeighborsClassifier(**values)
            from sklearn.neighbors import KNeighborsRegressor

            values.setdefault("n_jobs", 1)
            return KNeighborsRegressor(**values)

        if model_id == "decision_tree":
            if classification:
                from sklearn.tree import DecisionTreeClassifier

                values.setdefault("random_state", settings.random_state)
                return DecisionTreeClassifier(**values)
            from sklearn.tree import DecisionTreeRegressor

            values.setdefault("random_state", settings.random_state)
            return DecisionTreeRegressor(**values)

        if model_id in {"random_forest", "extra_trees"}:
            if classification:
                from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

                estimator_class = (
                    RandomForestClassifier
                    if model_id == "random_forest"
                    else ExtraTreesClassifier
                )
            else:
                from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

                estimator_class = (
                    RandomForestRegressor
                    if model_id == "random_forest"
                    else ExtraTreesRegressor
                )
            values.setdefault("n_estimators", 300)
            values.setdefault("random_state", settings.random_state)
            values.setdefault("n_jobs", 1)
            return estimator_class(**values)

        if model_id == "hist_gradient_boosting":
            if classification:
                from sklearn.ensemble import HistGradientBoostingClassifier

                values.setdefault("random_state", settings.random_state)
                return HistGradientBoostingClassifier(**values)
            from sklearn.ensemble import HistGradientBoostingRegressor

            values.setdefault("random_state", settings.random_state)
            return HistGradientBoostingRegressor(**values)

        if model_id == "gradient_boosting":
            if classification:
                from sklearn.ensemble import GradientBoostingClassifier

                values.setdefault("random_state", settings.random_state)
                return GradientBoostingClassifier(**values)
            from sklearn.ensemble import GradientBoostingRegressor

            values.setdefault("random_state", settings.random_state)
            return GradientBoostingRegressor(**values)
    except ImportError as error:  # pragma: no cover - environment dependent
        raise _sklearn_import_error() from error

    if model_id == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError(
                "Group external_boosting requires xgboost. Install project "
                "dependencies before selecting this group."
            ) from error
        values.setdefault("n_estimators", 300)
        values.setdefault("random_state", settings.random_state)
        values.setdefault("n_jobs", 1)
        values.setdefault("tree_method", "hist")
        if classification:
            values.setdefault("eval_metric", "logloss")
            return XGBClassifier(**values)
        return XGBRegressor(**values)

    if model_id == "lightgbm":
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError(
                "Group external_boosting requires lightgbm. Install project "
                "dependencies before selecting this group."
            ) from error
        values.setdefault("n_estimators", 300)
        values.setdefault("random_state", settings.random_state)
        values.setdefault("n_jobs", 1)
        values.setdefault("verbosity", -1)
        return (LGBMClassifier if classification else LGBMRegressor)(**values)

    if model_id == "catboost":
        try:
            import catboost  # noqa: F401
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError(
                "Group native_categorical requires catboost. Install project "
                "dependencies before selecting this group."
            ) from error
        values.setdefault("iterations", 300)
        values.setdefault("random_seed", settings.random_state)
        values.setdefault("verbose", False)
        values.setdefault("allow_writing_files", False)
        values.setdefault("thread_count", 1)
        values.setdefault("cat_features", list(categorical_features))
        adapter = CatBoostClassifierAdapter if classification else CatBoostRegressorAdapter
        return adapter(values)

    raise ValueError(f"Unsupported screening model_id: {model_id!r}")


__all__ = [
    "NativeCategoricalPreprocessor",
    "CatBoostClassifierAdapter",
    "CatBoostRegressorAdapter",
    "PREPROCESSING_PROFILES",
    "build_native_categorical_preprocessor",
    "build_screening_estimator",
    "settings_for_preprocessing_profile",
]
