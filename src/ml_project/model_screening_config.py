"""Редактируемый конфиг группового screening моделей для notebook 06."""

from pathlib import Path

from .modeling.contracts import (
    ModelGroupSettings,
    ModelScreeningSettings,
    ScreeningModelSpec,
)


# Пример настроен для classification. После EDA замените параметры и выключите
# ненужные модели через enabled=False. Один MS-запуск хранит одну группу.
MODEL_GROUPS = {
    "classic_scaled": ModelGroupSettings(
        group_id="classic_scaled",
        title="Масштабируемые классические модели",
        preprocessing_profile="scaled_dense",
        purpose="Проверить линейную границу и локальную геометрию KNN.",
        models=(
            ScreeningModelSpec(
                "linear_reference",
                "Linear reference — alternative regularization",
                {
                    "classification": {"C": 0.5, "solver": "liblinear"},
                    "regression": {"alpha": 1.0},
                },
            ),
            ScreeningModelSpec(
                "knn",
                "K-nearest neighbors",
                {"n_neighbors": 15, "weights": "distance", "p": 2},
            ),
        ),
    ),
    "tree_bagging": ModelGroupSettings(
        group_id="tree_bagging",
        title="Деревья и bagging",
        preprocessing_profile="unscaled_sparse",
        purpose="Проверить нелинейные зависимости и ансамбли деревьев.",
        models=(
            ScreeningModelSpec(
                "decision_tree", "Decision tree", {"max_depth": 5, "min_samples_leaf": 8}
            ),
            ScreeningModelSpec(
                "random_forest",
                "Random forest",
                {"n_estimators": 400, "max_depth": 7, "min_samples_leaf": 3, "max_features": "sqrt"},
            ),
            ScreeningModelSpec(
                "extra_trees",
                "Extra Trees",
                {"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 3, "max_features": "sqrt"},
            ),
        ),
    ),
    "sklearn_boosting": ModelGroupSettings(
        group_id="sklearn_boosting",
        title="Boosting из scikit-learn",
        preprocessing_profile="unscaled_dense",
        purpose="Быстрый boosting-screen без внешних библиотек.",
        models=(
            ScreeningModelSpec(
                "hist_gradient_boosting",
                "Histogram gradient boosting",
                {"learning_rate": 0.06, "max_iter": 250, "max_leaf_nodes": 15},
            ),
            ScreeningModelSpec(
                "gradient_boosting",
                "Gradient boosting",
                {"n_estimators": 250, "learning_rate": 0.04, "max_depth": 2},
            ),
        ),
    ),
    "external_boosting": ModelGroupSettings(
        group_id="external_boosting",
        title="XGBoost и LightGBM",
        preprocessing_profile="unscaled_sparse",
        purpose="Сравнить внешние реализации градиентного boosting.",
        models=(
            ScreeningModelSpec(
                "xgboost",
                "XGBoost",
                {"n_estimators": 350, "learning_rate": 0.04, "max_depth": 3, "subsample": 0.85, "colsample_bytree": 0.85},
            ),
            ScreeningModelSpec(
                "lightgbm",
                "LightGBM",
                {"n_estimators": 350, "learning_rate": 0.04, "num_leaves": 15, "max_depth": 5, "subsample": 0.85, "colsample_bytree": 0.85},
            ),
        ),
    ),
    "native_categorical": ModelGroupSettings(
        group_id="native_categorical",
        title="CatBoost с нативными категориями",
        preprocessing_profile="native_categorical",
        purpose="Проверить категории без one-hot encoding.",
        models=(
            ScreeningModelSpec(
                "catboost",
                "CatBoost",
                {"iterations": 400, "learning_rate": 0.04, "depth": 5, "l2_leaf_reg": 5.0},
            ),
        ),
    ),
}


SCREENING = ModelScreeningSettings(
    screening_id="MS-001",
    screening_title="Tree and bagging screening",
    screening_note=Path("model-screening/MS-001 Tree Bagging.md"),

    # None означает EXP-001 baseline. После feature engineering укажите модуль
    # принятого experiment-чемпиона, например ml_project.experiments.exp_003_*.
    feature_reference_module=None,
    active_group="tree_bagging",
    groups=MODEL_GROUPS,
    reference_model_id="feature_champion",
    diagnostic_model_id="random_forest",
    shortlist_size=2,
    run_name="ms_001_tree_bagging_v1",
    artifact_dir=Path("artifacts/model-screening"),
    results_registry=Path("model-screening/results.csv"),
    save_artifacts=True,
    save_figures=True,
    figure_dpi=160,
    sync_screening_note=True,
    sync_registry=True,
    allow_overwrite=True,
)


__all__ = ["MODEL_GROUPS", "SCREENING"]
