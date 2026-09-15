"""Single editable configuration for ``notebooks/03_baseline.ipynb``.

Edit the ``BASELINE`` object below. It is already the typed object consumed by
the notebook: no globals are collected or converted elsewhere.

Dataset paths, target and feature groups remain in ``ml_project.config``.
After saving this file, rerun the notebook cell "Reload configuration".
"""

from pathlib import Path

from .modeling.contracts import ModelingSettings


# BASELINE — стартовый экземпляр ModelingSettings для EXP-001.
# В следующем эксперименте он приходит как reference_settings; candidate
# создаёт изменённую копию через dataclasses.replace.
BASELINE = ModelingSettings(
    # 1. Task — fill before the first run:
    # binary_classification / multiclass_classification / regression
    task_type=None,

    # 2. Features
    # Group names come from FEATURE_GROUPS in src/ml_project/config.py.
    model_feature_groups=("numeric", "count", "categorical", "ordinal"),
    include_features=(),  # Empty means every feature from the selected groups.
    exclude_features=(),
    require_inference_features=True,

    # 3. Metrics
    # None reads (primary_metric:: ...) from docs/00_problem.md.
    primary_scorer=None,
    secondary_scorers={
        # "F1": "f1",
        # "ROC-AUC": "roc_auc",
    },

    # 4. Validation
    cv_strategy="auto",
    n_splits=5,
    shuffle=True,
    random_state=42,
    group_column=None,
    time_column=None,
    n_jobs=-1,
    error_score="raise",
    return_train_score=False,

    # 5. Preprocessing
    numeric_imputer="median",
    numeric_fill_value=0.0,
    add_numeric_missing_indicator=False,
    numeric_scaler="standard",
    categorical_imputer="most_frequent",
    categorical_fill_value="__MISSING__",
    onehot_handle_unknown="ignore",
    onehot_min_frequency=None,
    onehot_max_categories=None,
    onehot_sparse_output=True,
    column_transformer_sparse_threshold=0.3,

    # 6. Reference models
    run_dummy_baseline=True,
    dummy_strategy="auto",
    dummy_params={},
    model_name="auto",
    model_params={},

    # 7. Run identity
    experiment_id="EXP-001",
    experiment_title="Baseline",
    experiment_note=Path("experiments/EXP-001 Baseline.md"),
    run_name="baseline_v1",
    artifact_dir=Path("artifacts/baseline"),

    # 8. Explicit write actions — disabled until the result is reviewed.
    save_artifacts=False,
    save_metric_figures=True,
    metric_figure_dpi=160,
    save_final_model=False,
    sync_docs=False,
    sync_experiment_note=True,
    allow_overwrite=True,
)
