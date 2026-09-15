"""Build and evaluate one model group on a frozen feature reference."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..experiment import (
    build_experiment_candidates,
    load_experiment,
    prepare_experiment_candidate,
    prepare_experiment_data,
    validate_settings as validate_experiment_settings,
)
from .contracts import (
    CLASSIFICATION_TASKS,
    BuiltModelGroup,
    ModelScreeningContext,
    ModelScreeningResult,
    ModelScreeningSettings,
    ModelingSettings,
)
from .estimators import build_model_pipeline, build_simple_estimator
from .features import (
    build_tabular_preprocessor,
    prepare_training_data,
    resolve_feature_plan,
)
from .model_groups import (
    PREPROCESSING_PROFILES,
    build_native_categorical_preprocessor,
    build_screening_estimator,
    settings_for_preprocessing_profile,
)
from .screening_diagnostics import (
    build_oof_diagnostics as _oof_diagnostics,
    build_screening_feature_importance as _feature_importance,
)
from .settings import validate_modeling_settings
from .validation import (
    build_cv_splitter,
    cv_protocol_description,
    evaluate_models_cv,
    resolve_scoring_plan,
)


MODEL_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def validate_screening_settings(settings: ModelScreeningSettings) -> None:
    """Fail early when the editable screening config is inconsistent."""

    errors: list[str] = []
    for label, value in (
        ("screening_id", settings.screening_id),
        ("run_name", settings.run_name),
        ("reference_model_id", settings.reference_model_id),
    ):
        if not MODEL_ID_PATTERN.fullmatch(value):
            errors.append(f"SCREENING.{label} contains unsupported characters")
    if not settings.screening_title.strip():
        errors.append("SCREENING.screening_title cannot be empty")
    if settings.feature_reference_module is not None and not (
        settings.feature_reference_module.startswith("ml_project.experiments.")
    ):
        errors.append(
            "SCREENING.feature_reference_module must be an experiment module or None"
        )
    if settings.active_group not in settings.groups:
        errors.append("SCREENING.active_group is absent from SCREENING.groups")
    if settings.shortlist_size < 1:
        errors.append("SCREENING.shortlist_size must be at least 1")
    if settings.figure_dpi < 72:
        errors.append("SCREENING.figure_dpi must be at least 72")
    if (
        settings.save_figures
        or settings.sync_screening_note
        or settings.sync_registry
    ) and not settings.save_artifacts:
        errors.append(
            "SCREENING.save_artifacts must be True for an official tracked run"
        )
    for label, path in (
        ("screening_note", settings.screening_note),
        ("artifact_dir", settings.artifact_dir),
        ("results_registry", settings.results_registry),
    ):
        if path.is_absolute():
            errors.append(f"SCREENING.{label} must be relative to the project root")
    if settings.screening_note.suffix.lower() != ".md":
        errors.append("SCREENING.screening_note must point to Markdown")
    if settings.results_registry.suffix.lower() != ".csv":
        errors.append("SCREENING.results_registry must point to CSV")

    active_ids: set[str] = set()
    for key, group in settings.groups.items():
        if key != group.group_id:
            errors.append(f"Group key {key!r} differs from group_id {group.group_id!r}")
        if group.preprocessing_profile not in PREPROCESSING_PROFILES:
            errors.append(
                f"Group {key!r} has unknown preprocessing profile "
                f"{group.preprocessing_profile!r}"
            )
        enabled = [model for model in group.models if model.enabled]
        if not enabled:
            errors.append(f"Group {key!r} has no enabled models")
        model_ids = [model.model_id for model in group.models]
        duplicates = sorted({item for item in model_ids if model_ids.count(item) > 1})
        if duplicates:
            errors.append(f"Group {key!r} repeats model IDs: {', '.join(duplicates)}")
        if key == settings.active_group:
            active_ids.update(model.model_id for model in enabled)
    if settings.reference_model_id in active_ids:
        errors.append("reference_model_id is reserved and cannot be a candidate model_id")
    if (
        settings.diagnostic_model_id is not None
        and settings.diagnostic_model_id not in active_ids
    ):
        errors.append("diagnostic_model_id must be enabled in the active group")
    if errors:
        raise ValueError(
            "Invalid ModelScreeningSettings:\n"
            + "\n".join(f"- {error}" for error in errors)
        )


def screening_settings_report(settings: ModelScreeningSettings) -> pd.DataFrame:
    """Human-readable report of the values controlling the next notebook run."""

    group = settings.groups.get(settings.active_group)
    rows = [
        ("identity", "screening_id", settings.screening_id),
        ("identity", "title", settings.screening_title),
        ("feature set", "feature_reference_module", settings.feature_reference_module or "EXP-001"),
        ("group", "active_group", settings.active_group),
        ("group", "preprocessing_profile", group.preprocessing_profile if group else "—"),
        ("diagnostics", "diagnostic_model_id", settings.diagnostic_model_id or "auto"),
        ("decision", "shortlist_size", settings.shortlist_size),
        ("write", "screening_note", settings.screening_note),
        ("write", "results_registry", settings.results_registry),
        ("write", "save_artifacts", settings.save_artifacts),
        ("write", "save_figures", settings.save_figures),
        ("write", "sync_screening_note", settings.sync_screening_note),
        ("write", "sync_registry", settings.sync_registry),
    ]
    return pd.DataFrame(rows, columns=["section", "parameter", "value"])


def configured_models_report(settings: ModelScreeningSettings) -> pd.DataFrame:
    """Show all groups and exact user-editable starter parameters."""

    rows: list[dict[str, Any]] = []
    for group_id, group in settings.groups.items():
        for model in group.models:
            rows.append(
                {
                    "active_group": group_id == settings.active_group,
                    "group": group_id,
                    "group_title": group.title,
                    "preprocessing": group.preprocessing_profile,
                    "enabled": model.enabled,
                    "model_id": model.model_id,
                    "label": model.label,
                    "params": json.dumps(
                        dict(model.params), ensure_ascii=False, sort_keys=True
                    ),
                }
            )
    return pd.DataFrame(rows)


def prepare_screening_context(
    project_root: str | Path,
    train: pd.DataFrame,
    feature_groups: Mapping[str, Sequence[str]],
    *,
    target: str,
    key: str | None,
    initial_settings: ModelingSettings,
    feature_reference_module: str | None,
) -> ModelScreeningContext:
    """Rebuild the adopted feature recipe and its exact champion pipeline."""

    root = Path(project_root).resolve()
    validate_modeling_settings(initial_settings)
    initial_plan = resolve_feature_plan(
        train,
        feature_groups,
        target=target,
        key=key,
        settings=initial_settings,
    )
    initial_data = prepare_training_data(
        train,
        target=target,
        plan=initial_plan,
        settings=initial_settings,
    )

    if feature_reference_module is None:
        preprocessor = build_tabular_preprocessor(initial_settings, initial_plan)
        reference_pipeline = build_model_pipeline(
            preprocessor,
            build_simple_estimator(initial_settings),
        )
        return ModelScreeningContext(
            feature_reference_id=initial_settings.experiment_id,
            feature_reference_module=None,
            feature_reference_sha256=None,
            frame=train.copy(deep=True),
            feature_groups={key_: tuple(value) for key_, value in feature_groups.items()},
            settings=initial_settings,
            plan=initial_plan,
            data=initial_data,
            reference_pipeline=reference_pipeline,
        )

    definition = load_experiment(feature_reference_module)
    validate_experiment_settings(definition.settings)
    if definition.settings.decision != "adopt":
        raise ValueError(
            f"Feature reference {definition.settings.experiment_id} has decision "
            f"{definition.settings.decision!r}; select an adopted experiment"
        )
    candidate = prepare_experiment_candidate(
        definition,
        train,
        feature_groups,
        initial_settings,
    )
    plan = resolve_feature_plan(
        candidate.frame,
        candidate.feature_groups,
        target=target,
        key=key,
        settings=candidate.settings,
    )
    data = prepare_experiment_data(initial_data, candidate.frame, target=target)
    preprocessor = build_tabular_preprocessor(candidate.settings, plan)
    candidate_models = build_experiment_candidates(
        definition,
        preprocessor,
        candidate.settings,
    )
    reference_pipeline = candidate_models[definition.settings.primary_candidate]
    try:
        definition.source_path.relative_to(root)
    except ValueError as error:
        raise ValueError("Feature reference module is outside the project root") from error
    return ModelScreeningContext(
        feature_reference_id=definition.settings.experiment_id,
        feature_reference_module=definition.module_name,
        feature_reference_sha256=definition.source_sha256,
        frame=candidate.frame,
        feature_groups={
            key_: tuple(value) for key_, value in candidate.feature_groups.items()
        },
        settings=candidate.settings,
        plan=plan,
        data=data,
        reference_pipeline=reference_pipeline,
    )


def _pipeline_shell(
    context: ModelScreeningContext,
    effective_settings: ModelingSettings,
    profile: str,
) -> Any:
    if profile == "native_categorical":
        preprocessor = build_native_categorical_preprocessor(
            effective_settings,
            context.plan,
        )
    else:
        preprocessor = build_tabular_preprocessor(effective_settings, context.plan)

    if context.feature_reference_module is None:
        return build_model_pipeline(
            preprocessor,
            build_simple_estimator(effective_settings),
        )
    definition = load_experiment(
        context.feature_reference_module,
        reload_module=False,
    )
    if definition.source_sha256 != context.feature_reference_sha256:
        raise RuntimeError(
            "Feature reference code changed after context preparation; reload the notebook"
        )
    candidates = build_experiment_candidates(
        definition,
        preprocessor,
        effective_settings,
    )
    return candidates[definition.settings.primary_candidate]


def build_screening_candidate_pipeline(
    context: ModelScreeningContext,
    group: ModelGroupSettings,
    model_id: str,
    params: Mapping[str, Any],
) -> Any:
    """Rebuild one screening candidate with its recorded preprocessing profile."""

    effective_settings = settings_for_preprocessing_profile(
        context.settings,
        group.preprocessing_profile,
    )
    shell = _pipeline_shell(
        context,
        effective_settings,
        group.preprocessing_profile,
    )
    try:
        from sklearn.base import clone
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError("Model screening requires scikit-learn") from error
    if "model" not in getattr(shell, "named_steps", {}):
        raise ValueError("Feature reference pipeline must expose a final 'model' step")
    estimator = build_screening_estimator(
        model_id,
        effective_settings,
        params,
        categorical_features=context.plan.categorical,
    )
    pipeline = clone(shell)
    pipeline.set_params(model=estimator)
    return pipeline


def build_model_group(
    context: ModelScreeningContext,
    settings: ModelScreeningSettings,
) -> BuiltModelGroup:
    """Build the fixed champion plus every enabled model in one active group."""

    validate_screening_settings(settings)
    group = settings.groups[settings.active_group]
    effective_settings = settings_for_preprocessing_profile(
        context.settings,
        group.preprocessing_profile,
    )
    shell = _pipeline_shell(context, effective_settings, group.preprocessing_profile)
    try:
        from sklearn.base import clone
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError("Model screening requires scikit-learn") from error
    if "model" not in getattr(shell, "named_steps", {}):
        raise ValueError("Feature reference pipeline must expose a final 'model' step")

    models: dict[str, Any] = {
        settings.reference_model_id: clone(context.reference_pipeline)
    }
    parameter_rows = [
        {
            "model_id": settings.reference_model_id,
            "label": f"Feature champion ({context.feature_reference_id})",
            "role": "fixed_reference",
            "preprocessing": "champion_exact",
            "params": json.dumps(
                context.reference_pipeline.named_steps["model"].get_params(deep=False),
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
        }
    ]
    for spec in group.models:
        if not spec.enabled:
            continue
        estimator = build_screening_estimator(
            spec.model_id,
            effective_settings,
            spec.params,
            categorical_features=context.plan.categorical,
        )
        pipeline = clone(shell)
        pipeline.set_params(model=estimator)
        models[spec.model_id] = pipeline
        parameter_rows.append(
            {
                "model_id": spec.model_id,
                "label": spec.label,
                "role": "screening_candidate",
                "preprocessing": group.preprocessing_profile,
                "params": json.dumps(
                    estimator.get_params(deep=False),
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ),
            }
        )
    return BuiltModelGroup(
        group=group,
        context=context,
        effective_settings=effective_settings,
        models=models,
        parameters=pd.DataFrame(parameter_rows),
    )


def _paired_deltas(
    evaluation: Any,
    reference_model: str,
) -> pd.DataFrame:
    validation = evaluation.fold_scores[
        evaluation.fold_scores["split"].eq("validation")
    ]
    reference = validation[validation["model"].eq(reference_model)][
        ["fold", "metric_key", "metric", "direction", "value"]
    ].rename(columns={"value": "reference_value"})
    candidates = validation[~validation["model"].eq(reference_model)].copy()
    merged = candidates.merge(
        reference,
        on=["fold", "metric_key", "metric", "direction"],
        how="left",
        validate="many_to_one",
    )
    raw_delta = merged["value"] - merged["reference_value"]
    merged["improvement"] = np.where(
        merged["direction"].eq("minimize"),
        -raw_delta,
        raw_delta,
    )
    return merged.rename(columns={"value": "candidate_value"})


def _leaderboard(
    evaluation: Any,
    paired: pd.DataFrame,
    *,
    reference_model: str,
    shortlist_size: int,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    primary = evaluation.primary_summary().copy()
    if primary.empty:
        raise ValueError("CV result has no primary validation metric")
    direction = str(primary.iloc[0]["direction"])
    primary = primary.sort_values(
        "mean",
        ascending=direction == "minimize",
    ).reset_index(drop=True)
    primary.insert(0, "rank", np.arange(1, len(primary) + 1))
    reference_mean = float(
        primary.loc[primary["model"].eq(reference_model), "mean"].iloc[0]
    )
    raw_delta = primary["mean"] - reference_mean
    primary["improvement_vs_reference"] = np.where(
        primary["direction"].eq("minimize"), -raw_delta, raw_delta
    )

    fold_primary = paired[paired["metric_key"].eq("primary")]
    win_rows = []
    for model, rows in fold_primary.groupby("model", sort=False):
        values = rows["improvement"].to_numpy(dtype=float)
        win_rows.append(
            {
                "model": model,
                "fold_wins": int(np.sum(values > 1e-12)),
                "fold_ties": int(np.sum(np.isclose(values, 0.0, atol=1e-12))),
                "fold_losses": int(np.sum(values < -1e-12)),
                "paired_delta_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        )
    primary = primary.merge(pd.DataFrame(win_rows), on="model", how="left")
    reference_mask = primary["model"].eq(reference_model)
    primary.loc[reference_mask, ["fold_wins", "fold_ties", "fold_losses"]] = 0
    primary.loc[reference_mask, "paired_delta_std"] = 0.0
    candidate_order = primary.loc[~reference_mask, "model"].tolist()
    shortlist = tuple(candidate_order[:shortlist_size])
    primary["shortlisted"] = primary["model"].isin(shortlist)
    primary["mean ± std"] = primary.apply(
        lambda row: f"{float(row['mean']):.4f} ± {float(row['std']):.4f}",
        axis=1,
    )
    return primary, shortlist



def run_model_screening(
    project_root: str | Path,
    built: BuiltModelGroup,
    settings: ModelScreeningSettings,
) -> ModelScreeningResult:
    """Evaluate one group on the reference CV contract without writing files."""

    scoring = resolve_scoring_plan(project_root, built.context.settings)
    cv, strategy = build_cv_splitter(
        built.context.settings,
        built.context.data.y,
    )
    description = cv_protocol_description(built.context.settings, strategy)
    evaluation = evaluate_models_cv(
        built.models,
        built.context.data,
        cv=cv,
        scoring=scoring,
        settings=built.context.settings,
    )
    paired = _paired_deltas(evaluation, settings.reference_model_id)
    leaderboard, shortlist = _leaderboard(
        evaluation,
        paired,
        reference_model=settings.reference_model_id,
        shortlist_size=settings.shortlist_size,
    )
    oof, prediction_comparison = _oof_diagnostics(
        evaluation,
        built.context.data,
        reference_model=settings.reference_model_id,
        classification=built.context.settings.task_type in CLASSIFICATION_TASKS,
    )
    importance = _feature_importance(evaluation, built.context.plan)
    return ModelScreeningResult(
        built=built,
        scoring=scoring,
        cv_description=description,
        evaluation=evaluation,
        leaderboard=leaderboard,
        paired_deltas=paired,
        oof_predictions=oof,
        prediction_comparison=prediction_comparison,
        feature_importance=importance,
        shortlist=shortlist,
    )



__all__ = [
    "build_screening_candidate_pipeline",
    "build_model_group",
    "configured_models_report",
    "prepare_screening_context",
    "run_model_screening",
    "screening_settings_report",
    "validate_screening_settings",
]
