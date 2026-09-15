"""Universal reconstruction, full-train fit and Kaggle-style submission reporting."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..docsync import MarkdownDocument, dataframe_to_markdown
from .features import (
    build_tabular_preprocessor,
    prepare_training_data,
    resolve_feature_plan,
)
from .screening import (
    build_screening_candidate_pipeline,
    prepare_screening_context,
)


EXPERIMENT_RESULTS = Path("experiments/results.csv")
SCREENING_RESULTS = Path("model-screening/results.csv")
SUBMISSION_ROOT = Path("artifacts/submissions")
SUBMISSION_REGISTRY = Path("submissions/results.csv")
SUBMISSION_INDEX = Path("submissions/_index.md")


@dataclass(frozen=True)
class PreparedSubmissionCandidate:
    """One exact tested candidate rebuilt for full-train inference."""

    candidate_id: str
    source_type: str
    source_id: str
    feature_reference: str
    feature_module: str | None
    recorded_source_sha256: str | None
    current_source_sha256: str | None
    model_id: str
    model_label: str
    decision: str
    cv_metric: str | None
    cv_mean: float | None
    cv_std: float | None
    pipeline: Any
    train_X: pd.DataFrame
    train_y: pd.Series
    inference_X: pd.DataFrame
    inference_keys: pd.Series
    feature_names: tuple[str, ...]
    model_params: Mapping[str, Any]
    inference_audit: pd.DataFrame

    @property
    def inference_ready(self) -> bool:
        return self.inference_audit.empty


@dataclass(frozen=True)
class SubmissionPrediction:
    """A full-train fitted pipeline and a validated submission frame."""

    prepared: PreparedSubmissionCandidate
    fitted_pipeline: Any
    frame: pd.DataFrame
    validation: pd.DataFrame
    prediction_distribution: pd.DataFrame


@dataclass(frozen=True)
class SavedSubmission:
    """Paths created by an explicit final notebook save action."""

    run_dir: Path
    submission_path: Path
    model_path: Path
    metadata_path: Path
    note_path: Path
    registry_path: Path
    index_path: Path


def _safe_project_path(root: Path, relative: Path, label: str) -> Path:
    if relative.is_absolute():
        raise ValueError(f"{label} must be relative to the project root")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the project root: {relative}") from error
    return resolved


def _read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _optional_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _truthy(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _experiment_model_label(root: Path, run_name: str, fallback: str) -> str:
    metadata_path = root / "artifacts/experiments" / run_name / "metadata.json"
    if not metadata_path.is_file():
        return fallback
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback
    return str(metadata.get("model") or fallback)


def candidate_catalog(
    project_root: str | Path,
    *,
    baseline_model_label: str = "baseline model",
) -> pd.DataFrame:
    """Combine baseline, controlled EXP and model-screening candidates."""

    root = Path(project_root).resolve()
    experiment_registry = _read_optional_csv(root / EXPERIMENT_RESULTS)
    screening_registry = _read_optional_csv(root / SCREENING_RESULTS)
    rows: list[dict[str, Any]] = [
        {
            "candidate_id": "EXP-001/model",
            "source_type": "baseline",
            "source_id": "EXP-001",
            "feature_reference": "EXP-001",
            "feature_module": None,
            "recorded_source_sha256": None,
            "group": "baseline",
            "preprocessing_profile": "baseline_exact",
            "model_id": "model",
            "model": baseline_model_label,
            "cv_metric": None,
            "cv_mean": np.nan,
            "cv_std": np.nan,
            "decision": "baseline",
            "params": "{}",
            "reference_model": "model",
            "rank": 1,
        }
    ]

    if not experiment_registry.empty:
        latest_experiments = experiment_registry.drop_duplicates(
            subset=["experiment_id"], keep="last"
        )
        for item in latest_experiments.to_dict("records"):
            candidate_name = str(item.get("candidate") or "candidate")
            run_name = str(item.get("run_name") or "")
            rows.append(
                {
                    "candidate_id": f"{item['experiment_id']}/{candidate_name}",
                    "source_type": "experiment",
                    "source_id": str(item["experiment_id"]),
                    "feature_reference": str(item["experiment_id"]),
                    "feature_module": _optional_text(item.get("implementation_module")),
                    "recorded_source_sha256": _optional_text(
                        item.get("implementation_sha256")
                    ),
                    "group": "controlled_experiment",
                    "preprocessing_profile": "experiment_exact",
                    "model_id": candidate_name,
                    "model": _experiment_model_label(
                        root, run_name, "experiment candidate"
                    ),
                    "cv_metric": _optional_text(item.get("primary_metric")),
                    "cv_mean": _optional_float(item.get("candidate_score")),
                    "cv_std": _optional_float(item.get("candidate_std")),
                    "decision": str(item.get("decision") or "pending"),
                    "params": "{}",
                    "reference_model": str(item.get("candidate") or "candidate"),
                    "rank": 1,
                }
            )

    known_experiment_candidates = {
        str(row["candidate_id"])
        for row in rows
        if row["source_type"] == "experiment"
    }
    experiment_module_dir = root / "src/ml_project/experiments"
    if experiment_module_dir.is_dir():
        from ..experiment import load_experiment

        for source_path in sorted(experiment_module_dir.glob("exp_*.py")):
            module_name = f"ml_project.experiments.{source_path.stem}"
            try:
                definition = load_experiment(
                    module_name,
                    reload_module=False,
                )
            except (ImportError, TypeError, ValueError):
                continue
            candidate_name = definition.settings.primary_candidate
            candidate_id = (
                f"{definition.settings.experiment_id}/{candidate_name}"
            )
            if candidate_id in known_experiment_candidates:
                continue
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "source_type": "experiment",
                    "source_id": definition.settings.experiment_id,
                    "feature_reference": definition.settings.experiment_id,
                    "feature_module": definition.module_name,
                    "recorded_source_sha256": definition.source_sha256,
                    "group": "controlled_experiment",
                    "preprocessing_profile": "experiment_exact",
                    "model_id": candidate_name,
                    "model": "experiment candidate",
                    "cv_metric": None,
                    "cv_mean": np.nan,
                    "cv_std": np.nan,
                    "decision": definition.settings.decision,
                    "params": "{}",
                    "reference_model": candidate_name,
                    "rank": 1,
                }
            )

    experiment_modules: dict[str, str] = {}
    if not experiment_registry.empty:
        for item in experiment_registry.drop_duplicates(
            subset=["experiment_id"], keep="last"
        ).to_dict("records"):
            module = _optional_text(item.get("implementation_module"))
            if module:
                experiment_modules[str(item["experiment_id"])] = module

    if not screening_registry.empty:
        for item in screening_registry.to_dict("records"):
            screening_id = str(item["screening_id"])
            model_id = str(item["model"])
            feature_reference = str(item["feature_reference"])
            feature_module = _optional_text(item.get("feature_reference_module"))
            if feature_module is None:
                feature_module = experiment_modules.get(feature_reference)
            rows.append(
                {
                    "candidate_id": f"{screening_id}/{model_id}",
                    "source_type": "screening",
                    "source_id": screening_id,
                    "feature_reference": feature_reference,
                    "feature_module": feature_module,
                    "recorded_source_sha256": _optional_text(
                        item.get("feature_module_sha256")
                    ),
                    "group": str(item["group"]),
                    "preprocessing_profile": _optional_text(
                        item.get("preprocessing_profile")
                    ),
                    "model_id": model_id,
                    "model": model_id,
                    "cv_metric": _optional_text(item.get("primary_metric")),
                    "cv_mean": _optional_float(item.get("mean")),
                    "cv_std": _optional_float(item.get("std")),
                    "decision": (
                        "shortlisted" if _truthy(item.get("shortlisted")) else "screened"
                    ),
                    "params": str(item.get("params") or "{}"),
                    "reference_model": (
                        _optional_text(item.get("reference_model"))
                        or "feature_champion"
                    ),
                    "rank": int(item.get("rank") or 0),
                }
            )

    catalog = pd.DataFrame(rows)
    source_order = pd.Categorical(
        catalog["source_type"],
        categories=["baseline", "experiment", "screening"],
        ordered=True,
    )
    catalog = (
        catalog.assign(_source_order=source_order)
        .sort_values(
            ["_source_order", "source_id", "rank", "candidate_id"],
            kind="stable",
        )
        .drop(columns="_source_order")
        .reset_index(drop=True)
    )
    return catalog


def candidate_catalog_report(catalog: pd.DataFrame) -> pd.DataFrame:
    """Return the compact table intended for the notebook selector."""

    report = catalog[
        [
            "candidate_id",
            "source_type",
            "feature_reference",
            "model",
            "cv_metric",
            "cv_mean",
            "cv_std",
            "decision",
        ]
    ].copy()
    report["CV mean ± std"] = report.apply(
        lambda row: (
            "—"
            if pd.isna(row["cv_mean"])
            else (
                f"{float(row['cv_mean']):.4f}"
                if pd.isna(row["cv_std"])
                else f"{float(row['cv_mean']):.4f} ± {float(row['cv_std']):.4f}"
            )
        ),
        axis=1,
    )
    return report.drop(columns=["cv_mean", "cv_std"]).rename(
        columns={
            "candidate_id": "Candidate ID",
            "source_type": "Источник",
            "feature_reference": "Feature set",
            "model": "Модель",
            "cv_metric": "Метрика",
            "decision": "Статус",
        }
    )


def _selected_record(catalog: pd.DataFrame, candidate_id: str) -> dict[str, Any]:
    selected = catalog[catalog["candidate_id"].astype(str).eq(candidate_id)]
    if selected.empty:
        available = ", ".join(catalog["candidate_id"].astype(str))
        raise KeyError(
            f"Unknown candidate {candidate_id!r}. Available candidates: {available}"
        )
    if len(selected) != 1:
        raise ValueError(f"Candidate ID is not unique: {candidate_id}")
    return dict(selected.iloc[0])


def _compare_feature_values(left: pd.Series, right: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        return np.isclose(
            pd.to_numeric(left, errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(right, errors="coerce").to_numpy(dtype=float),
            equal_nan=True,
        )
    left_text = left.astype("string").fillna("__NA__").to_numpy(dtype=str)
    right_text = right.astype("string").fillna("__NA__").to_numpy(dtype=str)
    return left_text == right_text


def _experiment_inference_audit(
    definition: Any,
    raw_train: pd.DataFrame,
    prepared_full: pd.DataFrame,
    feature_groups: Mapping[str, Any],
    initial_settings: Any,
    feature_names: tuple[str, ...],
) -> pd.DataFrame:
    """Detect raw feature hooks whose values depend on the surrounding rows."""

    from ..experiment import prepare_experiment_candidate

    rows: list[dict[str, Any]] = []
    for modulus in (2, 3):
        for remainder in range(modulus):
            positions = np.flatnonzero(
                np.arange(len(raw_train), dtype=int) % modulus == remainder
            )
            if not len(positions):
                continue
            subset = raw_train.iloc[positions].copy()
            prepared_subset = prepare_experiment_candidate(
                definition,
                subset,
                feature_groups,
                initial_settings,
            ).frame
            full_subset = prepared_full.loc[subset.index]
            for feature in feature_names:
                if feature not in prepared_subset or feature not in full_subset:
                    rows.append(
                        {
                            "feature": feature,
                            "partition": f"index % {modulus} == {remainder}",
                            "changed_rows": len(subset),
                            "reason": "feature is missing after subset transformation",
                        }
                    )
                    continue
                equal = _compare_feature_values(
                    full_subset[feature].reset_index(drop=True),
                    prepared_subset[feature].reset_index(drop=True),
                )
                changed = int((~equal).sum())
                if changed:
                    rows.append(
                        {
                            "feature": feature,
                            "partition": f"index % {modulus} == {remainder}",
                            "changed_rows": changed,
                            "reason": (
                                "raw feature changes when neighbouring rows are removed"
                            ),
                        }
                    )
    if not rows:
        return pd.DataFrame(
            columns=["feature", "partition", "changed_rows", "reason"]
        )
    return (
        pd.DataFrame(rows)
        .groupby(["feature", "reason"], as_index=False)
        .agg(
            changed_rows=("changed_rows", "max"),
            partitions=("partition", "nunique"),
        )
        [["feature", "changed_rows", "partitions", "reason"]]
    )


def _align_inference_input(
    train_frame: pd.DataFrame,
    inference_frame: pd.DataFrame,
    *,
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_X = train_frame.drop(columns=[target], errors="ignore").copy()
    inference_X = inference_frame.drop(columns=[target], errors="ignore").copy()
    missing = [column for column in train_X.columns if column not in inference_X]
    if missing:
        raise ValueError(
            "Inference dataset lacks pipeline input columns: " + ", ".join(missing)
        )
    return train_X, inference_X.loc[:, list(train_X.columns)].copy()


def _decoded_params(value: Any, model_id: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        params = dict(value)
    else:
        params = json.loads(str(value or "{}"))
    if model_id == "catboost" and set(params) == {"params"}:
        nested = params["params"]
        if isinstance(nested, Mapping):
            return dict(nested)
    return params


def prepare_submission_candidate(
    project_root: str | Path,
    selected_candidate: str,
    train: pd.DataFrame,
    inference: pd.DataFrame,
    feature_groups: Mapping[str, Any],
    *,
    target: str,
    key: str,
    initial_settings: Any,
    model_groups: Mapping[str, Any] | None = None,
) -> PreparedSubmissionCandidate:
    """Rebuild one catalog candidate and audit train-serving parity."""

    from ..experiment import (
        build_experiment_candidates,
        load_experiment,
        prepare_experiment_candidate,
    )
    from .estimators import build_model_pipeline, build_simple_estimator

    root = Path(project_root).resolve()
    train = train.reset_index(drop=True).copy()
    inference = inference.reset_index(drop=True).copy()
    if target not in train:
        raise KeyError(f"TARGET {target!r} is absent from train")
    if key not in train or key not in inference:
        raise KeyError(f"KEY {key!r} must exist in both train and inference")
    if train[key].duplicated().any() or inference[key].duplicated().any():
        raise ValueError(f"KEY {key!r} must be unique in train and inference")

    catalog = candidate_catalog(root)
    record = _selected_record(catalog, selected_candidate)
    source_type = str(record["source_type"])
    feature_module = _optional_text(record.get("feature_module"))
    current_source_sha256: str | None = None

    if source_type == "baseline":
        plan = resolve_feature_plan(
            train,
            feature_groups,
            target=target,
            key=key,
            settings=initial_settings,
        )
        prepared_data = prepare_training_data(
            train,
            target=target,
            plan=plan,
            settings=initial_settings,
        )
        preprocessor = build_tabular_preprocessor(initial_settings, plan)
        pipeline = build_model_pipeline(
            preprocessor,
            build_simple_estimator(initial_settings),
        )
        train_frame = train
        inference_frame = inference
        audit = pd.DataFrame(
            columns=["feature", "changed_rows", "partitions", "reason"]
        )
        model_params = pipeline.named_steps["model"].get_params(deep=False)
        train_y = prepared_data.y
    elif source_type == "experiment":
        if not feature_module:
            raise ValueError(
                f"{selected_candidate} has no implementation_module in "
                f"{EXPERIMENT_RESULTS}"
            )
        definition = load_experiment(feature_module)
        current_source_sha256 = definition.source_sha256
        prepared_train = prepare_experiment_candidate(
            definition,
            train,
            feature_groups,
            initial_settings,
        )
        prepared_inference = prepare_experiment_candidate(
            definition,
            inference,
            feature_groups,
            initial_settings,
        )
        plan = resolve_feature_plan(
            prepared_train.frame,
            prepared_train.feature_groups,
            target=target,
            key=key,
            settings=prepared_train.settings,
        )
        prepared_data = prepare_training_data(
            prepared_train.frame,
            target=target,
            plan=plan,
            settings=prepared_train.settings,
        )
        preprocessor = build_tabular_preprocessor(prepared_train.settings, plan)
        candidates = build_experiment_candidates(
            definition,
            preprocessor,
            prepared_train.settings,
        )
        model_id = str(record["model_id"])
        if model_id not in candidates:
            raise KeyError(
                f"Experiment {record['source_id']} has no model {model_id!r}"
            )
        pipeline = candidates[model_id]
        train_frame = prepared_train.frame.loc[prepared_data.row_index]
        inference_frame = prepared_inference.frame
        audit = _experiment_inference_audit(
            definition,
            train,
            prepared_train.frame,
            feature_groups,
            initial_settings,
            tuple(plan.model_features),
        )
        model_params = pipeline.named_steps["model"].get_params(deep=False)
        train_y = prepared_data.y
    elif source_type == "screening":
        feature_definition = None
        context = prepare_screening_context(
            root,
            train,
            feature_groups,
            target=target,
            key=key,
            initial_settings=initial_settings,
            feature_reference_module=feature_module,
        )
        if feature_module:
            feature_definition = load_experiment(
                feature_module,
                reload_module=False,
            )
            current_source_sha256 = feature_definition.source_sha256
        if feature_definition is None:
            inference_frame = inference.copy()
            audit = pd.DataFrame(
                columns=["feature", "changed_rows", "partitions", "reason"]
            )
        else:
            prepared_inference = prepare_experiment_candidate(
                feature_definition,
                inference,
                feature_groups,
                initial_settings,
            )
            inference_frame = prepared_inference.frame
            audit = _experiment_inference_audit(
                feature_definition,
                train,
                context.frame,
                feature_groups,
                initial_settings,
                tuple(context.plan.model_features),
            )
        model_id = str(record["model_id"])
        reference_model = str(record.get("reference_model") or "feature_champion")
        if model_id == reference_model:
            pipeline = context.reference_pipeline
            model_params = pipeline.named_steps["model"].get_params(deep=False)
        else:
            if model_groups is None:
                raise ValueError(
                    "model_groups are required for a screening candidate"
                )
            group_id = str(record["group"])
            if group_id not in model_groups:
                raise KeyError(
                    f"Screening group {group_id!r} is absent from MODEL_GROUPS"
                )
            group = model_groups[group_id]
            recorded_profile = _optional_text(record.get("preprocessing_profile"))
            if (
                recorded_profile is not None
                and recorded_profile != group.preprocessing_profile
            ):
                raise ValueError(
                    f"Screening preprocessing profile changed: recorded "
                    f"{recorded_profile!r}, current {group.preprocessing_profile!r}"
                )
            model_params = _decoded_params(record.get("params"), model_id)
            pipeline = build_screening_candidate_pipeline(
                context,
                group,
                model_id,
                model_params,
            )
        plan = context.plan
        train_frame = context.frame.loc[context.data.row_index]
        train_y = context.data.y
    else:  # pragma: no cover - catalog controls values
        raise ValueError(f"Unsupported candidate source_type: {source_type}")

    train_X, inference_X = _align_inference_input(
        train_frame,
        inference_frame,
        target=target,
    )
    final_model = getattr(pipeline, "named_steps", {}).get("model")
    model_label = type(final_model).__name__ if final_model is not None else type(pipeline).__name__

    return PreparedSubmissionCandidate(
        candidate_id=selected_candidate,
        source_type=source_type,
        source_id=str(record["source_id"]),
        feature_reference=str(record["feature_reference"]),
        feature_module=feature_module,
        recorded_source_sha256=_optional_text(
            record.get("recorded_source_sha256")
        ),
        current_source_sha256=current_source_sha256,
        model_id=str(record["model_id"]),
        model_label=model_label,
        decision=str(record["decision"]),
        cv_metric=_optional_text(record.get("cv_metric")),
        cv_mean=_optional_float(record.get("cv_mean")),
        cv_std=_optional_float(record.get("cv_std")),
        pipeline=pipeline,
        train_X=train_X,
        train_y=train_y,
        inference_X=inference_X,
        inference_keys=inference[key].copy(),
        feature_names=tuple(plan.model_features),
        model_params=dict(model_params),
        inference_audit=audit,
    )


def submission_contract_report(
    prepared: PreparedSubmissionCandidate,
) -> pd.DataFrame:
    """Explain exactly what will be trained before the expensive fit."""

    recorded = prepared.recorded_source_sha256
    current = prepared.current_source_sha256
    code_status = (
        "not applicable"
        if recorded is None and current is None
        else "match"
        if recorded == current
        else "changed since recorded CV"
    )
    cv_value = (
        "—"
        if prepared.cv_mean is None
        else (
            f"{prepared.cv_mean:.4f}"
            if prepared.cv_std is None
            else f"{prepared.cv_mean:.4f} ± {prepared.cv_std:.4f}"
        )
    )
    return pd.DataFrame(
        [
            ("Candidate ID", prepared.candidate_id),
            ("Источник", f"{prepared.source_type}: {prepared.source_id}"),
            ("Feature set", prepared.feature_reference),
            ("Feature module", prepared.feature_module or "baseline"),
            ("Модель", f"{prepared.model_id} → {prepared.model_label}"),
            ("Исходное решение", prepared.decision),
            ("CV", f"{prepared.cv_metric or '—'}: {cv_value}"),
            ("Model features", len(prepared.feature_names)),
            ("Train rows", len(prepared.train_X)),
            ("Inference rows", len(prepared.inference_X)),
            ("Code status", code_status),
            (
                "Inference audit",
                "passed"
                if prepared.inference_ready
                else f"blocked: {len(prepared.inference_audit)} unstable features",
            ),
        ],
        columns=["Поле", "Значение"],
    )


def fit_submission_candidate(
    prepared: PreparedSubmissionCandidate,
    sample_submission: pd.DataFrame | None,
    *,
    key: str,
    target: str,
) -> SubmissionPrediction:
    """Fit on all train rows and build a sample-aligned prediction file."""

    if not prepared.inference_ready:
        features = ", ".join(
            prepared.inference_audit["feature"].astype(str).drop_duplicates()
        )
        raise ValueError(
            "Candidate raw feature preparation is not inference-safe. "
            "The following features depend on neighbouring rows: "
            f"{features}. Move learned/group statistics into a fitted transformer "
            "before creating a submission."
        )
    if sample_submission is not None:
        if key not in sample_submission or target not in sample_submission:
            raise KeyError(
                f"Sample submission must contain {key!r} and {target!r}"
            )
        if sample_submission[key].duplicated().any():
            raise ValueError(f"Sample submission KEY {key!r} is not unique")
        if len(sample_submission) != len(prepared.inference_X):
            raise ValueError(
                "Sample submission row count differs from inference row count"
            )

    try:
        from sklearn.base import clone
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError("Submission fitting requires scikit-learn") from error

    fitted = clone(prepared.pipeline)
    fitted.fit(prepared.train_X, prepared.train_y)
    predictions = np.asarray(fitted.predict(prepared.inference_X))
    if predictions.ndim != 1 or len(predictions) != len(prepared.inference_X):
        raise ValueError("Estimator returned an invalid prediction shape")

    prediction_by_key = pd.Series(
        predictions,
        index=prepared.inference_keys.to_numpy(),
    )
    output = (
        sample_submission.copy()
        if sample_submission is not None
        else pd.DataFrame({key: prepared.inference_keys.to_numpy()})
    )
    output[target] = output[key].map(prediction_by_key)
    if output[target].isna().any():
        missing = output.loc[output[target].isna(), key].head(10).tolist()
        raise ValueError(
            "Sample submission contains keys absent from inference: "
            + ", ".join(map(str, missing))
        )
    if sample_submission is not None:
        try:
            output[target] = output[target].astype(
                sample_submission[target].dtype
            )
        except (TypeError, ValueError):
            pass

    same_key_set = set(output[key]) == set(prepared.inference_keys)
    validation = pd.DataFrame(
        [
            ("rows", len(output), len(prepared.inference_X), len(output) == len(prepared.inference_X)),
            ("unique key", output[key].nunique(), len(output), output[key].is_unique),
            ("key set", len(set(output[key])), len(set(prepared.inference_keys)), same_key_set),
            ("missing predictions", int(output[target].isna().sum()), 0, not output[target].isna().any()),
        ],
        columns=["check", "observed", "expected", "passed"],
    )
    if not validation["passed"].all():
        failed = ", ".join(validation.loc[~validation["passed"], "check"])
        raise ValueError(f"Submission validation failed: {failed}")

    distribution = (
        output[target]
        .value_counts(dropna=False)
        .rename_axis("prediction")
        .reset_index(name="rows")
    )
    distribution["share"] = distribution["rows"] / len(output)
    return SubmissionPrediction(
        prepared=prepared,
        fitted_pipeline=fitted,
        frame=output,
        validation=validation,
        prediction_distribution=distribution,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serialize_fitted_pipeline(pipeline: Any, path: Path) -> str:
    """Serialize a fitted model, including classes reloaded in a notebook."""

    try:
        import joblib
    except ImportError as error:  # pragma: no cover - project dependency
        raise ImportError(
            "Saving a fitted submission model requires joblib"
        ) from error

    try:
        joblib.dump(pipeline, path)
        return "joblib"
    except (pickle.PicklingError, AttributeError, TypeError):
        # Notebook reloads can leave a fitted sklearn Pipeline referring to the
        # previous identity of a custom transformer class. Cloudpickle stores
        # that class by value, while the resulting artifact remains readable
        # through joblib.load in the recorded project environment.
        path.unlink(missing_ok=True)
        from joblib.externals import cloudpickle

        with path.open("wb") as stream:
            cloudpickle.dump(pipeline, stream)
        return "cloudpickle"


def _write_submission_artifacts(
    frame: pd.DataFrame,
    fitted_pipeline: Any,
    submission_path: Path,
    model_path: Path,
) -> str:
    """Stage CSV/model before replacing the visible submission artifacts."""

    csv_temporary = submission_path.with_name(f".{submission_path.name}.tmp")
    model_temporary = model_path.with_name(f".{model_path.name}.tmp")
    csv_temporary.unlink(missing_ok=True)
    model_temporary.unlink(missing_ok=True)
    try:
        frame.to_csv(csv_temporary, index=False)
        serializer = _serialize_fitted_pipeline(
            fitted_pipeline,
            model_temporary,
        )
        os.replace(model_temporary, model_path)
        os.replace(csv_temporary, submission_path)
        return serializer
    finally:
        csv_temporary.unlink(missing_ok=True)
        model_temporary.unlink(missing_ok=True)


def _environment_snapshot() -> dict[str, str]:
    names = ("numpy", "pandas", "scipy", "scikit-learn", "joblib")
    packages: dict[str, str] = {}
    for name in names:
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = "not installed"
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        **packages,
    }


def _frontmatter_value(path: Path, key: str) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    match = re.search(
        rf"(?m)^{re.escape(key)}:[ \t]*([^\r\n]*)$",
        text,
    )
    return match.group(1).strip() if match else ""


def _ensure_submission_note(path: Path, submission_id: str, prepared: PreparedSubmissionCandidate) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        "type: kaggle-submission\n"
        f"id: {submission_id}\n"
        "status: ready\n"
        f"candidate: {prepared.candidate_id}\n"
        f"source: {prepared.source_id}\n"
        f"model: {prepared.model_id}\n"
        "kaggle_public_score:\n"
        "kaggle_private_score:\n"
        "---\n\n"
        f"# {submission_id} — {prepared.candidate_id}\n\n"
        "<!-- auto:submission-report:start -->\n"
        "Отчёт появится после сохранения из notebook.\n"
        "<!-- auto:submission-report:end -->\n\n"
        "## Kaggle результат\n\n"
        "- Scores хранятся в полях `kaggle_public_score` и "
        "`kaggle_private_score` во frontmatter.\n"
        "- Дата отправки:\n"
        "- Наблюдение:\n",
        encoding="utf-8",
    )


def _submission_report(
    saved_relative: Path,
    prediction: SubmissionPrediction,
    dataset_versions: Mapping[str, str],
    file_sha256: str,
) -> str:
    prepared = prediction.prepared
    cv_value = (
        "—"
        if prepared.cv_mean is None
        else (
            f"{prepared.cv_mean:.4f}"
            if prepared.cv_std is None
            else f"{prepared.cv_mean:.4f} ± {prepared.cv_std:.4f}"
        )
    )
    contract = pd.DataFrame(
        [
            ("Candidate", prepared.candidate_id),
            ("Source", f"{prepared.source_type}: {prepared.source_id}"),
            ("Feature set", prepared.feature_reference),
            ("Feature module", prepared.feature_module or "baseline"),
            ("Model", prepared.model_label),
            ("CV", f"{prepared.cv_metric or '—'}: {cv_value}"),
            ("Original decision", prepared.decision),
            ("Train SHA-256", dataset_versions.get("train", "—")),
            ("Inference SHA-256", dataset_versions.get("inference", "—")),
            ("Rows", len(prediction.frame)),
            ("CSV", saved_relative.as_posix()),
            ("CSV SHA-256", file_sha256),
        ],
        columns=["Поле", "Значение"],
    )
    return (
        "## Контракт submission\n\n"
        + dataframe_to_markdown(contract, float_digits=4)
        + "\n\n## Проверки файла\n\n"
        + dataframe_to_markdown(prediction.validation, float_digits=4)
        + "\n\n## Распределение предсказаний\n\n"
        + dataframe_to_markdown(
            prediction.prediction_distribution, float_digits=4
        )
        + "\n\n> [!important]\n"
        "> Kaggle score заполняется вручную после загрузки файла. Он не заменяет "
        "локальный validation contract и сам по себе не назначает champion."
    )


def _write_submission_index(root: Path, registry: pd.DataFrame) -> Path:
    index_path = root / SUBMISSION_INDEX
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if not index_path.exists():
        index_path.write_text(
            "---\ntype: registry\nentity: kaggle-submissions\n---\n\n"
            "# Kaggle submissions\n\n"
            "Официальные CSV создаются только через "
            "[[notebooks/07_submission.ipynb]].\n\n"
            "<!-- auto:submission-registry:start -->\n"
            "Реестр появится после первого сохранения.\n"
            "<!-- auto:submission-registry:end -->\n",
            encoding="utf-8",
        )
    summary = registry.copy()
    if "kaggle_public_score" not in summary:
        summary["kaggle_public_score"] = ""
    summary["Submission"] = summary.apply(
        lambda item: f"[[{item['note']}|{item['submission_id']}]]",
        axis=1,
    )
    summary["CV"] = summary.apply(
        lambda item: (
            "—"
            if pd.isna(item["cv_mean"])
            else (
                f"{float(item['cv_mean']):.4f}"
                if pd.isna(item["cv_std"])
                else f"{float(item['cv_mean']):.4f} ± {float(item['cv_std']):.4f}"
            )
        ),
        axis=1,
    )
    summary = summary[
        [
            "Submission",
            "candidate_id",
            "feature_reference",
            "model",
            "CV",
            "rows",
            "kaggle_public_score",
        ]
    ].rename(
        columns={
            "candidate_id": "Candidate",
            "feature_reference": "Feature set",
            "model": "Model",
            "rows": "Rows",
            "kaggle_public_score": "Public score",
        }
    )
    MarkdownDocument(index_path).update_blocks(
        {"submission-registry": dataframe_to_markdown(summary, float_digits=4)}
    )
    return index_path


def _sync_submission_registry(
    root: Path,
    submission_id: str,
    prediction: SubmissionPrediction,
    note_path: Path,
    submission_path: Path,
    file_sha256: str,
) -> tuple[Path, Path]:
    prepared = prediction.prepared
    registry_path = root / SUBMISSION_REGISTRY
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "submission_id": submission_id,
        "note": note_path.relative_to(root).as_posix(),
        "candidate_id": prepared.candidate_id,
        "source_type": prepared.source_type,
        "source_id": prepared.source_id,
        "feature_reference": prepared.feature_reference,
        "model": prepared.model_label,
        "cv_metric": prepared.cv_metric or "",
        "cv_mean": prepared.cv_mean,
        "cv_std": prepared.cv_std,
        "original_decision": prepared.decision,
        "rows": len(prediction.frame),
        "submission_file": submission_path.relative_to(root).as_posix(),
        "submission_sha256": file_sha256,
        "kaggle_public_score": _frontmatter_value(
            note_path, "kaggle_public_score"
        ),
        "kaggle_private_score": _frontmatter_value(
            note_path, "kaggle_private_score"
        ),
    }
    current = _read_optional_csv(registry_path)
    if not current.empty:
        current = current[
            ~current["submission_id"].astype(str).eq(submission_id)
        ]
    registry = pd.concat([current, pd.DataFrame([row])], ignore_index=True)
    registry.to_csv(registry_path, index=False)
    index_path = _write_submission_index(root, registry)
    return registry_path, index_path


def sync_submission_scores(project_root: str | Path) -> pd.DataFrame:
    """Read manual Kaggle scores from SUB cards without fitting a model."""

    root = Path(project_root).resolve()
    registry_path = root / SUBMISSION_REGISTRY
    registry = _read_optional_csv(registry_path)
    if registry.empty:
        raise FileNotFoundError(
            "Submission registry is empty; save at least one SUB-run first"
        )
    for score_field in ("kaggle_public_score", "kaggle_private_score"):
        if score_field not in registry:
            registry[score_field] = ""
        else:
            registry[score_field] = (
                registry[score_field].fillna("").astype(str)
            )
    for index, item in registry.iterrows():
        note_path = _safe_project_path(
            root,
            Path(str(item["note"])),
            "submission note",
        )
        registry.loc[index, "kaggle_public_score"] = _frontmatter_value(
            note_path, "kaggle_public_score"
        )
        registry.loc[index, "kaggle_private_score"] = _frontmatter_value(
            note_path, "kaggle_private_score"
        )
    registry.to_csv(registry_path, index=False)
    _write_submission_index(root, registry)
    return registry[
        [
            "submission_id",
            "candidate_id",
            "kaggle_public_score",
            "kaggle_private_score",
        ]
    ].copy()


def save_submission(
    project_root: str | Path,
    submission_id: str,
    prediction: SubmissionPrediction,
    *,
    dataset_versions: Mapping[str, str],
    allow_overwrite: bool = True,
) -> SavedSubmission:
    """Persist CSV/model locally and sync the tracked SUB card and registry."""

    if not re.fullmatch(r"SUB-[0-9]{3}", submission_id):
        raise ValueError("submission_id must match SUB-001")
    root = Path(project_root).resolve()
    run_dir = _safe_project_path(
        root,
        SUBMISSION_ROOT / submission_id,
        "submission run directory",
    )
    safe_candidate = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        prediction.prepared.candidate_id,
    ).strip("_")
    submission_path = run_dir / f"{submission_id}_{safe_candidate}.csv"
    model_path = run_dir / f"{submission_id}_model.joblib"
    metadata_path = run_dir / "metadata.json"
    note_path = root / "submissions" / f"{submission_id}.md"

    contract = {
        "submission_id": submission_id,
        "candidate_id": prediction.prepared.candidate_id,
        "source_type": prediction.prepared.source_type,
        "source_id": prediction.prepared.source_id,
        "feature_reference": prediction.prepared.feature_reference,
        "feature_module": prediction.prepared.feature_module,
        "recorded_source_sha256": prediction.prepared.recorded_source_sha256,
        "current_source_sha256": prediction.prepared.current_source_sha256,
        "model_id": prediction.prepared.model_id,
        "model_label": prediction.prepared.model_label,
        "model_params": prediction.prepared.model_params,
        "dataset_versions": dict(dataset_versions),
        "rows": len(prediction.frame),
        "columns": list(prediction.frame.columns),
    }
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        previous_contract = previous.get("contract", {})
        if previous_contract != contract:
            raise FileExistsError(
                f"{submission_id} already exists with another contract. "
                "Use a new SUB-ID."
            )
        if not allow_overwrite:
            raise FileExistsError(
                f"{submission_id} already exists; enable overwrite or use a new ID"
            )

    run_dir.mkdir(parents=True, exist_ok=True)
    model_serializer = _write_submission_artifacts(
        prediction.frame,
        prediction.fitted_pipeline,
        submission_path,
        model_path,
    )
    file_sha256 = _sha256(submission_path)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "submission_file": submission_path.relative_to(root).as_posix(),
        "submission_sha256": file_sha256,
        "model_file": model_path.relative_to(root).as_posix(),
        "model_serializer": model_serializer,
        "environment": _environment_snapshot(),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    _ensure_submission_note(note_path, submission_id, prediction.prepared)
    MarkdownDocument(note_path).update_blocks(
        {
            "submission-report": _submission_report(
                submission_path.relative_to(root),
                prediction,
                dataset_versions,
                file_sha256,
            )
        }
    )
    registry_path, index_path = _sync_submission_registry(
        root,
        submission_id,
        prediction,
        note_path,
        submission_path,
        file_sha256,
    )
    return SavedSubmission(
        run_dir=run_dir,
        submission_path=submission_path,
        model_path=model_path,
        metadata_path=metadata_path,
        note_path=note_path,
        registry_path=registry_path,
        index_path=index_path,
    )


__all__ = [
    "PreparedSubmissionCandidate",
    "SavedSubmission",
    "SubmissionPrediction",
    "candidate_catalog",
    "candidate_catalog_report",
    "fit_submission_candidate",
    "prepare_submission_candidate",
    "save_submission",
    "sync_submission_scores",
    "submission_contract_report",
]
