"""EXP-002: replace this example with one controlled experiment."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ml_project.modeling import (
    ModelingSettings,
    ExperimentData,
    ExperimentSettings,
)


EXPERIMENT = ExperimentSettings(
    experiment_id="EXP-002",
    experiment_title="CHANGE ME — controlled experiment",
    experiment_note=Path("experiments/EXP-002 Controlled Experiment.md"),
    hypothesis="CHANGE ME — if ..., then ..., because ...",
    change_description="CHANGE ME — exactly one controlled change",
    success_criterion=(
        "Primary improvement >= +0.0000; add explicit metric guardrails below."
    ),
    primary_improvement_min=0.0,
    metric_guardrails={
        # "Recall": -0.005,
    },
    reference_model="baseline_reference",
    primary_candidate="candidate",
    experiment_parameters={},
    run_name="exp_002_v1",
    artifact_dir=Path("artifacts/experiments"),
    results_registry=Path("experiments/results.csv"),
    save_artifacts=True,
    save_metric_figures=True,
    metric_figure_dpi=160,
    save_final_model=False,
    sync_experiment_note=True,
    sync_docs=True,
    allow_overwrite=True,
)


def prepare_candidate_data(
    train: pd.DataFrame,
    feature_groups: Mapping[str, Any],
    reference_settings: ModelingSettings,
) -> ExperimentData:
    """Подготовить данные candidate поверх настроек текущего reference."""

    # reference_settings — настройки сравниваемой модели: EXP-001 baseline
    # либо принятого parent-чемпиона. Если эксперимент меняет отбор признаков,
    # preprocessing или estimator, создайте candidate_settings через replace.
    # Validation contract обычный эксперимент наследует без изменений.
    candidate_settings = reference_settings

    return ExperimentData(
        frame=train.copy(deep=True),
        feature_groups=copy.deepcopy(feature_groups),
        settings=candidate_settings,
        diagnostics={},
    )


def build_candidate_models(
    preprocessor: Any,
    candidate_settings: ModelingSettings,
    experiment_settings: ExperimentSettings,
) -> dict[str, Any]:
    """Собрать candidate с уже подготовленными candidate_settings."""

    raise NotImplementedError(
        "CHANGE ME — build one candidate Pipeline for EXP-002"
    )


__all__ = [
    "EXPERIMENT",
    "build_candidate_models",
    "prepare_candidate_data",
]
