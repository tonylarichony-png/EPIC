"""Stable public facade for notebook-first grouped model screening."""

from .modeling.screening import (
    build_screening_candidate_pipeline,
    build_model_group,
    configured_models_report,
    prepare_screening_context,
    run_model_screening,
    screening_settings_report,
    validate_screening_settings,
)
from .modeling.screening_reporting import (
    TRACKED_SCREENING_FIGURE_ROOT,
    build_screening_figures,
    save_model_screening,
)

__all__ = [
    "TRACKED_SCREENING_FIGURE_ROOT",
    "build_screening_candidate_pipeline",
    "build_model_group",
    "build_screening_figures",
    "configured_models_report",
    "prepare_screening_context",
    "run_model_screening",
    "save_model_screening",
    "screening_settings_report",
    "validate_screening_settings",
]
