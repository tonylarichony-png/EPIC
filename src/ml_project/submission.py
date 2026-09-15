"""Stable notebook-first API for selecting and saving submission candidates."""

from .modeling.submission import (
    PreparedSubmissionCandidate,
    SavedSubmission,
    SubmissionPrediction,
    candidate_catalog,
    candidate_catalog_report,
    fit_submission_candidate,
    prepare_submission_candidate,
    save_submission,
    sync_submission_scores,
    submission_contract_report,
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
