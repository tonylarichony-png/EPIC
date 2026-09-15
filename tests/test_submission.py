from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

import ml_project.baseline_config as baseline_config
import ml_project.modeling.submission as submission_impl
import ml_project.submission as submission_tools
from ml_project.modeling import ModelGroupSettings, ScreeningModelSpec


FEATURE_GROUPS = {
    "numeric": ["numeric"],
    "count": [],
    "categorical": ["category"],
    "ordinal": [],
    "text": [],
    "datetime": [],
    "identifier": ["id"],
    "ignored": [],
}


def settings_for_test():
    return replace(
        baseline_config.BASELINE,
        task_type="binary_classification",
        model_feature_groups=("numeric", "categorical"),
        include_features=(),
        exclude_features=(),
        require_inference_features=True,
        model_name="logistic_regression",
        model_params={"max_iter": 500, "random_state": 42},
        n_jobs=1,
    )


def train_frame() -> pd.DataFrame:
    numeric = np.linspace(-2.0, 2.0, 60)
    category = np.where(np.arange(60) % 3 == 0, "a", "b")
    target = (numeric + (category == "a") * 0.4 > 0).astype(int)
    return pd.DataFrame(
        {
            "id": np.arange(1, 61),
            "numeric": numeric,
            "category": category,
            "target": target,
        }
    )


def inference_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": np.arange(101, 113),
            "numeric": np.linspace(-1.5, 1.5, 12),
            "category": np.where(np.arange(12) % 2 == 0, "a", "c"),
        }
    )


class SubmissionNotebookTests(unittest.TestCase):
    def test_notebook_has_selection_and_save_cells(self) -> None:
        path = Path(__file__).resolve().parents[1] / "notebooks/07_submission.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cell_ids = {cell.get("id") for cell in notebook["cells"]}
        self.assertIn("submission-selection", cell_ids)
        self.assertIn("submission-save", cell_ids)
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if source.strip():
                compile(source, f"{path.name}:{cell.get('id')}", "exec")


class SubmissionTests(unittest.TestCase):
    def test_reloaded_notebook_model_uses_cloudpickle_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.joblib"
            with patch(
                "joblib.dump",
                side_effect=pickle.PicklingError("reloaded class identity"),
            ):
                serializer = submission_impl._serialize_fitted_pipeline(
                    {"fitted": True},
                    artifact,
                )

            self.assertEqual(serializer, "cloudpickle")
            self.assertEqual(joblib.load(artifact), {"fitted": True})

    def test_failed_model_serialization_preserves_visible_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submission_path = root / "submission.csv"
            model_path = root / "model.joblib"
            submission_path.write_text("previous csv", encoding="utf-8")
            model_path.write_bytes(b"previous model")

            with patch.object(
                submission_impl,
                "_serialize_fitted_pipeline",
                side_effect=pickle.PicklingError("cannot serialize"),
            ):
                with self.assertRaises(pickle.PicklingError):
                    submission_impl._write_submission_artifacts(
                        pd.DataFrame({"id": [1], "target": [0]}),
                        object(),
                        submission_path,
                        model_path,
                    )

            self.assertEqual(
                submission_path.read_text(encoding="utf-8"),
                "previous csv",
            )
            self.assertEqual(model_path.read_bytes(), b"previous model")
            self.assertFalse((root / ".submission.csv.tmp").exists())
            self.assertFalse((root / ".model.joblib.tmp").exists())

    def test_baseline_candidate_fits_validates_and_saves(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = submission_tools.candidate_catalog(root)
            self.assertEqual(catalog["candidate_id"].tolist(), ["EXP-001/model"])

            prepared = submission_tools.prepare_submission_candidate(
                root,
                "EXP-001/model",
                train_frame(),
                inference_frame(),
                FEATURE_GROUPS,
                target="target",
                key="id",
                initial_settings=settings_for_test(),
                model_groups={},
            )
            self.assertTrue(prepared.inference_ready)
            self.assertEqual(prepared.model_label, "LogisticRegression")

            prediction = submission_tools.fit_submission_candidate(
                prepared,
                None,
                key="id",
                target="target",
            )
            self.assertTrue(prediction.validation["passed"].all())
            self.assertEqual(prediction.frame.columns.tolist(), ["id", "target"])
            self.assertEqual(len(prediction.frame), len(inference_frame()))

            saved = submission_tools.save_submission(
                root,
                "SUB-001",
                prediction,
                dataset_versions={"train": "train-sha", "inference": "test-sha"},
            )
            self.assertTrue(saved.submission_path.is_file())
            self.assertTrue(saved.model_path.is_file())
            self.assertTrue(saved.metadata_path.is_file())
            self.assertTrue(saved.note_path.is_file())
            self.assertTrue(saved.registry_path.is_file())
            self.assertIn(
                "EXP-001/model",
                saved.note_path.read_text(encoding="utf-8"),
            )
            self.assertIn(
                "SUB-001",
                saved.index_path.read_text(encoding="utf-8"),
            )
            empty_scores = pd.read_csv(
                saved.registry_path,
                keep_default_na=False,
            ).iloc[0]
            self.assertEqual(empty_scores["kaggle_public_score"], "")
            self.assertEqual(empty_scores["kaggle_private_score"], "")
            note = saved.note_path.read_text(encoding="utf-8").replace(
                "kaggle_public_score:\n",
                "kaggle_public_score: 0.76555\n",
            )
            saved.note_path.write_text(note, encoding="utf-8")
            scores = submission_tools.sync_submission_scores(root)
            self.assertEqual(
                str(scores.iloc[0]["kaggle_public_score"]),
                "0.76555",
            )
            self.assertIn(
                "0.76555",
                saved.index_path.read_text(encoding="utf-8"),
            )

    def test_screening_candidate_uses_recorded_model_and_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_dir = root / "model-screening"
            registry_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "screening_id": "MS-TEST",
                        "title": "Test trees",
                        "note": "model-screening/MS-TEST.md",
                        "feature_reference": "EXP-001",
                        "feature_reference_module": "",
                        "feature_module_sha256": "",
                        "group": "tree_test",
                        "preprocessing_profile": "unscaled_sparse",
                        "reference_model": "feature_champion",
                        "model": "random_forest",
                        "primary_metric": "accuracy",
                        "direction": "maximize",
                        "mean": 0.8,
                        "std": 0.02,
                        "improvement_vs_reference": 0.01,
                        "rank": 1,
                        "shortlisted": True,
                        "params": json.dumps(
                            {
                                "n_estimators": 15,
                                "max_depth": 3,
                                "min_samples_leaf": 2,
                                "random_state": 42,
                                "n_jobs": 1,
                            }
                        ),
                        "run_name": "ms_test",
                        "dataset_version": "test-data",
                    }
                ]
            ).to_csv(registry_dir / "results.csv", index=False)
            group = ModelGroupSettings(
                group_id="tree_test",
                title="Trees",
                preprocessing_profile="unscaled_sparse",
                models=(
                    ScreeningModelSpec(
                        "random_forest",
                        "Forest",
                        {"n_estimators": 15, "max_depth": 3},
                    ),
                ),
            )

            prepared = submission_tools.prepare_submission_candidate(
                root,
                "MS-TEST/random_forest",
                train_frame(),
                inference_frame(),
                FEATURE_GROUPS,
                target="target",
                key="id",
                initial_settings=settings_for_test(),
                model_groups={"tree_test": group},
            )
            self.assertEqual(prepared.model_label, "RandomForestClassifier")
            self.assertEqual(prepared.model_params["n_estimators"], 15)
            prediction = submission_tools.fit_submission_candidate(
                prepared,
                None,
                key="id",
                target="target",
            )
            self.assertTrue(prediction.validation["passed"].all())


if __name__ == "__main__":
    unittest.main()
