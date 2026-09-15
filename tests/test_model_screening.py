from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import ml_project.baseline_config as baseline_config
import ml_project.model_screening as screening_tools
import ml_project.model_screening_config as screening_config
from ml_project.modeling import (
    ModelGroupSettings,
    NativeCategoricalPreprocessor,
    ScreeningModelSpec,
    build_screening_estimator,
)

try:
    import sklearn  # noqa: F401
except ImportError:  # pragma: no cover - local dependency state
    DEPENDENCIES_AVAILABLE = False
else:
    DEPENDENCIES_AVAILABLE = True


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


def sample_frame() -> pd.DataFrame:
    rows = 60
    numeric = np.linspace(-2.0, 2.0, rows)
    category = np.where(np.arange(rows) % 3 == 0, "a", "b")
    target = (numeric + (category == "a") * 0.45 > 0).astype(int)
    return pd.DataFrame(
        {"id": np.arange(rows), "numeric": numeric, "category": category, "target": target}
    )


def settings_for_test():
    return replace(
        baseline_config.BASELINE,
        task_type="binary_classification",
        include_features=(),
        exclude_features=(),
        primary_scorer="accuracy",
        secondary_scorers={"F1": "f1"},
        cv_strategy="stratified_kfold",
        n_splits=3,
        n_jobs=1,
        model_name="logistic_regression",
        model_params={"max_iter": 1000, "random_state": 42},
    )


def screening_for_test():
    group = ModelGroupSettings(
        group_id="tree_test",
        title="Trees",
        preprocessing_profile="unscaled_sparse",
        models=(
            ScreeningModelSpec(
                "decision_tree", "Tree", {"max_depth": 3, "min_samples_leaf": 2}
            ),
            ScreeningModelSpec(
                "random_forest",
                "Forest",
                {"n_estimators": 20, "max_depth": 4, "min_samples_leaf": 2},
            ),
        ),
    )
    return replace(
        screening_config.SCREENING,
        screening_id="MS-TEST",
        screening_title="Test screening",
        screening_note=Path("model-screening/MS-TEST Trees.md"),
        feature_reference_module=None,
        active_group="tree_test",
        groups={"tree_test": group},
        reference_model_id="feature_champion",
        diagnostic_model_id="random_forest",
        shortlist_size=1,
        run_name="ms_test_v1",
        artifact_dir=Path("artifacts/model-screening"),
        results_registry=Path("model-screening/results.csv"),
        save_artifacts=True,
        save_figures=True,
        sync_screening_note=True,
        sync_registry=True,
        allow_overwrite=True,
    )


def boosting_screening_for_test():
    group = ModelGroupSettings(
        group_id="boosting_test",
        title="Boosting",
        preprocessing_profile="unscaled_dense",
        models=(
            ScreeningModelSpec(
                "hist_gradient_boosting",
                "Histogram boosting",
                {
                    "max_iter": 12,
                    "learning_rate": 0.08,
                    "max_leaf_nodes": 7,
                    "min_samples_leaf": 5,
                },
            ),
            ScreeningModelSpec(
                "gradient_boosting",
                "Gradient boosting",
                {
                    "n_estimators": 12,
                    "learning_rate": 0.08,
                    "max_depth": 2,
                },
            ),
        ),
    )
    return replace(
        screening_for_test(),
        screening_id="MS-BOOST",
        screening_title="Boosting loss test",
        screening_note=Path("model-screening/MS-BOOST Boosting.md"),
        active_group="boosting_test",
        groups={"boosting_test": group},
        diagnostic_model_id="hist_gradient_boosting",
        shortlist_size=2,
        run_name="ms_boost_v1",
    )


class ModelScreeningNotebookTests(unittest.TestCase):
    def test_notebook_has_valid_code_cells(self) -> None:
        path = Path(__file__).resolve().parents[1] / "notebooks/06_model_screening.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cell_ids = {cell.get("id") for cell in notebook["cells"]}
        self.assertIn("screening-save", cell_ids)
        self.assertIn("screening-diagnostics", cell_ids)
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if source.strip():
                compile(source, f"{path.name}:{cell.get('id')}", "exec")


@unittest.skipUnless(
    DEPENDENCIES_AVAILABLE,
    "model screening requires scikit-learn",
)
class ModelScreeningTests(unittest.TestCase):
    def test_boosting_screening_adds_one_compact_loss_figure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs/00_problem.md").write_text(
                "(primary_metric:: accuracy)\n",
                encoding="utf-8",
            )
            settings = settings_for_test()
            screening = boosting_screening_for_test()
            context = screening_tools.prepare_screening_context(
                root,
                sample_frame(),
                FEATURE_GROUPS,
                target="target",
                key="id",
                initial_settings=settings,
                feature_reference_module=None,
            )
            built = screening_tools.build_model_group(context, screening)
            result = screening_tools.run_model_screening(root, built, screening)
            figures = screening_tools.build_screening_figures(result, screening)

            self.assertIn("boosting-log-loss.png", figures)
            self.assertEqual(len(figures["boosting-log-loss.png"].axes), 2)
            saved = screening_tools.save_model_screening(
                root,
                result,
                screening,
                dataset_version="test-data",
                figures=figures,
            )
            note = saved.note_path.read_text(encoding="utf-8")
            self.assertEqual(note.count("boosting-log-loss.png"), 1)

    def test_task_specific_params_support_regression_templates(self) -> None:
        regression = replace(settings_for_test(), task_type="regression")
        estimator = build_screening_estimator(
            "linear_reference",
            regression,
            {
                "classification": {"C": 0.5},
                "regression": {"alpha": 2.5},
            },
        )
        self.assertEqual(type(estimator).__name__, "Ridge")
        self.assertEqual(estimator.alpha, 2.5)

    def test_native_preprocessor_is_cloneable_and_preserves_categories(self) -> None:
        from sklearn.base import clone

        transformer = NativeCategoricalPreprocessor(
            numeric_features=("numeric",),
            categorical_features=("category",),
        )
        cloned = clone(transformer)
        frame = sample_frame().drop(columns=["id", "target"])
        transformed = cloned.fit_transform(frame)
        self.assertEqual(transformed.columns.tolist(), ["numeric", "category"])
        self.assertTrue(transformed["category"].map(lambda value: isinstance(value, str)).all())

    def test_group_run_reuses_folds_and_builds_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs/00_problem.md").write_text(
                "(primary_metric:: accuracy)\n", encoding="utf-8"
            )
            (root / "docs/05_experiments.md").write_text(
                "# Experiments\n\n"
                "<!-- auto:latest-model-screening:start -->\n"
                "pending\n"
                "<!-- auto:latest-model-screening:end -->\n\n"
                "<!-- auto:model-screening-summary:start -->\n"
                "pending\n"
                "<!-- auto:model-screening-summary:end -->\n",
                encoding="utf-8",
            )
            frame = sample_frame()
            settings = settings_for_test()
            screening = screening_for_test()
            context = screening_tools.prepare_screening_context(
                root,
                frame,
                FEATURE_GROUPS,
                target="target",
                key="id",
                initial_settings=settings,
                feature_reference_module=None,
            )
            built = screening_tools.build_model_group(context, screening)
            result = screening_tools.run_model_screening(root, built, screening)

            self.assertEqual(
                list(built.models),
                ["feature_champion", "decision_tree", "random_forest"],
            )
            self.assertEqual(len(result.evaluation.cv_splits), 3)
            self.assertEqual(len(result.shortlist), 1)
            self.assertEqual(
                set(result.prediction_comparison["model"]),
                {"decision_tree", "random_forest"},
            )
            self.assertIn("random_forest", result.feature_importance["model"].tolist())
            self.assertTrue(
                result.paired_deltas["improvement"].map(np.isfinite).all()
            )

            figures = screening_tools.build_screening_figures(result, screening)
            self.assertIn("ranking-primary.png", figures)
            self.assertIn("paired-primary-delta.png", figures)
            self.assertIn("importance-random_forest.png", figures)

            saved = screening_tools.save_model_screening(
                root,
                result,
                screening,
                dataset_version="test-data",
                figures=figures,
            )
            self.assertTrue(saved.leaderboard_path.is_file())
            self.assertTrue(saved.note_path.is_file())
            self.assertTrue(saved.registry_path.is_file())
            self.assertTrue((root / "model-screening/_index.md").is_file())
            stage = (root / "docs/05_experiments.md").read_text(encoding="utf-8")
            self.assertIn("[[model-screening/MS-TEST Trees.md\\|MS-TEST]]", stage)
            self.assertIn(str(result.leaderboard.iloc[0]["model"]), stage)
            self.assertIn("Shortlist", stage)
            self.assertTrue(saved.figure_paths)
            self.assertTrue(all((root / path).is_file() for path in saved.figure_paths.values()))
            note = saved.note_path.read_text(encoding="utf-8")
            self.assertIn("## Результат группы", note)
            self.assertIn("feature_champion", note)
            self.assertIn("assets/model-screening/MS-TEST", note)

            rerun = screening_tools.save_model_screening(
                root,
                result,
                screening,
                dataset_version="test-data",
                figures=figures,
            )
            self.assertEqual(rerun.run_dir, saved.run_dir)
            with self.assertRaisesRegex(FileExistsError, "another screening contract"):
                screening_tools.save_model_screening(
                    root,
                    result,
                    screening,
                    dataset_version="changed-data",
                    figures=figures,
                )

    def test_invalid_diagnostic_model_is_rejected(self) -> None:
        settings = replace(
            screening_for_test(), diagnostic_model_id="missing_model"
        )
        with self.assertRaisesRegex(ValueError, "diagnostic_model_id"):
            screening_tools.validate_screening_settings(settings)


if __name__ == "__main__":
    unittest.main()
