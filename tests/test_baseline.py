from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import ml_project.baseline_config as baseline_config
import ml_project.modeling as modeling_tools
from ml_project.baseline import (
    CVEvaluation,
    FeaturePlan,
    ScoringPlan,
    build_baseline_experiment_report,
    build_key_results_block,
    build_metric_figures,
    build_secondary_metrics_block,
    build_cv_splitter,
    build_dummy_estimator,
    build_model_pipeline,
    build_simple_estimator,
    build_tabular_preprocessor,
    cv_protocol_description,
    evaluate_models_cv,
    prepare_training_data,
    resolve_feature_plan,
    resolve_scoring_plan,
    save_baseline_run,
    sync_baseline_docs,
    sync_baseline_experiment_note,
    validate_modeling_settings,
)

try:
    import sklearn  # noqa: F401
except ImportError:  # pragma: no cover - local dependency state
    SKLEARN_AVAILABLE = False
else:
    SKLEARN_AVAILABLE = True

try:
    import matplotlib  # noqa: F401
except ImportError:  # pragma: no cover - local dependency state
    MATPLOTLIB_AVAILABLE = False
else:
    MATPLOTLIB_AVAILABLE = True


def classification_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": range(12),
            "numeric": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0] * 2,
            "category": ["a", "a", "b", "b", "c", "c"] * 2,
            "text": [f"free text {index}" for index in range(12)],
            "target": [0, 1] * 6,
        }
    )


FEATURE_GROUPS = {
    "numeric": ["numeric"],
    "count": [],
    "categorical": ["category"],
    "ordinal": [],
    "text": ["text"],
    "datetime": [],
    "identifier": ["id"],
    "ignored": [],
}


def configured_settings(**changes: object):
    settings = baseline_config.BASELINE
    defaults = {
        "task_type": "binary_classification",
        "model_feature_groups": (
            "numeric",
            "count",
            "categorical",
            "ordinal",
        ),
        "include_features": (),
        "exclude_features": (),
        "primary_scorer": None,
        "secondary_scorers": {},
        "cv_strategy": "stratified_kfold",
        "n_splits": 3,
        "group_column": None,
        "time_column": None,
        "n_jobs": 1,
        "model_name": "auto",
        "model_params": {},
        "experiment_id": "EXP-001",
        "experiment_title": "Baseline",
        "experiment_note": Path("experiments/EXP-001 Baseline.md"),
        "run_name": "test_baseline",
        "artifact_dir": Path("artifacts/baseline"),
        "save_artifacts": False,
        "save_metric_figures": False,
        "metric_figure_dpi": 100,
        "save_final_model": False,
        "sync_docs": False,
        "sync_experiment_note": False,
        "allow_overwrite": False,
    }
    defaults.update(changes)
    settings = replace(settings, **defaults)
    validate_modeling_settings(settings)
    return settings


def problem_doc(root: Path, metric: str) -> None:
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "00_problem.md").write_text(
        f"# Problem\n\n- Metric: (primary_metric:: {metric})\n",
        encoding="utf-8",
    )


class TrackedMetricFigureTests(unittest.TestCase):
    class FakeFigure:
        def savefig(self, path: Path, **_: object) -> None:
            path.write_bytes(b"png")

    def test_legacy_baseline_settings_names_remain_compatible(self) -> None:
        self.assertIs(
            modeling_tools.BaselineSettings,
            modeling_tools.ModelingSettings,
        )
        self.assertIs(
            modeling_tools.validate_baseline_settings,
            modeling_tools.validate_modeling_settings,
        )

    def test_modeling_package_prepares_data_through_public_api(self) -> None:
        settings = configured_settings()
        source = classification_frame()
        plan = modeling_tools.resolve_feature_plan(
            source,
            FEATURE_GROUPS,
            target="target",
            key="id",
            settings=settings,
        )
        prepared = modeling_tools.prepare_training_data(
            source,
            target="target",
            plan=plan,
            settings=settings,
        )
        self.assertEqual(prepared.X.shape, (12, 2))
        self.assertEqual(tuple(prepared.X.columns), ("numeric", "category"))
        self.assertIs(modeling_tools.CVEvaluation, CVEvaluation)

    def test_metric_figure_is_tracked_and_embedded_by_experiment_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = configured_settings(
                save_metric_figures=True,
                allow_overwrite=True,
            )
            plan = FeaturePlan(
                numeric=("numeric",),
                categorical=(),
                excluded=(),
                group_by_feature={"numeric": "numeric"},
                exclusion_reason={},
            )
            scoring = ScoringPlan(
                contract_metric="accuracy",
                scorers={"primary": "accuracy"},
                labels={"primary": "Accuracy"},
                directions={"primary": "maximize"},
                negated={"primary": False},
            )
            evaluation = CVEvaluation(
                fold_scores=pd.DataFrame(
                    [
                        {
                            "model": "simple_model",
                            "split": "validation",
                            "fold": 1,
                            "metric_key": "primary",
                            "metric": "Accuracy",
                            "direction": "maximize",
                            "value": 0.8,
                            "fit_seconds": 0.1,
                            "score_seconds": 0.01,
                        }
                    ]
                ),
                summary=pd.DataFrame(
                    [
                        {
                            "model": "simple_model",
                            "split": "validation",
                            "metric_key": "primary",
                            "metric": "Accuracy",
                            "direction": "maximize",
                            "mean": 0.8,
                            "std": 0.0,
                            "min": 0.8,
                            "max": 0.8,
                            "folds": 1,
                            "fit_seconds_mean": 0.1,
                            "score_seconds_mean": 0.01,
                        }
                    ]
                ),
                raw_results={},
            )

            saved = save_baseline_run(
                root,
                settings,
                evaluation,
                plan,
                scoring,
                dataset_version="abc123",
                cv_description="one test fold",
                metric_figures={"primary": self.FakeFigure()},
            )

            expected = (
                root
                / "assets/experiments/EXP-001/metric-primary-accuracy.png"
            )
            self.assertEqual((saved.metric_figure_paths or {})["primary"], expected)
            self.assertTrue(expected.exists())
            self.assertFalse((saved.run_dir / expected.name).exists())

            report = build_baseline_experiment_report(
                root,
                settings,
                evaluation,
                scoring,
                saved,
                dataset_version="abc123",
                cv_description="one test fold",
            )
            self.assertIn(
                "![[assets/experiments/EXP-001/metric-primary-accuracy.png]]",
                report,
            )
            metadata = saved.metadata_path.read_text(encoding="utf-8")
            self.assertIn(
                '"primary": "assets/experiments/EXP-001/'
                'metric-primary-accuracy.png"',
                metadata,
            )
            self.assertIn('"environment": {', metadata)
            self.assertIn('"scikit-learn":', metadata)

    def test_dashboard_and_secondary_metrics_blocks_are_generated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "experiments").mkdir()
            (root / "README.md").write_text(
                "Manual dashboard\n"
                "<!-- auto:key-results:start -->\nold\n"
                "<!-- auto:key-results:end -->\n",
                encoding="utf-8",
            )
            (root / "docs/03_validation.md").write_text(
                "Manual validation\n"
                "<!-- auto:secondary-metrics:start -->\nold\n"
                "<!-- auto:secondary-metrics:end -->\n"
                "<!-- auto:baseline-results:start -->\nold\n"
                "<!-- auto:baseline-results:end -->\n",
                encoding="utf-8",
            )
            (root / "docs/05_experiments.md").write_text(
                "Manual experiments\n"
                "<!-- auto:current-baseline:start -->\nold\n"
                "<!-- auto:current-baseline:end -->\n"
                "<!-- auto:best-measured-result:start -->\nold\n"
                "<!-- auto:best-measured-result:end -->\n",
                encoding="utf-8",
            )
            (root / "experiments/_index.md").write_text(
                "<!-- auto:experiment-registry:start -->\nold\n"
                "<!-- auto:experiment-registry:end -->\n",
                encoding="utf-8",
            )
            evaluation = CVEvaluation(
                fold_scores=pd.DataFrame(),
                summary=pd.DataFrame(
                    [
                        {
                            "model": "simple_model",
                            "split": "validation",
                            "metric_key": "primary",
                            "metric": "Accuracy",
                            "direction": "maximize",
                            "mean": 0.8,
                            "std": 0.02,
                            "min": 0.78,
                            "max": 0.82,
                            "folds": 5,
                        }
                    ]
                ),
                raw_results={},
            )
            scoring = ScoringPlan(
                contract_metric="accuracy",
                scorers={"primary": "accuracy", "secondary_1": "f1"},
                labels={"primary": "accuracy", "secondary_1": "F1"},
                directions={"primary": "maximize", "secondary_1": "maximize"},
                negated={"primary": False, "secondary_1": False},
            )
            sync_baseline_docs(
                root,
                evaluation,
                scoring,
                dataset_version="abc123",
                cv_description="5-fold CV",
                experiment_note=Path("experiments/EXP-001 Baseline.md"),
            )

            readme = (root / "README.md").read_text(encoding="utf-8")
            validation = (root / "docs/03_validation.md").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "[[experiments/EXP-001 Baseline.md\\|EXP-001 Baseline]]",
                readme,
            )
            self.assertIn("0.8000", readme)
            self.assertIn("F1", validation)
            self.assertIn("f1", validation)
            self.assertNotIn("\nold\n", readme)

    def test_key_results_include_controlled_experiment_links(self) -> None:
        experiments = pd.DataFrame(
            [
                {
                    "experiment_id": "EXP-002",
                    "title": "New feature",
                    "note": "experiments/EXP-002 New feature.md",
                    "primary_metric": "accuracy",
                    "candidate_score": 0.81,
                    "improvement": 0.01,
                    "decision": "adopt",
                }
            ]
        )
        block = build_key_results_block(
            baseline_metric="accuracy",
            baseline_score=0.8,
            baseline_note=Path("experiments/EXP-001 Baseline.md"),
            experiments=experiments,
        )
        self.assertIn("EXP-001 Baseline.md", block)
        self.assertIn("EXP-002 New feature.md", block)
        self.assertIn("+0.0100", block)
        self.assertIn("adopt", block)

    def test_secondary_metrics_empty_state_points_to_config(self) -> None:
        scoring = ScoringPlan(
            contract_metric="accuracy",
            scorers={"primary": "accuracy"},
            labels={"primary": "accuracy"},
            directions={"primary": "maximize"},
            negated={"primary": False},
        )
        self.assertIn(
            "BASELINE", build_secondary_metrics_block(scoring)
        )
        self.assertIn(
            "secondary_scorers", build_secondary_metrics_block(scoring)
        )

    def test_modeling_report_audit_finds_missing_artifact_and_decision_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experiments").mkdir()
            note = root / "experiments/EXP-002 Test.md"
            note.write_text(
                "---\n"
                "decision: iterate\n"
                "---\n\n"
                "<!-- auto:experiment-report:start -->\n"
                "[[artifacts/experiments/test/metadata.json]]\n"
                "<!-- auto:experiment-report:end -->\n",
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "experiment_id": "EXP-002",
                        "note": "experiments/EXP-002 Test.md",
                        "decision": "pending",
                    }
                ]
            ).to_csv(root / "experiments/results.csv", index=False)

            issues = modeling_tools.audit_modeling_report(root)

            self.assertTrue(any("missing artifacts/" in issue for issue in issues))
            self.assertTrue(any("card decision 'iterate'" in issue for issue in issues))


@unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is not installed")
class BaselinePipelineTests(unittest.TestCase):
    def build_classification_evaluation(self, root: Path, **setting_changes: object):
        frame = classification_frame()
        settings = configured_settings(**setting_changes)
        plan = resolve_feature_plan(
            frame,
            FEATURE_GROUPS,
            target="target",
            key="id",
            settings=settings,
        )
        data = prepare_training_data(
            frame,
            target="target",
            plan=plan,
            settings=settings,
        )
        scoring = resolve_scoring_plan(root, settings)
        cv, strategy = build_cv_splitter(settings, data.y)
        preprocessor = build_tabular_preprocessor(settings, plan)
        models = {
            "dummy": build_model_pipeline(
                preprocessor, build_dummy_estimator(settings)
            ),
            "simple_model": build_model_pipeline(
                preprocessor, build_simple_estimator(settings)
            ),
        }
        evaluation = evaluate_models_cv(
            models,
            data,
            cv=cv,
            scoring=scoring,
            settings=settings,
        )
        return settings, plan, data, scoring, strategy, models, evaluation

    def test_feature_plan_is_explicit_and_does_not_mutate_raw_frame(self) -> None:
        frame = classification_frame()
        original = frame.copy(deep=True)
        settings = configured_settings(exclude_features=("category",))

        plan = resolve_feature_plan(
            frame,
            FEATURE_GROUPS,
            target="target",
            key="id",
            settings=settings,
        )

        self.assertEqual(plan.numeric, ("numeric",))
        self.assertEqual(plan.categorical, ())
        self.assertIn("category", plan.excluded)
        self.assertIn("text", plan.excluded)
        self.assertIn("id", plan.excluded)
        pd.testing.assert_frame_equal(frame, original)

    def test_classification_dummy_and_simple_model_share_cv_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            problem_doc(root, "accuracy")
            _, _, _, scoring, _, _, evaluation = (
                self.build_classification_evaluation(root)
            )

            primary = evaluation.primary_summary()
            self.assertEqual(set(primary["model"]), {"dummy", "simple_model"})
            self.assertEqual(scoring.contract_metric, "accuracy")
            self.assertEqual(set(primary["direction"]), {"maximize"})
            self.assertTrue(np.isfinite(primary["mean"]).all())
            self.assertEqual(set(primary["folds"]), {3})

    def test_regression_rmse_is_reported_as_positive_minimized_loss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            problem_doc(root, "RMSE")
            frame = pd.DataFrame(
                {
                    "id": range(15),
                    "numeric": np.arange(15, dtype=float),
                    "target": np.arange(15, dtype=float) * 2.0 + 1.0,
                }
            )
            groups = {
                "numeric": ["numeric"],
                "identifier": ["id"],
            }
            settings = configured_settings(
                task_type="regression",
                cv_strategy="kfold",
                model_name="ridge",
            )
            plan = resolve_feature_plan(
                frame,
                groups,
                target="target",
                key="id",
                settings=settings,
            )
            data = prepare_training_data(
                frame,
                target="target",
                plan=plan,
                settings=settings,
            )
            scoring = resolve_scoring_plan(root, settings)
            cv, _ = build_cv_splitter(settings, data.y)
            preprocessor = build_tabular_preprocessor(settings, plan)
            models = {
                "simple_model": build_model_pipeline(
                    preprocessor, build_simple_estimator(settings)
                )
            }
            evaluation = evaluate_models_cv(
                models,
                data,
                cv=cv,
                scoring=scoring,
                settings=settings,
            )

            primary = evaluation.primary_summary().iloc[0]
            self.assertEqual(scoring.scorers["primary"], "neg_root_mean_squared_error")
            self.assertEqual(primary["direction"], "minimize")
            self.assertGreaterEqual(float(primary["mean"]), 0.0)

    def test_save_refuses_to_overwrite_existing_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            problem_doc(root, "accuracy")
            settings, plan, data, scoring, strategy, models, evaluation = (
                self.build_classification_evaluation(root)
            )
            description = cv_protocol_description(settings, strategy)

            saved = save_baseline_run(
                root,
                settings,
                evaluation,
                plan,
                scoring,
                dataset_version="abc123",
                cv_description=description,
                final_pipeline=models["simple_model"],
                data=data,
            )
            self.assertTrue(saved.summary_path.exists())
            self.assertTrue(saved.metadata_path.exists())
            with self.assertRaises(FileExistsError):
                save_baseline_run(
                    root,
                    settings,
                    evaluation,
                    plan,
                    scoring,
                    dataset_version="abc123",
                    cv_description=description,
                )

    def test_safe_overwrite_refreshes_generated_files_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            problem_doc(root, "accuracy")
            settings, plan, _, scoring, strategy, _, evaluation = (
                self.build_classification_evaluation(root, allow_overwrite=True)
            )
            description = cv_protocol_description(settings, strategy)
            first = save_baseline_run(
                root,
                settings,
                evaluation,
                plan,
                scoring,
                dataset_version="abc123",
                cv_description=description,
            )
            manual_path = first.run_dir / "manual-notes.txt"
            stale_figure = first.run_dir / "metric-old.png"
            manual_path.write_text("keep me", encoding="utf-8")
            stale_figure.write_bytes(b"stale")

            second = save_baseline_run(
                root,
                settings,
                evaluation,
                plan,
                scoring,
                dataset_version="def456",
                cv_description=description,
            )

            self.assertEqual(manual_path.read_text(encoding="utf-8"), "keep me")
            self.assertFalse(stale_figure.exists())
            self.assertIn("def456", second.metadata_path.read_text(encoding="utf-8"))

    @unittest.skipUnless(MATPLOTLIB_AVAILABLE, "matplotlib is not installed")
    def test_metric_figures_and_experiment_report_are_saved_and_embedded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            problem_doc(root, "accuracy")
            settings, plan, _, scoring, strategy, _, evaluation = (
                self.build_classification_evaluation(
                    root,
                    secondary_scorers={"F1": "f1"},
                    save_metric_figures=True,
                    allow_overwrite=True,
                    sync_experiment_note=True,
                )
            )
            description = cv_protocol_description(settings, strategy)
            figures = build_metric_figures(evaluation, scoring)
            saved = save_baseline_run(
                root,
                settings,
                evaluation,
                plan,
                scoring,
                dataset_version="abc123",
                cv_description=description,
                metric_figures=figures,
            )
            sync_baseline_experiment_note(
                root,
                settings,
                evaluation,
                scoring,
                saved,
                dataset_version="abc123",
                cv_description=description,
            )

            self.assertEqual(set(saved.metric_figure_paths or {}), {"primary", "secondary_1"})
            self.assertTrue(all(path.exists() for path in (saved.metric_figure_paths or {}).values()))
            self.assertTrue(
                all(
                    path.parent == root / "assets/experiments/EXP-001"
                    for path in (saved.metric_figure_paths or {}).values()
                )
            )
            note_path = root / settings.experiment_note
            note = note_path.read_text(encoding="utf-8")
            self.assertIn("![[assets/experiments/EXP-001/metric-primary-accuracy.png]]", note)
            self.assertIn("## Метрика: F1", note)
            self.assertIn("### Значения по folds", note)
            self.assertIn("[[artifacts/baseline/test_baseline/cv_summary.csv|cv_summary.csv]]", note)

            note_path.write_text(note + "\nРучной вывод: сохранить.\n", encoding="utf-8")
            sync_baseline_experiment_note(
                root,
                settings,
                evaluation,
                scoring,
                saved,
                dataset_version="abc123",
                cv_description=description,
            )
            self.assertIn(
                "Ручной вывод: сохранить.",
                note_path.read_text(encoding="utf-8"),
            )

    def test_docs_sync_changes_only_marked_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            problem_doc(root, "accuracy")
            (root / "experiments").mkdir()
            settings, plan, prepared, scoring, strategy, _, evaluation = (
                self.build_classification_evaluation(root)
            )
            (root / "docs/03_validation.md").write_text(
                "Manual before\n"
                "<!-- auto:validation-protocol:start -->\nold\n"
                "<!-- auto:validation-protocol:end -->\n"
                "<!-- auto:secondary-metrics:start -->\nold\n"
                "<!-- auto:secondary-metrics:end -->\n"
                "<!-- auto:baseline-results:start -->\nold\n"
                "<!-- auto:baseline-results:end -->\n"
                "<!-- auto:reproducibility:start -->\nold\n"
                "<!-- auto:reproducibility:end -->\nManual after\n",
                encoding="utf-8",
            )
            (root / "docs/04_features.md").write_text(
                "Manual feature rationale\n"
                "<!-- auto:model-ready-contract:start -->\nold\n"
                "<!-- auto:model-ready-contract:end -->\n"
                "<!-- auto:feature-registry:start -->\nold\n"
                "<!-- auto:feature-registry:end -->\n"
                "<!-- auto:preprocessing-contract:start -->\nold\n"
                "<!-- auto:preprocessing-contract:end -->\n",
                encoding="utf-8",
            )
            (root / "docs/05_experiments.md").write_text(
                "Manual before\n"
                "<!-- auto:current-baseline:start -->\nold\n"
                "<!-- auto:current-baseline:end -->\n"
                "<!-- auto:best-measured-result:start -->\nold\n"
                "<!-- auto:best-measured-result:end -->\nManual after\n",
                encoding="utf-8",
            )
            (root / "experiments/_index.md").write_text(
                "<!-- auto:experiment-registry:start -->\nold\n"
                "<!-- auto:experiment-registry:end -->\n",
                encoding="utf-8",
            )
            (root / "README.md").write_text(
                "Manual dashboard\n"
                "<!-- auto:key-results:start -->\nold\n"
                "<!-- auto:key-results:end -->\n",
                encoding="utf-8",
            )

            sync_baseline_docs(
                root,
                evaluation,
                scoring,
                dataset_version="abc123",
                cv_description=cv_protocol_description(settings, strategy),
                settings=settings,
                feature_plan=plan,
                prepared_data=prepared,
            )

            validation = (root / "docs/03_validation.md").read_text(
                encoding="utf-8"
            )
            experiments = (root / "docs/05_experiments.md").read_text(
                encoding="utf-8"
            )
            features = (root / "docs/04_features.md").read_text(encoding="utf-8")
            self.assertIn("simple_model", validation)
            self.assertIn("accuracy", validation)
            self.assertIn("stratified_kfold", validation)
            self.assertIn("Python", validation)
            self.assertIn("Manual before", validation)
            self.assertIn("Manual after", validation)
            self.assertIn("numeric", features)
            self.assertIn("standard", features)
            self.assertIn("Manual feature rationale", features)
            self.assertIn("Baseline / simple_model", experiments)
            self.assertNotIn("\nold\n", experiments)


if __name__ == "__main__":
    unittest.main()
