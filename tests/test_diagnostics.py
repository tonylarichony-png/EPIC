from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import ml_project.baseline_config as baseline_config
import ml_project.modeling as modeling_tools
from ml_project.experiment import build_experiment_report

try:
    import matplotlib  # noqa: F401
    import sklearn  # noqa: F401
except ImportError:  # pragma: no cover - local dependency state
    DIAGNOSTIC_DEPENDENCIES_AVAILABLE = False
else:
    DIAGNOSTIC_DEPENDENCIES_AVAILABLE = True


class DiagnosticsNotebookTests(unittest.TestCase):
    def test_experiment_and_diagnostics_notebooks_have_valid_code_cells(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for relative in (
            "notebooks/04_experiment.ipynb",
            "notebooks/05_diagnostics.ipynb",
        ):
            path = root / relative
            notebook = json.loads(path.read_text(encoding="utf-8"))
            cell_ids = {cell.get("id") for cell in notebook["cells"]}
            if path.name == "04_experiment.ipynb":
                self.assertIn("exp-diagnostics", cell_ids)
            for cell in notebook["cells"]:
                if cell.get("cell_type") != "code":
                    continue
                source = "".join(cell.get("source", []))
                if source.strip():
                    compile(source, f"{path.name}:{cell.get('id')}", "exec")


@unittest.skipUnless(
    DIAGNOSTIC_DEPENDENCIES_AVAILABLE,
    "diagnostics require scikit-learn and matplotlib",
)
class ExperimentDiagnosticsTests(unittest.TestCase):
    def test_diagnostics_reuse_fitted_folds_and_trace_added_feature(self) -> None:
        rows = 60
        x1 = np.linspace(-2.0, 2.0, rows)
        x2 = np.sin(np.arange(rows) / 4.0)
        frame = pd.DataFrame(
            {
                "x1": x1,
                "x2": x2,
                "category": np.where(np.arange(rows) % 3 == 0, "a", "b"),
            }
        )
        target = pd.Series(
            (x1 + 0.35 * x2 > 0).astype(int),
            name="target",
        )
        settings = replace(
            baseline_config.BASELINE,
            task_type="binary_classification",
            primary_scorer="accuracy",
            secondary_scorers={"F1": "f1"},
            n_splits=3,
            n_jobs=1,
            model_name="logistic_regression",
            model_params={"max_iter": 1000, "random_state": 42},
            experiment_id="EXP-TEST",
            experiment_title="Diagnostics",
            run_name="diagnostics_test",
            artifact_dir=Path("artifacts/experiments"),
            metric_figure_dpi=80,
        )
        reference_plan = modeling_tools.FeaturePlan(
            numeric=("x1",),
            categorical=("category",),
            excluded=("x2",),
            group_by_feature={
                "x1": "numeric",
                "x2": "numeric",
                "category": "categorical",
            },
            exclusion_reason={"x2": "reference"},
        )
        candidate_plan = modeling_tools.FeaturePlan(
            numeric=("x1", "x2"),
            categorical=("category",),
            excluded=(),
            group_by_feature=reference_plan.group_by_feature,
            exclusion_reason={},
        )
        data = modeling_tools.PreparedData(
            X=frame,
            y=target,
            groups=None,
            row_index=frame.index,
        )
        reference = modeling_tools.build_model_pipeline(
            modeling_tools.build_tabular_preprocessor(settings, reference_plan),
            modeling_tools.build_simple_estimator(settings),
        )
        candidate = modeling_tools.build_model_pipeline(
            modeling_tools.build_tabular_preprocessor(settings, candidate_plan),
            modeling_tools.build_simple_estimator(settings),
        )
        scoring = modeling_tools.ScoringPlan(
            contract_metric="accuracy",
            scorers={"primary": "accuracy", "secondary_1": "f1"},
            labels={"primary": "accuracy", "secondary_1": "F1"},
            directions={"primary": "maximize", "secondary_1": "maximize"},
            negated={"primary": False, "secondary_1": False},
        )
        cv, _ = modeling_tools.build_cv_splitter(settings, target)
        evaluation = modeling_tools.evaluate_models_cv(
            {"reference": reference, "candidate": candidate},
            data,
            cv=cv,
            scoring=scoring,
            settings=settings,
        )

        self.assertEqual(len(evaluation.cv_splits), 3)
        self.assertEqual(len(evaluation.raw_results["candidate"]["estimator"]), 3)
        diagnostics = modeling_tools.diagnose_experiment(
            evaluation,
            data,
            scoring,
            settings,
            reference_plan,
            candidate_plan,
            reference_model="reference",
            candidate_model="candidate",
            permutation_repeats=2,
        )

        self.assertEqual(diagnostics.focus_features, ("x2",))
        self.assertEqual(len(diagnostics.oof_predictions), rows)
        self.assertEqual(
            int(diagnostics.prediction_changes["rows"].sum()),
            rows,
        )
        self.assertIn("x2", diagnostics.permutation_importance["feature"].tolist())
        self.assertTrue(
            diagnostics.transformed_features["source_feature"].eq("x2").any()
        )
        self.assertFalse(diagnostics.native_importance.empty)
        self.assertFalse(diagnostics.threshold_metrics.empty)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = modeling_tools.save_experiment_diagnostics(
                root,
                diagnostics,
                settings,
            )
            self.assertTrue(saved.summary_path.is_file())
            self.assertTrue(saved.table_paths["oof_predictions"].is_file())
            self.assertTrue(saved.figure_paths["prediction_changes"].is_file())
            run_dir = root / "artifacts/experiments/diagnostics_test"
            run_dir.mkdir(parents=True, exist_ok=True)
            saved_run = modeling_tools.SavedBaselineRun(
                run_dir=run_dir,
                fold_scores_path=run_dir / "cv_fold_scores.csv",
                summary_path=run_dir / "cv_summary.csv",
                metadata_path=run_dir / "metadata.json",
                metric_figure_paths={},
            )
            experiment_settings = modeling_tools.ExperimentSettings(
                experiment_id="EXP-TEST",
                experiment_title="Diagnostics",
                experiment_note=Path("experiments/EXP-TEST Diagnostics.md"),
                hypothesis="New feature helps",
                change_description="Add x2",
                success_criterion="accuracy improves",
                primary_improvement_min=0.0,
                metric_guardrails={},
                reference_model="reference",
                primary_candidate="candidate",
                experiment_parameters={},
                decision="pending",
                run_name="diagnostics_test",
                artifact_dir=Path("artifacts/experiments"),
                results_registry=Path("experiments/results.csv"),
                save_artifacts=True,
                save_metric_figures=True,
                metric_figure_dpi=80,
                save_final_model=False,
                sync_experiment_note=True,
                sync_docs=True,
                allow_overwrite=True,
            )
            report = build_experiment_report(
                root,
                experiment_settings,
                evaluation,
                scoring,
                saved_run,
                dataset_version="test-data",
                cv_description="3-fold",
                diagnostics=diagnostics,
                saved_diagnostics=saved,
            )
            self.assertIn("## Диагностика candidate", report)
            self.assertIn("05_diagnostics.ipynb", report)


if __name__ == "__main__":
    unittest.main()
