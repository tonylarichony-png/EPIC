from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml_project.experiment_scaffold import (
    find_adopted_champion_module,
    find_next_experiment_id,
    main,
    parse_guardrails,
    scaffold_experiment,
    selected_experiment_spec,
    slug_from_title,
)
from ml_project.experiment_workbench import create_experiment_workbench


class ExperimentScaffoldTests(unittest.TestCase):
    def test_workbench_is_source_only_and_bound_to_module(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = create_experiment_workbench(
                root,
                "EXP-003",
                "Family size",
                slug="family_size",
                module_name="ml_project.experiments.exp_003_family_size",
                parent_experiment_module=(
                    "ml_project.experiments.exp_002_age_imputation"
                ),
            )

            notebook = json.loads(path.read_text(encoding="utf-8"))
            code_cells = [
                cell for cell in notebook["cells"] if cell["cell_type"] == "code"
            ]
            source = "".join(
                "".join(cell["source"]) for cell in notebook["cells"]
            )
            code_source = "".join(
                "".join(cell["source"]) for cell in code_cells
            )
            self.assertEqual(path.name, "EXP-003_family_size.ipynb")
            self.assertTrue(all(cell["outputs"] == [] for cell in code_cells))
            self.assertTrue(
                all(cell["execution_count"] is None for cell in code_cells)
            )
            self.assertIn(
                "ml_project.experiments.exp_003_family_size",
                source,
            )
            self.assertIn(
                "ml_project.experiments.exp_002_age_imputation",
                source,
            )
            self.assertIn("build_reference_pipeline", source)
            self.assertIn("initial_settings = replace(", source)
            self.assertIn("reference_settings = parent_data.settings", source)
            self.assertIn("candidate_settings = reference_settings", source)
            self.assertIn("settings=candidate_settings", source)
            self.assertIn("def prepare_candidate_data(", source)
            self.assertIn("frame = train.copy(deep=True)", source)
            self.assertIn("groups = copy.deepcopy(feature_groups)", source)
            self.assertIn("feature_groups=groups", source)
            self.assertNotIn("def draft_transform(", code_source)
            self.assertNotIn("draft_feature_groups", code_source)
            self.assertNotIn("result = frame.copy(deep=True)", code_source)
            self.assertIn("RUN_MODULE_SMOKE = False", source)
            self.assertNotIn("sync_experiment_docs(", source)

    def test_selected_spec_does_not_import_unfinished_module(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "src/ml_project/experiments"
            package.mkdir(parents=True)
            selector = root / "src/ml_project/experiment_config.py"
            selector.write_text(
                'EXPERIMENT_MODULE = "ml_project.experiments.exp_003_draft"\n',
                encoding="utf-8",
            )
            (package / "exp_003_draft.py").write_text(
                "from package_that_does_not_exist import value\n\n"
                "EXPERIMENT = ExperimentSettings(\n"
                '    experiment_id="EXP-003",\n'
                '    experiment_title="Draft feature",\n'
                ")\n",
                encoding="utf-8",
            )

            self.assertEqual(
                selected_experiment_spec(root),
                (
                    "EXP-003",
                    "Draft feature",
                    "draft",
                    "ml_project.experiments.exp_003_draft",
                    None,
                ),
            )
            result = main(
                ["--project-root", str(root), "--workbench-only"]
            )
            self.assertEqual(result, 0)
            self.assertTrue(
                (root / "notebooks/workbench/EXP-003_draft.ipynb").is_file()
            )
            original = (
                root / "notebooks/workbench/EXP-003_draft.ipynb"
            ).read_bytes()
            self.assertEqual(
                main(["--project-root", str(root), "--workbench-only"]),
                0,
            )
            self.assertEqual(
                (
                    root / "notebooks/workbench/EXP-003_draft.ipynb"
                ).read_bytes(),
                original,
            )

    def test_cli_creates_module_and_workbench_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "src/ml_project/experiments"
            package.mkdir(parents=True)
            selector = root / "src/ml_project/experiment_config.py"
            selector.write_text(
                'EXPERIMENT_MODULE = "ml_project.experiments.old"\n',
                encoding="utf-8",
            )

            result = main(
                [
                    "EXP-003",
                    "--title",
                    "Family size",
                    "--slug",
                    "family_size",
                    "--min-improvement",
                    "0.005",
                    "--project-root",
                    str(root),
                ]
            )

            self.assertEqual(result, 0)
            self.assertTrue(
                (package / "exp_003_family_size.py").is_file()
            )
            self.assertTrue(
                (
                    root
                    / "notebooks/workbench/EXP-003_family_size.ipynb"
                ).is_file()
            )
            module_source = (
                package / "exp_003_family_size.py"
            ).read_text(encoding="utf-8")
            notebook = json.loads(
                (
                    root
                    / "notebooks/workbench/EXP-003_family_size.ipynb"
                ).read_text(encoding="utf-8")
            )
            notebook_source = "\n".join(
                "".join(cell["source"])
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )
            module_prepare = next(
                node
                for node in ast.parse(module_source).body
                if isinstance(node, ast.FunctionDef)
                and node.name == "prepare_candidate_data"
            )
            notebook_prepare = next(
                node
                for node in ast.parse(notebook_source).body
                if isinstance(node, ast.FunctionDef)
                and node.name == "prepare_candidate_data"
            )
            self.assertEqual(
                ast.dump(module_prepare, include_attributes=False),
                ast.dump(notebook_prepare, include_attributes=False),
            )

    def test_latest_adopted_champion_is_embedded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "src/ml_project/experiments"
            package.mkdir(parents=True)
            selector = root / "src/ml_project/experiment_config.py"
            selector.write_text(
                'EXPERIMENT_MODULE = "ml_project.experiments.old"\n',
                encoding="utf-8",
            )
            parent_module = package / "exp_002_parent.py"
            parent_module.write_text(
                "EXPERIMENT = ExperimentSettings(\n"
                '    experiment_id="EXP-002",\n'
                '    experiment_title="Parent",\n'
                ")\n",
                encoding="utf-8",
            )
            cards = root / "experiments"
            cards.mkdir()
            (cards / "EXP-002 Parent.md").write_text(
                "---\n"
                "id: EXP-002\n"
                "decision: adopt\n"
                "---\n\n"
                "# EXP-002 — Parent\n",
                encoding="utf-8",
            )

            self.assertEqual(
                find_adopted_champion_module(root),
                "ml_project.experiments.exp_002_parent",
            )
            result = main(
                [
                    "EXP-003",
                    "--title",
                    "Child",
                    "--slug",
                    "child",
                    "--min-improvement",
                    "0.01",
                    "--project-root",
                    str(root),
                ]
            )

            self.assertEqual(result, 0)
            source = (package / "exp_003_child.py").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "parent_experiment_module="
                "'ml_project.experiments.exp_002_parent'",
                source,
            )
            self.assertIn("reference_model='champion_reference'", source)

    def test_interactive_preview_can_be_cancelled_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answers = ["Тест признака", "", "", "", "n"]
            with (
                patch("ml_project.experiment_scaffold.sys.stdin") as stdin,
                patch("builtins.input", side_effect=answers),
            ):
                stdin.isatty.return_value = True
                result = main(["--project-root", str(root)])

            self.assertEqual(result, 0)
            self.assertFalse((root / "src").exists())

    def test_next_id_slug_and_guardrail_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "src/ml_project/experiments"
            package.mkdir(parents=True)
            (package / "exp_002_existing.py").write_text(
                'experiment_id = "EXP-002"\n',
                encoding="utf-8",
            )

            self.assertEqual(find_next_experiment_id(root), "EXP-003")
            self.assertEqual(
                slug_from_title("Возраст по титулу"),
                "vozrast_po_titulu",
            )
            self.assertEqual(
                parse_guardrails("Recall=-0.005, F1=-0.002"),
                {"Recall": -0.005, "F1": -0.002},
            )

    def test_scaffold_creates_immutable_module_and_selects_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "src/ml_project/experiments"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            selector = root / "src/ml_project/experiment_config.py"
            selector.write_text(
                'EXPERIMENT_MODULE = "ml_project.experiments.old"\n',
                encoding="utf-8",
            )

            path, module_name = scaffold_experiment(
                root,
                "EXP-003",
                "Title feature",
                slug="title_feature",
                primary_improvement_min=0.01,
                metric_guardrails={"Recall": -0.005},
            )

            self.assertEqual(path.name, "exp_003_title_feature.py")
            self.assertEqual(
                module_name,
                "ml_project.experiments.exp_003_title_feature",
            )
            source = path.read_text(encoding="utf-8")
            self.assertIn("experiment_id='EXP-003'", source)
            self.assertIn("primary_improvement_min=0.01", source)
            self.assertIn("metric_guardrails={'Recall': -0.005}", source)
            self.assertIn("prepare_candidate_data", source)
            self.assertIn("build_candidate_models", source)
            self.assertIn("ModelingSettings", source)
            self.assertIn("reference_settings: ModelingSettings", source)
            self.assertIn("candidate_settings = reference_settings", source)
            self.assertNotIn("baseline_settings: ModelingSettings", source)
            self.assertIn(
                module_name,
                selector.read_text(encoding="utf-8"),
            )

            with self.assertRaises(FileExistsError):
                scaffold_experiment(
                    root,
                    "EXP-003",
                    "Duplicate",
                    slug="another_slug",
                )


if __name__ == "__main__":
    unittest.main()
