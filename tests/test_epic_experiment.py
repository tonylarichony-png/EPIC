"""Tests for the notebook-first EPIC experiment contract."""

import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.epic_experiment import ExperimentSpec, save_run_record
from ml_project.epic_experiment_scaffold import (
    Champion,
    build_experiment_notebook,
    create_experiment,
    find_champion,
    find_next_experiment_id,
)


class TestEpicExperimentScaffold(unittest.TestCase):
    def test_empty_registry_starts_with_exp_001(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(find_next_experiment_id(directory), "EXP-001")

    def test_notebook_has_readable_protocol_and_shared_data_import(self):
        notebook = build_experiment_notebook(
            "EXP-006",
            "Контекст 101 bp",
            "Широкий контекст улучшит AP",
            "Только ширина окна",
            "Delta AP > 0.01",
            champion=Champion(
                "EXP-005", "Accepted model", "run_001", "notebook.ipynb"
            ),
            implementation="module",
            module_name="ml_project.epic_experiments.exp_006_context",
        )
        source = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"]
        )
        self.assertIn("ensure_prepared_split", source)
        self.assertIn("Единственное основное изменение", source)
        self.assertIn("INTERPRETATION", source)
        self.assertIn("average_precision", source)
        self.assertIn("epic_spearman", source)
        self.assertIn("EXP-005", source)
        self.assertIn("RUN_FEATURE_PILOT = False", source)
        self.assertIn("RUN_TRAIN_COUNT = False", source)
        self.assertIn("RUN_CV = False", source)
        self.assertIn("RUN_FULL_FIT = False", source)
        self.assertIn("RUN_VALIDATION = False", source)
        self.assertNotIn("read_sparse_signal", source)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"<{cell['id']}>", "exec")

    def test_create_pair_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experiments").mkdir()
            (root / "experiments/EXP-005.md").write_text("old", encoding="utf-8")
            self.assertEqual(find_next_experiment_id(root), "EXP-006")
            notebook, card, module = create_experiment(
                root,
                experiment_id="EXP-006",
                title="Проверка окна",
                hypothesis="Контекст полезен",
                changed_variable="Ширина окна",
                success_criterion="AP выше reference",
                champion=Champion(
                    "EXP-005", "Accepted model", "run_001", "old.ipynb"
                ),
            )
            parsed = json.loads(notebook.read_text(encoding="utf-8"))
            self.assertEqual(parsed["metadata"]["epic"]["experiment_id"], "EXP-006")
            card_text = card.read_text(encoding="utf-8")
            self.assertIn("decision: pending", card_text)
            self.assertIn("Average Precision", card_text)
            self.assertIn("EPIC Spearman", card_text)
            self.assertIsNotNone(module)
            self.assertTrue(module.is_file())
            self.assertIn("FEATURE_READY = False", module.read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                create_experiment(
                    root,
                    experiment_id="EXP-006",
                    title="Проверка окна",
                    hypothesis="Контекст полезен",
                    changed_variable="Ширина окна",
                    success_criterion="AP выше reference",
                    champion=Champion(
                        "EXP-005", "Accepted model", "run_001", "old.ipynb"
                    ),
                )

    def test_latest_saved_adopted_experiment_is_champion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experiments").mkdir()
            for number, decision in ((7, "adopt"), (8, "reject"), (9, "adopt")):
                experiment_id = f"EXP-{number:03d}"
                (root / "experiments" / f"{experiment_id}.md").write_text(
                    f"---\nid: {experiment_id}\ndecision: {decision}\n"
                    "run_name: run_001\n---\n",
                    encoding="utf-8",
                )
                run_dir = root / "artifacts/experiments" / experiment_id / "run_001"
                run_dir.mkdir(parents=True)
                (run_dir / "metadata.json").write_text(
                    json.dumps({
                        "experiment": {"experiment_id": experiment_id, "title": experiment_id},
                        "run": {"decision": decision, "notebook": "x.ipynb"},
                    }),
                    encoding="utf-8",
                )
            self.assertEqual(find_champion(root).experiment_id, "EXP-009")

    def test_spec_and_saved_record_require_stable_identifiers(self):
        spec = ExperimentSpec(
            experiment_id="EXP-006",
            title="Title",
            hypothesis="Hypothesis",
            changed_variable="One change",
            success_criterion="AP improves",
        )
        self.assertEqual(spec.seed, 42)
        self.assertEqual(spec.primary_metric, "average_precision")
        self.assertEqual(spec.secondary_metric, "epic_spearman")
        with self.assertRaises(ValueError):
            ExperimentSpec("6", "t", "h", "c", "s")
        with tempfile.TemporaryDirectory() as directory:
            record = {
                "experiment": {"experiment_id": "EXP-006"},
                "run": {"name": "run_001"},
                "metrics": {"ap": 0.5},
            }
            output = save_run_record(directory, record)
            self.assertTrue((output / "metadata.json").is_file())
            self.assertTrue((output / "metrics.csv").is_file())
            with self.assertRaises(FileExistsError):
                save_run_record(directory, record)


if __name__ == "__main__":
    unittest.main()
