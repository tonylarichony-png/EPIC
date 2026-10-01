from pathlib import Path
import json
import sys
import tempfile
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.cnn_experiment_report import finalize_cnn_experiment
from ml_project.cnn_experiment_scaffold import create_cnn_experiment


class TestCNNExperimentWorkflow(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "cnn_exp").mkdir()
        notebook_dir = self.root / "notebooks/experiments/CNN_experiments"
        notebook_dir.mkdir(parents=True)
        reference_notebook = notebook_dir / "CNN_EXP002.ipynb"
        reference_notebook.write_text(
            json.dumps(
                {
                    "cells": [
                        {
                            "cell_type": "markdown",
                            "id": "title",
                            "metadata": {},
                            "source": ["# CNN-EXP-002 — reference"],
                        },
                        {
                            "cell_type": "code",
                            "execution_count": 1,
                            "id": "config",
                            "metadata": {},
                            "outputs": [{"output_type": "stream", "name": "stdout", "text": ["old"]}],
                            "source": [
                                "EXP002_DIR = 'CNN_EXP002/presence_intensity_v1'\n"
                            ],
                        },
                    ],
                    "metadata": {},
                    "nbformat": 4,
                    "nbformat_minor": 5,
                }
            ),
            encoding="utf-8",
        )
        metrics_dir = self.root / "artifacts/reference"
        metrics_dir.mkdir(parents=True)
        self.reference_metrics = metrics_dir / "validation_metrics.csv"
        self._metrics(0.01, 0.15).to_csv(self.reference_metrics, index=False)
        (self.root / "cnn_exp/CNN-EXP-002.md").write_text("reference", encoding="utf-8")
        (self.root / "cnn_exp/champion.json").write_text(
            json.dumps(
                {
                    "experiment_id": "CNN-EXP-002",
                    "notebook": reference_notebook.relative_to(self.root).as_posix(),
                    "metrics": self.reference_metrics.relative_to(self.root).as_posix(),
                    "checkpoint": "artifacts/reference/model.pt",
                }
            ),
            encoding="utf-8",
        )
        (self.root / "cnn_exp/_index.md").write_text(
            "# CNN\n\n| ID | Модель | Validation AP | EPIC Spearman | Решение |\n"
            "|---|---|---:|---:|---|\n"
            "| [[cnn_exp/CNN-EXP-002.md\\|CNN-EXP-002]] | ref | 0.01 | 0.15 | `adopt` |\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _metrics(ap, spearman):
        return pd.DataFrame(
            {
                "score": ["expected_signal", "expected_signal"],
                "segment": ["overall", "c1"],
                "average_precision": [ap, ap * 1.1],
                "epic_spearman": [spearman, spearman + 0.01],
            }
        )

    def test_scaffold_and_finalize(self):
        notebook, card, artifacts = create_cnn_experiment(
            self.root,
            title="Wide context",
            hypothesis="A wider context improves both metrics",
        )
        self.assertEqual(card.name, "CNN-EXP-003.md")
        created = json.loads(notebook.read_text(encoding="utf-8"))
        self.assertTrue(all(cell.get("execution_count") is None for cell in created["cells"] if cell["cell_type"] == "code"))
        sources = "\n".join("".join(cell.get("source", [])) for cell in created["cells"])
        self.assertIn("CNN_EXP003", sources)
        self.assertIn("EXP003_DIR", sources)
        self.assertIn("SAVE_EXPERIMENT = False", sources)

        history = pd.DataFrame(
            {
                "step": [1, 10],
                "ema_presence": [0.7, 0.5],
                "ema_intensity": [0.8, 0.4],
                "ema_total": [0.78, 0.54],
            }
        )
        history_path = artifacts / "training_history.csv"
        history.to_csv(history_path, index=False)
        candidate_path = artifacts / "validation_metrics.csv"
        self._metrics(0.012, 0.17).to_csv(candidate_path, index=False)

        finalized = finalize_cnn_experiment(
            project_root=self.root,
            experiment_id="CNN-EXP-003",
            title="Wide context",
            hypothesis="A wider context improves both metrics",
            notebook_path=notebook.relative_to(self.root),
            artifact_dir=artifacts,
            training_history=history_path,
            candidate_metrics=candidate_path,
            reference_experiment="CNN-EXP-002",
            reference_metrics=self.reference_metrics.relative_to(self.root),
            decision="review",
        )
        self.assertTrue(finalized.summary_path.is_file())
        self.assertTrue((finalized.plot_dir / "01_training_losses.png").is_file())
        self.assertIn("0.012000000", card.read_text(encoding="utf-8"))
        self.assertIn("CNN-EXP-003", (self.root / "cnn_exp/_index.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
