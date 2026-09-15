from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ml_project.experiment_state import (
    experiment_card_decisions,
    sync_experiment_state,
)


class ExperimentStateTests(unittest.TestCase):
    def test_card_decision_updates_registry_reports_without_touching_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiments = root / "experiments"
            docs = root / "docs"
            source_dir = root / "src/ml_project/experiments"
            experiments.mkdir(parents=True)
            docs.mkdir()
            source_dir.mkdir(parents=True)

            source = source_dir / "exp_002_feature.py"
            source.write_text("EXPERIMENT_CODE = 'immutable'\n", encoding="utf-8")
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            card = experiments / "EXP-002 Feature.md"
            card.write_text(
                "---\n"
                "id: EXP-002\n"
                "decision: reject\n"
                "eda_findings: []\n"
                "---\n\n"
                "# EXP-002 — Feature\n\n"
                "<!-- auto:experiment-report:start -->\n\n"
                "| Поле | Значение |\n"
                "| --- | --- |\n"
                "| Решение | pending |\n\n"
                "<!-- auto:experiment-report:end -->\n",
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "experiment_id": "EXP-002",
                        "title": "Feature",
                        "note": "experiments/EXP-002 Feature.md",
                        "hypothesis": "test",
                        "change": "one feature",
                        "run_name": "exp_002_v1",
                        "primary_metric": "accuracy",
                        "direction": "maximize",
                        "reference_score": 0.80,
                        "reference_std": 0.01,
                        "candidate_score": 0.79,
                        "improvement": -0.01,
                        "criteria_passed": False,
                        "decision": "pending",
                        "parent_experiment_id": "EXP-001",
                    }
                ]
            ).to_csv(experiments / "results.csv", index=False)
            (docs / "05_experiments.md").write_text(
                "<!-- auto:latest-experiment:start -->\nold\n"
                "<!-- auto:latest-experiment:end -->\n"
                "<!-- auto:experiment-leaderboard:start -->\nold\n"
                "<!-- auto:experiment-leaderboard:end -->\n"
                "<!-- auto:best-measured-result:start -->\nold\n"
                "<!-- auto:best-measured-result:end -->\n",
                encoding="utf-8",
            )
            (experiments / "_index.md").write_text(
                "<!-- auto:experiment-registry:start -->\nold\n"
                "<!-- auto:experiment-registry:end -->\n",
                encoding="utf-8",
            )
            (root / "README.md").write_text(
                "<!-- auto:key-results:start -->\nold\n"
                "<!-- auto:key-results:end -->\n",
                encoding="utf-8",
            )

            result = sync_experiment_state(root)

            self.assertEqual(result["registry_rows_updated"], 1)
            self.assertEqual(
                pd.read_csv(experiments / "results.csv").iloc[0]["decision"],
                "reject",
            )
            self.assertIn("| Решение | reject |", card.read_text(encoding="utf-8"))
            self.assertIn(
                "reject",
                (docs / "05_experiments.md").read_text(encoding="utf-8"),
            )
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), source_hash)

    def test_invalid_card_decision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cards = root / "experiments"
            cards.mkdir()
            (cards / "EXP-002 Invalid.md").write_text(
                "---\nid: EXP-002\ndecision: maybe\n---\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Invalid decision"):
                experiment_card_decisions(root)


if __name__ == "__main__":
    unittest.main()
