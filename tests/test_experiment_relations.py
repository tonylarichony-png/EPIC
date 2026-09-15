from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ml_project import sync_experiment_eda_relations


class ExperimentEdaRelationTests(unittest.TestCase):
    def _write_cards(self, root: Path) -> tuple[Path, Path]:
        experiments = root / "experiments"
        findings = root / "eda/findings"
        experiments.mkdir(parents=True)
        findings.mkdir(parents=True)
        experiment_path = experiments / "EXP-010 Sex Pclass.md"
        experiment_path.write_text(
            "---\n"
            'id: "EXP-010"\n'
            "type: experiment\n"
            'hypothesis: "Совместная категория улучшит accuracy"\n'
            'primary_metric: "accuracy"\n'
            "decision: pending\n"
            'eda_findings: ["EDA-003"]\n'
            "---\n\n"
            "# EXP-010 — Sex × Pclass\n\n"
            "## Анализ результата — заполнить вручную\n",
            encoding="utf-8",
        )
        finding_path = findings / "EDA-003.md"
        finding_path.write_text(
            "---\n"
            'id: "EDA-003"\n'
            'title: "Влияние класса и пола"\n'
            'features: ["Sex", "Pclass"]\n'
            "---\n\n"
            "# EDA-003 — Влияние класса и пола\n\n"
            "> [!abstract] Ключевой вывод\n"
            "> Эффект класса различается по полу.\n\n"
            "## Источник\n",
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {
                    "experiment_id": "EXP-010",
                    "primary_metric": "accuracy",
                    "improvement": 0.0067,
                }
            ]
        ).to_csv(experiments / "results.csv", index=False)
        return experiment_path, finding_path

    def test_links_are_generated_in_both_directions_and_can_be_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment_path, finding_path = self._write_cards(root)

            result = sync_experiment_eda_relations(root)

            self.assertEqual(result["relations"], 1)
            experiment = experiment_path.read_text(encoding="utf-8")
            finding = finding_path.read_text(encoding="utf-8")
            self.assertIn("EDA-003 — Влияние класса и пола", experiment)
            self.assertIn("Sex, Pclass", experiment)
            self.assertIn("EXP-010 — Sex × Pclass", finding)
            self.assertIn("+0.0067", finding)

            experiment_path.write_text(
                experiment.replace(
                    'eda_findings: ["EDA-003"]',
                    "eda_findings: []",
                ),
                encoding="utf-8",
            )
            sync_experiment_eda_relations(root)
            finding = finding_path.read_text(encoding="utf-8")
            self.assertNotIn("EXP-010 — Sex × Pclass", finding)
            self.assertIn("пока не зарегистрированы", finding)

    def test_missing_finding_is_rejected_before_relation_blocks_are_added(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment_path, finding_path = self._write_cards(root)
            finding_path.unlink()

            with self.assertRaisesRegex(ValueError, "EDA-003"):
                sync_experiment_eda_relations(root)

            experiment = experiment_path.read_text(encoding="utf-8")
            self.assertNotIn("auto:experiment-eda-links", experiment)

    def test_obsidian_yaml_block_list_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment_path, finding_path = self._write_cards(root)
            experiment = experiment_path.read_text(encoding="utf-8")
            experiment_path.write_text(
                experiment.replace(
                    'eda_findings: ["EDA-003"]',
                    "eda_findings:\n  - EDA-003",
                ),
                encoding="utf-8",
            )

            result = sync_experiment_eda_relations(root)

            self.assertEqual(result["relations"], 1)
            self.assertIn(
                "EDA-003",
                experiment_path.read_text(encoding="utf-8"),
            )
            self.assertIn(
                "EXP-010",
                finding_path.read_text(encoding="utf-8"),
            )

    def test_empty_existing_card_gets_editable_field_and_safe_block(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiments = root / "experiments"
            experiments.mkdir(parents=True)
            path = experiments / "EXP-002 Test.md"
            path.write_text(
                "---\n"
                "id: EXP-002\n"
                "type: experiment\n"
                "decision: reject\n"
                "---\n\n"
                "# EXP-002 — Test\n\n"
                "## Анализ результата — заполнить вручную\n",
                encoding="utf-8",
            )

            sync_experiment_eda_relations(root)
            text = path.read_text(encoding="utf-8")

            self.assertIn("eda_findings: []", text)
            self.assertIn("auto:experiment-eda-links:start", text)
            self.assertIn("Анализ результата — заполнить вручную", text)


if __name__ == "__main__":
    unittest.main()
