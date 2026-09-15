from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ml_project import (
    DataCatalog,
    DatasetProfiler,
    MarkdownDocument,
    build_eda_blocks,
    build_field_descriptions_template,
    grouped_target_report,
    save_eda_finding,
    validate_feature_groups,
)
from ml_project.docsync import dataframe_to_markdown


class DatasetProfilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "id": [1, 2, 2],
                "target": [0, 1, 1],
                "value": [10.0, None, 30.0],
            }
        )
        self.profile = DatasetProfiler(
            self.frame,
            name="train",
            key="id",
            target="target",
        )

    def test_schema_and_missing_are_separate_reports(self) -> None:
        schema = self.profile.schema_report()
        missing = self.profile.missing_report()

        self.assertEqual(schema.columns.tolist(), ["field", "dtype"])
        self.assertEqual(missing["field"].tolist(), ["value"])
        self.assertEqual(int(missing.iloc[0]["missing"]), 1)

    def test_duplicate_and_target_reports(self) -> None:
        duplicate = self.profile.duplicate_report().iloc[0]
        target = self.profile.target_report()

        self.assertEqual(int(duplicate["key_duplicates"]), 1)
        self.assertEqual(target["count"].tolist(), [1, 2])

    def test_explicit_groups_control_numeric_and_categorical_reports(self) -> None:
        groups = validate_feature_groups(
            self.frame,
            {
                "identifier": ["id"],
                "categorical": [],
                "numeric": ["value"],
            },
            target="target",
        )

        numeric = self.profile.numeric_report(columns=groups["numeric"])
        categorical = self.profile.categorical_report(
            columns=groups["categorical"]
        )

        self.assertEqual(numeric["field"].tolist(), ["value"])
        self.assertTrue(categorical.empty)

    def test_feature_groups_reject_duplicate_and_unassigned_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "назначены несколько раз"):
            validate_feature_groups(
                self.frame,
                {
                    "identifier": ["id"],
                    "numeric": ["value", "id"],
                },
                target="target",
            )


class DataCatalogTests(unittest.TestCase):
    def test_file_report_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "data" / "raw"
            raw.mkdir(parents=True)
            pd.DataFrame({"id": [1, 2]}).to_csv(raw / "train.csv", index=False)

            catalog = DataCatalog(
                root,
                Path("data/raw"),
                {
                    "train": {
                        "filename": "train.csv",
                        "role": "train",
                        "required": True,
                    }
                },
            )
            report = catalog.file_report().iloc[0]

            self.assertEqual(int(report["rows"]), 2)
            self.assertEqual(int(report["columns"]), 1)
            self.assertEqual(len(str(report["sha256"])), 64)

            template = build_field_descriptions_template(
                catalog,
                {"id": "Идентификатор строки."},
            )
            self.assertEqual(template.count("'id':"), 1)
            self.assertIn("'id': 'Идентификатор строки.'", template)


class EdaBlockTests(unittest.TestCase):
    def test_service_dataset_is_excluded_from_generated_eda_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "data" / "raw"
            raw.mkdir(parents=True)

            pd.DataFrame(
                {"id": [1, 2], "target": [0, 1], "value": [10.0, None]}
            ).to_csv(raw / "train.csv", index=False)
            pd.DataFrame(
                {"id": [3], "value": [20.0]}
            ).to_csv(raw / "test.csv", index=False)
            pd.DataFrame(
                {"id": [3], "target": [0]}
            ).to_csv(raw / "submission.csv", index=False)

            specs = {
                "train": {"filename": "train.csv", "role": "train"},
                "test": {"filename": "test.csv", "role": "inference"},
                "submission": {
                    "filename": "submission.csv",
                    "role": "submission_example",
                },
            }
            catalog = DataCatalog(root, Path("data/raw"), specs)
            datasets = catalog.load_all()
            profiles = {
                "train": DatasetProfiler(
                    datasets["train"], name="train", key="id", target="target"
                ),
                "test": DatasetProfiler(
                    datasets["test"], name="test", key="id"
                ),
            }

            blocks = build_eda_blocks(
                catalog,
                profiles,
                train_dataset="train",
            )

            self.assertIn("train", blocks["eda-snapshot"])
            self.assertIn("test", blocks["eda-snapshot"])
            self.assertNotIn("submission", blocks["eda-snapshot"])
            self.assertNotIn("submission", blocks["eda-quality"])


class GroupedTargetReportTests(unittest.TestCase):
    def test_binary_report_contains_rate_and_bootstrap_interval(self) -> None:
        frame = pd.DataFrame(
            {
                "segment": ["a", "a", "a", "b", "b"],
                "target": [1, 1, 1, 0, 0],
            }
        )

        report = grouped_target_report(
            frame,
            feature="segment",
            target="target",
            n_boot=100,
            seed=7,
        )

        self.assertEqual(
            report.columns.tolist(),
            ["rows", "target_rate", "ci_low", "ci_high"],
        )
        self.assertEqual(float(report.loc["a", "target_rate"]), 1.0)
        self.assertEqual(float(report.loc["a", "ci_low"]), 1.0)
        self.assertEqual(float(report.loc["b", "ci_high"]), 0.0)


class MarkdownDocumentTests(unittest.TestCase):
    def test_markdown_table_is_padded_and_numeric_columns_are_aligned(self) -> None:
        table = dataframe_to_markdown(
            pd.DataFrame(
                {
                    "Название": ["короткое", "длинное значение"],
                    "Строки": [9, 125],
                }
            )
        )
        lines = table.splitlines()

        self.assertEqual(len({len(line) for line in lines}), 1)
        self.assertIn("---:", lines[1])

    def test_only_generated_block_is_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            path.write_text(
                "Manual before\n"
                "<!-- auto:test:start -->\n"
                "old\n"
                "<!-- auto:test:end -->\n"
                "Manual after\n",
                encoding="utf-8",
            )

            MarkdownDocument(path).update_blocks({"test": "new"})
            result = path.read_text(encoding="utf-8")

            self.assertIn("Manual before", result)
            self.assertIn("Manual after", result)
            self.assertIn(
                "<!-- auto:test:start -->\n\nnew\n\n<!-- auto:test:end -->",
                result,
            )
            self.assertNotIn("\nold\n", result)

    def test_windows_lock_falls_back_to_in_place_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            path.write_text(
                "<!-- auto:test:start -->\nold\n<!-- auto:test:end -->\n",
                encoding="utf-8",
            )

            with patch(
                "ml_project.docsync.os.replace",
                side_effect=PermissionError("destination is open"),
            ):
                updated = MarkdownDocument(path).update_blocks({"test": "new"})

            self.assertEqual(updated, ["test"])
            self.assertIn("\n\nnew\n\n", path.read_text(encoding="utf-8"))
            self.assertFalse(path.with_suffix(".md.tmp").exists())


class EdaFindingTests(unittest.TestCase):
    class FakeFigure:
        def savefig(self, path: Path, **_: object) -> None:
            Path(path).write_bytes(b"PNG")

    def test_finding_creates_card_figure_and_docs_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            docs.mkdir()
            (docs / "02_eda.md").write_text(
                "Manual before\n"
                "<!-- auto:eda-findings:start -->\n"
                "empty\n"
                "<!-- auto:eda-findings:end -->\n"
                "Manual after\n",
                encoding="utf-8",
            )

            finding = save_eda_finding(
                project_root=root,
                title="Важная связь",
                question="Связаны ли признаки?",
                method="Корреляция",
                conclusion="Обнаружена устойчивая связь.",
                figure=self.FakeFigure(),
                features=["feature_a", "feature_b"],
                hypothesis="Проверить interaction-feature.",
            )

            self.assertEqual(finding.finding_id, "EDA-001")
            self.assertTrue((root / finding.note_path).exists())
            self.assertTrue((root / finding.figure_path).exists())

            note = (root / finding.note_path).read_text(encoding="utf-8")
            report = (docs / "02_eda.md").read_text(encoding="utf-8")
            self.assertIn("Обнаружена устойчивая связь.", note)
            self.assertIn("![[assets/eda/EDA-001.png]]", note)
            self.assertIn(
                "[[eda/findings/EDA-001.md|EDA-001 — Важная связь]]",
                report,
            )
            self.assertIn("Manual before", report)
            self.assertIn("Manual after", report)

    def test_table_only_finding_embeds_formatted_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            docs.mkdir()
            (docs / "02_eda.md").write_text(
                "<!-- auto:eda-findings:start -->\n"
                "empty\n"
                "<!-- auto:eda-findings:end -->\n",
                encoding="utf-8",
            )
            frame = pd.DataFrame(
                {
                    "rows": [314, 577],
                    "target_rate": [0.742, 0.1889],
                },
                index=pd.Index(["group_a", "group_b"], name="segment"),
            )

            finding = save_eda_finding(
                project_root=root,
                title="Табличное наблюдение",
                question="Различается ли target?",
                method="Группировка",
                conclusion="Группы различаются.",
                tables={"Target report": frame},
                table_formats={
                    "Target report": {"target_rate": "{:.2%}"}
                },
            )

            self.assertIsNone(finding.figure_path)
            self.assertEqual(finding.table_paths, ())
            note = (root / finding.note_path).read_text(encoding="utf-8")
            self.assertIn("## Таблицы", note)
            self.assertIn("### Target report", note)
            self.assertIn("segment", note)
            self.assertIn("74.20%", note)
            self.assertNotIn("## График", note)

    def test_large_table_gets_preview_and_full_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            docs.mkdir()
            (docs / "02_eda.md").write_text(
                "<!-- auto:eda-findings:start -->\n"
                "empty\n"
                "<!-- auto:eda-findings:end -->\n",
                encoding="utf-8",
            )
            frame = pd.DataFrame({"value": range(35)})

            finding = save_eda_finding(
                project_root=root,
                title="Большая таблица",
                question="Какие строки найдены?",
                method="Табличный отчёт",
                conclusion="Таблица сохранена полностью.",
                tables={"Полный отчёт": frame},
                table_preview_rows=3,
            )

            self.assertIsNone(finding.figure_path)
            self.assertEqual(len(finding.table_paths), 1)
            csv_path = root / finding.table_paths[0]
            self.assertTrue(csv_path.exists())
            self.assertEqual(len(pd.read_csv(csv_path)), 35)
            note = (root / finding.note_path).read_text(encoding="utf-8")
            self.assertIn("Показаны первые 3 из 35 строк", note)
            self.assertIn(f"[[{finding.table_paths[0].as_posix()}]]", note)


if __name__ == "__main__":
    unittest.main()
