from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

import pandas as pd

from ml_project import analyze_numeric_anomalies


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NumericAnomalyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = pd.DataFrame(
            {
                "value": [0.0, 1.0, 2.0, 3.0, 4.0, 100.0],
                "stable": [1.0] * 6,
            },
            index=[10, 11, 12, 13, 14, 15],
        )
        self.test = pd.DataFrame(
            {
                "value": [2.0, 200.0],
                "stable": [1.0, 2.0],
            },
            index=[20, 21],
        )

    def test_reference_bounds_are_reused_and_inputs_are_not_changed(self) -> None:
        original_train = self.train.copy(deep=True)
        report = analyze_numeric_anomalies(
            {"train": self.train, "test": self.test},
            reference_dataset="train",
            columns=["value", "stable"],
        )

        pd.testing.assert_frame_equal(self.train, original_train)
        value_bounds = report.feature_bounds.set_index("feature").loc["value"]
        self.assertTrue(bool(value_bounds["iqr_available"]))
        stable_bounds = report.feature_bounds.set_index("feature").loc["stable"]
        self.assertFalse(bool(stable_bounds["iqr_available"]))
        self.assertFalse(bool(stable_bounds["mad_available"]))

        flagged = report.flagged_values
        self.assertEqual(
            set(flagged["dataset"]),
            {"train", "test"},
        )
        self.assertEqual(
            flagged.loc[flagged["dataset"].eq("test"), "row_index"].tolist(),
            [21],
        )
        test_stable = report.dataset_summary[
            report.dataset_summary["dataset"].eq("test")
            & report.dataset_summary["feature"].eq("stable")
        ].iloc[0]
        self.assertEqual(int(test_stable["any_outliers"]), 0)
        self.assertEqual(int(test_stable["outside_reference_range"]), 1)
        self.assertEqual(report.row_summary.iloc[0]["feature_count"], 1)

    def test_invalid_parameters_and_schema_raise_clear_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "iqr_multiplier"):
            analyze_numeric_anomalies(
                {"train": self.train},
                reference_dataset="train",
                columns=["value"],
                iqr_multiplier=0,
            )
        with self.assertRaisesRegex(KeyError, "absent"):
            analyze_numeric_anomalies(
                {"train": self.train},
                reference_dataset="train",
                columns=["missing"],
            )


class AnomalyNotebookTests(unittest.TestCase):
    def test_notebook_is_valid_and_has_no_data_mutation(self) -> None:
        path = PROJECT_ROOT / "notebooks/02_eda_anomalies.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        ast.parse(code)
        self.assertIn("analyze_numeric_anomalies", code)
        self.assertNotIn("drop(", code)
        self.assertNotIn("clip(", code)
        self.assertNotIn("fillna(", code)

        report = (PROJECT_ROOT / "docs/02_eda.md").read_text(encoding="utf-8")
        self.assertIn("<!-- auto:eda-anomalies:start -->", report)
        self.assertIn("<!-- auto:eda-anomalies:end -->", report)


if __name__ == "__main__":
    unittest.main()
