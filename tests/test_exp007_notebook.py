"""Contract tests for the deliberately unexecuted EXP-007 notebook."""

from pathlib import Path
import unittest

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/experiments/EXP-007_gc201_bins_logistic.ipynb"


class TestExp007Notebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
        cls.source = "\n".join(cell.source for cell in cls.notebook.cells)
        cls.ids = [cell.id for cell in cls.notebook.cells]

    def test_metadata_registers_prepared_experiment(self):
        metadata = self.notebook.metadata["epic"]
        self.assertEqual(metadata["experiment_id"], "EXP-007")
        self.assertEqual(metadata["status"], "prepared_unexecuted")
        self.assertEqual(metadata["reference_experiment"], "EXP-003")
        self.assertEqual(metadata["ablation_reference"], "EXP-005")
        self.assertEqual(metadata["diagnostic_reference"], "EXP-006")

    def test_every_code_cell_is_unexecuted_and_empty(self):
        code_cells = [cell for cell in self.notebook.cells if cell.cell_type == "code"]
        self.assertTrue(code_cells)
        self.assertTrue(all(cell.execution_count is None for cell in code_cells))
        self.assertTrue(all(cell.outputs == [] for cell in code_cells))

    def test_every_code_cell_compiles(self):
        for cell in self.notebook.cells:
            if cell.cell_type == "code":
                with self.subTest(cell=cell.id):
                    compile(cell.source, f"<EXP-007:{cell.id}>", "exec")

    def test_training_and_validation_have_explicit_gates(self):
        self.assertLess(self.ids.index("stop-before-training"), self.ids.index("run-train-cv"))
        self.assertLess(self.ids.index("paired-cv-gate"), self.ids.index("enforce-cv-gate"))
        self.assertLess(self.ids.index("enforce-cv-gate"), self.ids.index("fit-full-train"))
        self.assertLess(self.ids.index("fit-full-train"), self.ids.index("validation-gate"))
        self.assertLess(self.ids.index("validation-gate"), self.ids.index("count-validation"))

    def test_protocol_is_visible(self):
        for required in (
            "[0.35,0.40)",
            "candidate_columns=4379",
            "sample_weight",
            "verbose=True",
            "CV_GATE_PASSED",
            "AP > EXP-005 on every identical train-CV fold",
            "validation_used_for_C_selection",
            "EXP-006",
            "EXP-005",
            "EXP-003",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.source)

    def test_linear_gc_feature_is_not_used(self):
        self.assertNotIn("gc201_z", self.source)
        self.assertIn("aggregate_binned_gc_binary_likelihood", self.source)

    def test_report_sync_is_last(self):
        self.assertEqual(self.ids[-1], "sync-report")
        self.assertIn("sync_exp007_report", self.notebook.cells[-1].source)


if __name__ == "__main__":
    unittest.main()
