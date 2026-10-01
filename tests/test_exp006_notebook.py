"""Contract tests for the deliberately unexecuted EXP-006 notebook."""

from pathlib import Path
import unittest

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/experiments/EXP-006_gc201_logistic.ipynb"


class TestExp006Notebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
        cls.source = "\n".join(cell.source for cell in cls.notebook.cells)
        cls.ids = [cell.id for cell in cls.notebook.cells]

    def test_metadata_registers_prepared_experiment(self):
        metadata = self.notebook.metadata["epic"]
        self.assertEqual(metadata["experiment_id"], "EXP-006")
        self.assertEqual(metadata["status"], "prepared_unexecuted")
        self.assertEqual(metadata["reference_experiment"], "EXP-003")
        self.assertEqual(metadata["ablation_reference"], "EXP-005")
        self.assertEqual(metadata["candidate_feature"], "GC201")

    def test_every_code_cell_is_unexecuted_and_empty(self):
        code_cells = [cell for cell in self.notebook.cells if cell.cell_type == "code"]
        self.assertTrue(code_cells)
        self.assertTrue(all(cell.execution_count is None for cell in code_cells))
        self.assertTrue(all(cell.outputs == [] for cell in code_cells))

    def test_every_code_cell_compiles(self):
        for cell in self.notebook.cells:
            if cell.cell_type == "code":
                with self.subTest(cell=cell.id):
                    compile(cell.source, f"<EXP-006:{cell.id}>", "exec")

    def test_training_and_validation_have_explicit_gates(self):
        self.assertLess(self.ids.index("stop-before-training"), self.ids.index("run-train-cv"))
        self.assertLess(self.ids.index("select-c"), self.ids.index("fit-full-train"))
        self.assertLess(self.ids.index("fit-full-train"), self.ids.index("validation-gate"))
        self.assertLess(self.ids.index("validation-gate"), self.ids.index("count-validation"))

    def test_protocol_is_visible(self):
        for required in (
            "[-100,+100]",
            "gc_count",
            "valid_count",
            "4372",
            "sample_weight",
            "cross_validate_gc_logistic",
            "fit_gc_logistic",
            "validation_used_for_C_selection",
            "EXP-005",
            "EXP-003",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.source)

    def test_report_sync_is_last(self):
        self.assertEqual(self.ids[-1], "sync-report")
        self.assertIn("sync_exp006_report", self.notebook.cells[-1].source)


if __name__ == "__main__":
    unittest.main()
