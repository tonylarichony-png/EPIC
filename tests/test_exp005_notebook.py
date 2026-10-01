"""Contract tests for the completed, manually executed EXP-005 notebook."""
from pathlib import Path
import unittest

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "notebooks/experiments/EXP-005_hierarchical_6mer_logistic.ipynb"
)


class TestExp005Notebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
        cls.source = "\n".join(cell.source for cell in cls.notebook.cells)
        cls.ids = [cell.id for cell in cls.notebook.cells]

    def test_metadata_registers_completed_experiment(self):
        metadata = self.notebook.metadata["epic"]
        self.assertEqual(metadata["experiment_id"], "EXP-005")
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(metadata["reference_experiment"], "EXP-003")
        self.assertEqual(metadata["candidate_k"], 6)

    def test_executed_cells_have_no_error_outputs(self):
        code_cells = [
            cell for cell in self.notebook.cells if cell.cell_type == "code"
        ]
        self.assertTrue(code_cells)
        self.assertTrue(any(cell.execution_count is not None for cell in code_cells))
        errors = [
            output
            for cell in code_cells
            for output in cell.outputs
            if output.output_type == "error"
        ]
        self.assertEqual(errors, [])

    def test_every_code_cell_compiles(self):
        for cell in self.notebook.cells:
            if cell.cell_type == "code":
                with self.subTest(cell=cell.id):
                    compile(cell.source, f"<EXP-005:{cell.id}>", "exec")

    def test_training_and_validation_are_behind_explicit_gates(self):
        self.assertLess(
            self.ids.index("stop-before-training"),
            self.ids.index("run-train-cv"),
        )
        self.assertLess(
            self.ids.index("select-c"),
            self.ids.index("fit-full-train"),
        )
        self.assertLess(
            self.ids.index("validation-gate"),
            self.ids.index("count-validation"),
        )
        self.assertLess(
            self.ids.index("fit-full-train"),
            self.ids.index("count-validation"),
        )

    def test_protocol_is_fully_visible(self):
        for required in (
            "2-mer + 4-mer + 6-mer",
            "4371",
            "8194",
            "solver=\"lbfgs\"",
            "learning rate",
            "validation_used_for_C_selection",
            "relative validation AP gain >= 1%",
            "cross_validate_hierarchical_kmer_logistic",
            "fit_hierarchical_kmer_logistic",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.source)

    def test_report_sync_is_the_last_step(self):
        self.assertEqual(self.ids[-1], "sync-report")
        self.assertIn("sync_exp005_report", self.notebook.cells[-1].source)


if __name__ == "__main__":
    unittest.main()
