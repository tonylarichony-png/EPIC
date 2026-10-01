"""Contract tests for the readable EXP-001 notebook."""

from pathlib import Path
import unittest

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "notebooks/experiments/EXP-001_dinucleotide_baseline.ipynb"
)


class TestEpicBaselineNotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
        cls.source = "\n".join(cell.source for cell in cls.notebook.cells)

    def test_notebook_is_registered_as_exp_001(self):
        self.assertEqual(
            self.notebook.metadata["epic"]["experiment_id"],
            "EXP-001",
        )
        self.assertEqual(
            self.notebook.metadata["epic"]["baseline_module"],
            "ml_project.epic_baseline",
        )

    def test_protocol_makes_exact_aggregation_and_metrics_explicit(self):
        for required in (
            "17 × 2 = 34",
            "Это **не subsampling**",
            "Average Precision",
            "EPIC Spearman",
            "N/edge",
            "loss vs epoch",
            "ConvergenceWarning",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.source)

    def test_notebook_uses_shared_modules_and_saves_both_metrics(self):
        for required in (
            "ensure_prepared_split",
            "fit_dinucleotide_baseline",
            "evaluate_dinucleotide_baseline",
            '"reference_average_precision"',
            '"candidate_average_precision"',
            '"reference_epic_spearman"',
            '"candidate_epic_spearman"',
            "save_run_record",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.source)

    def test_every_code_cell_compiles(self):
        for cell in self.notebook.cells:
            if cell.cell_type == "code":
                with self.subTest(cell=cell.id):
                    compile(cell.source, f"<{cell.id}>", "exec")

    def test_saved_notebook_has_successful_outputs(self):
        code_cells = [
            cell for cell in self.notebook.cells if cell.cell_type == "code"
        ]
        self.assertTrue(all(cell.execution_count is not None for cell in code_cells))
        errors = [
            output
            for cell in code_cells
            for output in cell.outputs
            if output.output_type == "error"
        ]
        self.assertEqual(errors, [])
        self.assertIn(
            'DECISION = "adopt" if success else "iterate"',
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
