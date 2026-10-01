"""Contract tests for the prepared or manually executed EXP-008 notebook."""

from pathlib import Path
import unittest

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/experiments/EXP-008_k2_gc_interaction.ipynb"


class TestExp008Notebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
        cls.source = "\n".join(cell.source for cell in cls.notebook.cells)
        cls.ids = [cell.id for cell in cls.notebook.cells]

    def test_metadata_registers_prepared_experiment(self):
        metadata = self.notebook.metadata["epic"]
        self.assertEqual(metadata["experiment_id"], "EXP-008")
        self.assertIn(
            metadata["status"],
            {"prepared_unexecuted", "in_progress_cv_complete", "completed"},
        )
        self.assertEqual(metadata["reference_experiment"], "EXP-007")
        self.assertEqual(metadata["planned_cv_fits"], 15)
        self.assertFalse(metadata["threadpoolctl_during_fit"])

    def test_execution_history_is_well_formed(self):
        code_cells = [cell for cell in self.notebook.cells if cell.cell_type == "code"]
        self.assertTrue(code_cells)
        self.assertTrue(
            all(
                cell.execution_count is None
                or isinstance(cell.execution_count, int)
                for cell in code_cells
            )
        )
        self.assertTrue(all(isinstance(cell.outputs, list) for cell in code_cells))

    def test_every_code_cell_compiles(self):
        for cell in self.notebook.cells:
            if cell.cell_type == "code":
                with self.subTest(cell=cell.id):
                    compile(cell.source, f"<EXP-008:{cell.id}>", "exec")

    def test_training_and_validation_have_explicit_gates(self):
        self.assertLess(self.ids.index("stop-before-training"), self.ids.index("run-train-cv"))
        self.assertLess(self.ids.index("paired-cv-gate"), self.ids.index("enforce-cv-gate"))
        self.assertLess(self.ids.index("enforce-cv-gate"), self.ids.index("fit-full-train"))
        self.assertLess(self.ids.index("fit-full-train"), self.ids.index("validation-gate"))
        self.assertLess(self.ids.index("validation-gate"), self.ids.index("count-validation"))

    def test_protocol_is_visible(self):
        for required in (
            "128 reference-coded",
            "candidate_columns=4507",
            "len(CONFIG.C_grid) * CONFIG.cv_folds == 15",
            "verbose=True",
            "CV_GATE_PASSED",
            "EXP-007",
            "threadpoolctl_during_fit",
            "MKL_THREADING_LAYER",
            "Restart the kernel before EXP-008",
            "ALLOW_EXPLORATORY_COMPLETION",
            "exploratory_after_failed_cv_gate",
            "if CV_GATE_PASSED and VALIDATION_CHECKS_PASSED",
            "validation_used_for_C_selection",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.source)

    def test_notebook_has_no_corrupted_unicode(self):
        self.assertNotIn("�", self.source)

    def test_threadpoolctl_is_not_called(self):
        self.assertNotIn("threadpool_limits(", self.source)

    def test_report_sync_is_last(self):
        self.assertEqual(self.ids[-1], "sync-report")
        self.assertIn("sync_exp008_report", self.notebook.cells[-1].source)


if __name__ == "__main__":
    unittest.main()
