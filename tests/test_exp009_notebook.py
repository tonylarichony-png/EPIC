"""Contract tests for the prepared or manually started EXP-009 notebook."""

from pathlib import Path
import unittest

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT / "notebooks/experiments/EXP-009_cpg_oe201_logistic.ipynb"
)


class TestExp009Notebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
        cls.source = "\n".join(cell.source for cell in cls.notebook.cells)
        cls.ids = [cell.id for cell in cls.notebook.cells]

    def test_metadata_registers_prepared_experiment(self):
        metadata = self.notebook.metadata["epic"]
        self.assertEqual(metadata["experiment_id"], "EXP-009")
        self.assertIn(
            metadata["status"],
            {"prepared_unexecuted", "in_progress_setup_complete", "completed"},
        )
        self.assertEqual(metadata["reference_experiment"], "EXP-007")
        self.assertEqual(metadata["rejected_predecessor"], "EXP-008")
        self.assertEqual(metadata["planned_cv_fits"], 15)
        self.assertFalse(metadata["threadpoolctl_during_fit"])

    def test_execution_history_is_well_formed(self):
        code_cells = [cell for cell in self.notebook.cells if cell.cell_type == "code"]
        self.assertTrue(code_cells)
        self.assertTrue(
            all(
                cell.execution_count is None or isinstance(cell.execution_count, int)
                for cell in code_cells
            )
        )
        self.assertTrue(all(isinstance(cell.outputs, list) for cell in code_cells))

    def test_notebook_schema_and_code_are_valid(self):
        nbformat.validate(self.notebook)
        self.assertEqual(len(self.ids), len(set(self.ids)))
        for cell in self.notebook.cells:
            if cell.cell_type == "code":
                with self.subTest(cell=cell.id):
                    compile(cell.source, f"<EXP-009:{cell.id}>", "exec")

    def test_training_and_validation_have_explicit_gates(self):
        self.assertLess(self.ids.index("stop-before-training"), self.ids.index("run-train-cv"))
        self.assertLess(self.ids.index("paired-cv-gate"), self.ids.index("enforce-cv-gate"))
        self.assertLess(self.ids.index("enforce-cv-gate"), self.ids.index("fit-full-train"))
        self.assertLess(self.ids.index("fit-full-train"), self.ids.index("validation-gate"))
        self.assertLess(self.ids.index("validation-gate"), self.ids.index("count-validation"))

    def test_protocol_is_visible_and_fixed(self):
        for required in (
            "CpG_count × valid_ACGT / (C_count × G_count)",
            "без clipping",
            "отдельная категория",
            "Reference — `[0.75,1.00)`",
            "candidate_columns=4386",
            "bounded chunks",
            "len(CONFIG.C_grid) == 5",
            "len(CONFIG.C_grid) * CONFIG.cv_folds == 15",
            "assert CONFIG.cv_folds * len(CONFIG.C_grid) == 15",
            "verbose=True",
            "mean AP > EXP-007",
            "AP wins on at least 2/3 folds",
            "mean Spearman >= EXP-007",
            "CV_GATE_PASSED",
            "EXP-007/run_001",
            "rejected_predecessor",
            "ALLOW_EXPLORATORY_COMPLETION",
            "exploratory_after_failed_cv_gate",
            "validation_used_for_C_selection",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.source)

    def test_rejected_exp008_is_not_the_candidate_baseline(self):
        self.assertIn("EXP-008 отклонён", self.source)
        self.assertNotIn("interaction(k2, GC-bin) + CpG", self.source)

    def test_threadpoolctl_is_not_called(self):
        self.assertNotIn("threadpool_limits(", self.source)

    def test_report_sync_is_last(self):
        nonempty_cells = [
            cell for cell in self.notebook.cells if cell.source.strip()
        ]
        self.assertEqual(nonempty_cells[-1].id, "sync-report")
        self.assertIn("sync_exp009_report", nonempty_cells[-1].source)

    def test_save_cell_supports_older_pandas(self):
        save_cell = self.notebook.cells[self.ids.index("save")].source
        self.assertNotIn("reset_index(names=", save_cell)
        self.assertIn(
            'rename("passed").rename_axis("criterion").reset_index()',
            save_cell,
        )

    def test_notebook_has_no_corrupted_unicode(self):
        self.assertNotIn("�", self.source)


if __name__ == "__main__":
    unittest.main()
