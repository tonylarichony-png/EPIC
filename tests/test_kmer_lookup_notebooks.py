"""Contract tests for the EXP-002/003/004 k-mer lookup notebooks."""

from pathlib import Path
import unittest

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "EXP-002": (
        PROJECT_ROOT / "notebooks/experiments/EXP-002_4mer_lookup.ipynb",
        4,
    ),
    "EXP-003": (
        PROJECT_ROOT / "notebooks/experiments/EXP-003_6mer_lookup.ipynb",
        6,
    ),
    "EXP-004": (
        PROJECT_ROOT / "notebooks/experiments/EXP-004_8mer_lookup.ipynb",
        8,
    ),
}


class TestKmerLookupNotebooks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebooks = {
            experiment_id: nbformat.read(path, as_version=4)
            for experiment_id, (path, _) in NOTEBOOKS.items()
        }

    def test_metadata_identifies_experiment_and_candidate_k(self):
        for experiment_id, (_, candidate_k) in NOTEBOOKS.items():
            with self.subTest(experiment_id=experiment_id):
                metadata = self.notebooks[experiment_id].metadata["epic"]
                self.assertEqual(metadata["experiment_id"], experiment_id)
                self.assertEqual(metadata["candidate_k"], candidate_k)
                self.assertEqual(
                    metadata["lookup_module"],
                    "ml_project.epic_kmer",
                )

    def test_smoothing_is_fixed_before_evaluation(self):
        for experiment_id, notebook in self.notebooks.items():
            source = "\n".join(cell.source for cell in notebook.cells)
            with self.subTest(experiment_id=experiment_id):
                self.assertIn(
                    'smoothing="one_expected_positive"',
                    source,
                )
                self.assertIn(
                    '"one_expected_positive"',
                    source,
                )

    def test_code_contains_no_iterative_training_controls(self):
        for experiment_id, notebook in self.notebooks.items():
            code_source = "\n".join(
                cell.source
                for cell in notebook.cells
                if cell.cell_type == "code"
            ).lower()
            with self.subTest(experiment_id=experiment_id):
                for forbidden in ("solver", "epoch", "loss"):
                    self.assertNotIn(forbidden, code_source)

    def test_every_code_cell_compiles(self):
        for experiment_id, notebook in self.notebooks.items():
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    with self.subTest(
                        experiment_id=experiment_id,
                        cell=cell.id,
                    ):
                        compile(
                            cell.source,
                            f"<{experiment_id}:{cell.id}>",
                            "exec",
                        )

    def test_executed_notebooks_are_complete_and_error_free(self):
        for experiment_id, notebook in self.notebooks.items():
            code_cells = [
                cell for cell in notebook.cells if cell.cell_type == "code"
            ]
            execution_counts = [cell.execution_count for cell in code_cells]
            if not any(count is not None for count in execution_counts):
                continue

            with self.subTest(experiment_id=experiment_id):
                self.assertTrue(
                    all(count is not None for count in execution_counts),
                    "Partially executed notebook must not be committed",
                )
                errors = [
                    output
                    for cell in code_cells
                    for output in cell.outputs
                    if output.output_type == "error"
                ]
                self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
