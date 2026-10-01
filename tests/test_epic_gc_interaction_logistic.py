"""Synthetic contract tests for EXP-008 interaction mechanics."""

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.epic_gc_interaction_logistic import (
    INTERACTION_C_GRID,
    GCInteractionLogisticConfig,
    K2_REFERENCE_CODE,
    _interaction_matrix,
    aggregate_gc_interaction_likelihood,
    interaction_enrichment_table,
    interaction_feature_table,
)
from ml_project.epic_kmer_logistic import hierarchical_kmer_design
from tests.test_epic_gc_binned_logistic import _toy_counts


class TestEpicGCInteractionLogistic(unittest.TestCase):
    def setUp(self):
        self.config = GCInteractionLogisticConfig()
        self.design = hierarchical_kmer_design(self.config)

    def test_grid_is_five_values_and_exactly_fifteen_cv_fits(self):
        self.assertEqual(self.config.C_grid, INTERACTION_C_GRID)
        self.assertEqual(len(self.config.C_grid) * self.config.cv_folds, 15)
        self.assertIn(3e-4, self.config.C_grid)

    def test_interaction_vocabulary_is_reference_coded(self):
        table = interaction_feature_table(self.config)
        self.assertEqual(len(table), 128)
        self.assertNotIn("TT", set(table["k2"]))
        self.assertNotIn("[0.35,0.40)", set(table["gc_bin"]))
        self.assertFalse(table[["k2_code", "gc_bin_code"]].duplicated().any())

    def test_interaction_activates_only_when_both_terms_are_nonreference(self):
        matrix = _interaction_matrix(
            np.array([K2_REFERENCE_CODE, 0, K2_REFERENCE_CODE, 0]),
            np.array([3, 2, 2, 3]),
            self.config,
        )
        active = np.asarray(matrix.sum(axis=1)).ravel()
        np.testing.assert_array_equal(active, np.array([0.0, 0.0, 0.0, 1.0]))

    def test_exact_aggregate_adds_128_sparse_columns(self):
        aggregate = aggregate_gc_interaction_likelihood(
            _toy_counts(), self.design, config=self.config
        )
        self.assertEqual(aggregate.matrix.shape[1], 4507)
        self.assertEqual(int(aggregate.rows["population_count"].sum()), 300)
        weights = aggregate.rows.groupby("target")["sample_weight"].sum()
        self.assertAlmostEqual(weights.loc[0], weights.loc[1])

    def test_train_diagnostic_preserves_population(self):
        table = interaction_enrichment_table(_toy_counts(), self.design)
        self.assertEqual(len(table), 17 * 9)
        self.assertEqual(int(table["positions"].sum()), 300)
        self.assertEqual(int(table["positives"].sum()), 7)

    def test_grid_cannot_be_changed_inside_exp008(self):
        with self.assertRaises(ValueError):
            GCInteractionLogisticConfig(C_grid=(1e-4, 3e-4))


if __name__ == "__main__":
    unittest.main()
