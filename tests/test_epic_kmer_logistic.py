"""Non-training contract tests for the EXP-005 feature machinery."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.epic_kmer import KmerCounts
from ml_project.epic_kmer_logistic import (
    HierarchicalKmerLogisticConfig,
    aggregate_binary_likelihood,
    hierarchical_kmer_design,
    make_balanced_contig_folds,
    select_regularization,
)


def _toy_counts() -> KmerCounts:
    totals = pd.DataFrame(
        {
            "k": [6] * 8,
            "contig": ["c1", "c2", "c3", "c4", "c5", "c6", "c1", "c2"],
            "strand": ["+"] * 6 + ["-", "-"],
            "category_code": [0, 1, 2, 3, 4, 5, 1, 2],
            "total_positions": [100, 90, 80, 70, 60, 50, 20, 10],
        }
    )
    positives = pd.DataFrame(
        {
            "template_index": range(8),
            "contig": ["c1", "c1", "c2", "c3", "c4", "c5", "c6", "c6"],
            "coordinate_0based": range(8),
            "strand": ["+"] * 8,
            "count": [1, 2, 1, 3, 1, 4, 2, 5],
            "kmer_2_code": [0, 1, 1, 2, 3, 4, 5, 5],
            "kmer_4_code": [0, 1, 1, 2, 3, 4, 5, 5],
            "kmer_6_code": [0, 1, 1, 2, 3, 4, 5, 5],
        }
    )
    return KmerCounts("train", (2, 4, 6), totals, positives)


class TestEpicKmerLogisticFeatures(unittest.TestCase):
    def setUp(self):
        self.config = HierarchicalKmerLogisticConfig()
        self.design = hierarchical_kmer_design(self.config)

    def test_design_shape_and_one_active_feature_per_level(self):
        self.assertEqual(self.design.matrix.shape, (4097, 4371))
        self.assertTrue(np.all(self.design.matrix.getnnz(axis=1) == 3))
        self.assertEqual(len(self.design.feature_table), 4371)

    def test_canonical_category_maps_to_suffixes(self):
        code = int("123", 4)
        row = self.design.category_table.iloc[code]
        self.assertEqual(row["kmer_2_code"], code % (4**2))
        self.assertEqual(row["kmer_4_code"], code % (4**4))
        self.assertEqual(row["kmer_6_code"], code)

    def test_n_edge_activates_n_edge_at_every_level(self):
        row = self.design.category_table.iloc[-1]
        self.assertEqual(row["max_category"], "N/edge")
        self.assertEqual(row["kmer_2_code"], 4**2)
        self.assertEqual(row["kmer_4_code"], 4**4)
        self.assertEqual(row["kmer_6_code"], 4**6)

    def test_exact_aggregate_preserves_population_and_balances_weights(self):
        aggregate = aggregate_binary_likelihood(
            _toy_counts(), self.design, balance_classes=True
        )
        self.assertEqual(int(aggregate.rows["population_count"].sum()), 480)
        self.assertEqual(aggregate.matrix.shape[1], 4371)
        weight_sums = aggregate.rows.groupby("target")["sample_weight"].sum()
        self.assertAlmostEqual(weight_sums.loc[0], weight_sums.loc[1])
        self.assertEqual(int(aggregate.target.sum()), 6)

    def test_fold_assignment_keeps_contigs_whole_and_is_deterministic(self):
        first = make_balanced_contig_folds(_toy_counts(), n_splits=3)
        second = make_balanced_contig_folds(_toy_counts(), n_splits=3)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(set(first["contig"]), {f"c{i}" for i in range(1, 7)})
        self.assertFalse(first["contig"].duplicated().any())
        self.assertEqual(set(first["fold"]), {0, 1, 2})

    def test_regularization_selection_prefers_stronger_near_best_model(self):
        summary = pd.DataFrame(
            {
                "C": [0.01, 0.1, 1.0],
                "mean_average_precision": [0.0992, 0.1000, 0.1001],
                "mean_epic_spearman": [0.21, 0.20, 0.205],
                "all_folds_converged": [True, True, True],
            }
        )
        selected = select_regularization(summary, relative_ap_tolerance=0.01)
        self.assertEqual(selected["C"], 0.01)

    def test_config_rejects_non_champion_window(self):
        with self.assertRaises(ValueError):
            HierarchicalKmerLogisticConfig(ks=(2, 4, 8))


if __name__ == "__main__":
    unittest.main()
