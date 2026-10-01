"""Small synthetic tests for the unexecuted EXP-007 machinery."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.epic_gc_binned_logistic import (
    GC_BIN_LABELS,
    GCBinnedLogisticConfig,
    aggregate_binned_gc_binary_likelihood,
    gc_bin_codes,
    gc_bin_enrichment_table,
    gc_bin_feature_table,
)
from ml_project.epic_gc_logistic import KmerGCCounts
from ml_project.epic_kmer_logistic import hierarchical_kmer_design


def _toy_counts() -> KmerGCCounts:
    totals = pd.DataFrame(
        {
            "contig": ["c1", "c1", "c2", "c2", "c2"],
            "strand": ["+", "-", "+", "-", "+"],
            "kmer_6_code": [0, 1, 0, 2, 3],
            "gc_count": [70, 80, 95, 125, 0],
            "valid_count": [201, 201, 201, 201, 0],
            "total_positions": [100, 80, 60, 40, 20],
        }
    )
    positives = pd.DataFrame(
        {
            "template_index": range(7),
            "contig": ["c1", "c1", "c1", "c2", "c2", "c2", "c2"],
            "coordinate_0based": range(7),
            "strand": ["+", "+", "-", "+", "+", "-", "+"],
            "count": [1, 2, 1, 3, 1, 4, 1],
            "kmer_6_code": [0, 0, 1, 0, 0, 2, 3],
            "gc_count": [70, 70, 80, 95, 95, 125, 0],
            "valid_count": [201, 201, 201, 201, 201, 201, 0],
            "gc_fraction": [
                70 / 201,
                70 / 201,
                80 / 201,
                95 / 201,
                95 / 201,
                125 / 201,
                np.nan,
            ],
        }
    )
    return KmerGCCounts("train", -100, 100, totals, positives)


class TestEpicGCBinnedLogistic(unittest.TestCase):
    def setUp(self):
        self.config = GCBinnedLogisticConfig(C_grid=(0.001, 0.01))
        self.design = hierarchical_kmer_design(self.config)

    def test_boundaries_and_missing_are_fixed(self):
        gc = np.array([29, 30, 34, 35, 39, 40, 44, 45, 49, 50, 54, 55, 59, 60, 0])
        valid = np.array([100] * 14 + [0])
        actual = gc_bin_codes(gc, valid)
        expected = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8])
        np.testing.assert_array_equal(actual, expected)

    def test_feature_vocabulary_has_one_omitted_reference(self):
        table = gc_bin_feature_table(self.config)
        self.assertEqual(len(GC_BIN_LABELS), 9)
        self.assertEqual(len(table), 8)
        self.assertNotIn("[0.35,0.40)", set(table["bin_label"]))

    def test_exact_aggregate_preserves_population_and_replaces_linear_gc(self):
        aggregate = aggregate_binned_gc_binary_likelihood(
            _toy_counts(), self.design, config=self.config
        )
        self.assertEqual(int(aggregate.rows["population_count"].sum()), 300)
        self.assertEqual(aggregate.matrix.shape[1], 4379)
        weights = aggregate.rows.groupby("target")["sample_weight"].sum()
        self.assertAlmostEqual(weights.loc[0], weights.loc[1])
        gc_part = aggregate.matrix[:, 4371:]
        reference = aggregate.rows["gc_bin_code"].to_numpy() == 2
        self.assertTrue((np.asarray(gc_part[reference].sum(axis=1)).ravel() == 0).all())
        self.assertTrue((np.asarray(gc_part[~reference].sum(axis=1)).ravel() == 1).all())

    def test_enrichment_uses_exact_population(self):
        table = gc_bin_enrichment_table(_toy_counts())
        self.assertEqual(int(table["positions"].sum()), 300)
        self.assertEqual(int(table["positives"].sum()), 7)
        self.assertEqual(table["gc_bin"].tolist(), list(GC_BIN_LABELS))

    def test_config_rejects_changed_boundaries(self):
        with self.assertRaises(ValueError):
            GCBinnedLogisticConfig(bin_edges=(0.25, 0.50, 0.75))


if __name__ == "__main__":
    unittest.main()
