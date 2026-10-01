"""Non-production-data tests for the EXP-006 GC201 machinery."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.epic_gc_logistic import (
    GCWindowLogisticConfig,
    KmerGCCounts,
    _average_precision_from_rows,
    _count_oriented_ranges,
    aggregate_gc_binary_likelihood,
    gc_count_diagnostics,
    kmer_counts_from_gc_counts,
)
from ml_project.epic_kmer import _uppercase_letters, oriented_kmer_codes
from ml_project.epic_kmer_logistic import hierarchical_kmer_design
from ml_project.sequence_composition import gc_window_counts
from ml_project.sequence_context import encode_fasta_sequence


def _toy_counts() -> KmerGCCounts:
    totals = pd.DataFrame(
        {
            "contig": ["c1", "c1", "c2", "c2"],
            "strand": ["+", "-", "+", "-"],
            "kmer_6_code": [0, 1, 0, 2],
            "gc_count": [80, 120, 80, 40],
            "valid_count": [201, 201, 201, 200],
            "total_positions": [100, 80, 60, 40],
        }
    )
    positives = pd.DataFrame(
        {
            "template_index": range(6),
            "contig": ["c1", "c1", "c1", "c2", "c2", "c2"],
            "coordinate_0based": range(6),
            "strand": ["+", "+", "-", "+", "+", "-"],
            "count": [1, 2, 1, 3, 1, 4],
            "kmer_6_code": [0, 0, 1, 0, 0, 2],
            "gc_count": [80, 80, 120, 80, 80, 40],
            "valid_count": [201, 201, 201, 201, 201, 200],
            "gc_fraction": [80 / 201, 80 / 201, 120 / 201, 80 / 201, 80 / 201, 0.2],
        }
    )
    return KmerGCCounts("train", -100, 100, totals, positives)


class TestEpicGCLogistic(unittest.TestCase):
    def setUp(self):
        self.config = GCWindowLogisticConfig(C_grid=(0.001, 0.01))
        self.design = hierarchical_kmer_design(self.config)

    def test_exact_aggregate_preserves_population_and_adds_one_feature(self):
        aggregate = aggregate_gc_binary_likelihood(_toy_counts(), self.design)
        self.assertEqual(int(aggregate.rows["population_count"].sum()), 280)
        self.assertEqual(aggregate.matrix.shape[1], 4372)
        self.assertTrue(np.isfinite(aggregate.gc_mean))
        self.assertGreater(aggregate.gc_std, 0)
        weights = aggregate.rows.groupby("target")["sample_weight"].sum()
        self.assertAlmostEqual(weights.loc[0], weights.loc[1])

    def test_batched_joint_counter_matches_direct_coordinate_features(self):
        raw = encode_fasta_sequence("ACGTNACGT")
        config = GCWindowLogisticConfig(
            C_grid=(0.001, 0.01), count_batch_size=4
        )
        ranges = [(0, 3), (5, 9)]
        actual = _count_oriented_ranges(_uppercase_letters(raw), ranges, config)
        coordinates = np.array([0, 1, 2, 5, 6, 7, 8])
        codes = oriented_kmer_codes(raw, 6)[0][coordinates]
        gc_count, valid_count = gc_window_counts(
            raw,
            coordinates,
            window_start=-100,
            window_end=100,
        )
        expected = pd.DataFrame(
            {
                "kmer_6_code": codes,
                "gc_count": gc_count,
                "valid_count": valid_count,
            }
        )
        expected = (
            expected.value_counts(sort=False)
            .rename("total_positions")
            .reset_index()
        )
        columns = ["kmer_6_code", "gc_count", "valid_count", "total_positions"]
        pd.testing.assert_frame_equal(
            actual[columns].sort_values(columns[:-1]).reset_index(drop=True),
            expected[columns].sort_values(columns[:-1]).reset_index(drop=True),
            check_dtype=False,
        )

    def test_kmer_marginals_are_recovered_without_rescanning(self):
        counts = kmer_counts_from_gc_counts(_toy_counts())
        by_k = counts.totals.groupby("k")["total_positions"].sum()
        self.assertEqual(set(by_k.index), {2, 4, 6})
        self.assertTrue((by_k == 280).all())
        self.assertIn("kmer_2_code", counts.positives)
        self.assertIn("kmer_4_code", counts.positives)
        self.assertIn("kmer_6_code", counts.positives)

    def test_grouped_average_precision_matches_expanded_rows(self):
        total = np.array([4, 3, 2])
        positive = np.array([1, 2, 1])
        scores = np.array([0.2, 0.8, 0.2])
        exact, _ = _average_precision_from_rows(
            total, positive, scores, include_curve=True
        )
        expanded_y: list[int] = []
        expanded_score: list[float] = []
        for n, y, score in zip(total, positive, scores):
            expanded_y.extend([1] * int(y) + [0] * int(n - y))
            expanded_score.extend([float(score)] * int(n))
        expected = average_precision_score(expanded_y, expanded_score)
        self.assertAlmostEqual(exact, expected)

    def test_diagnostics_report_exact_counts(self):
        report = gc_count_diagnostics(_toy_counts()).iloc[0]
        self.assertEqual(int(report["positions"]), 280)
        self.assertEqual(int(report["positives"]), 6)
        self.assertEqual(int(report["zero_valid_positions"]), 0)

    def test_config_rejects_a_different_window(self):
        with self.assertRaises(ValueError):
            GCWindowLogisticConfig(window_start=-50, window_end=50)


if __name__ == "__main__":
    unittest.main()
