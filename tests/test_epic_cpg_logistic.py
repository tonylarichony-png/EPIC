"""Synthetic contract tests for the exact EXP-009 CpG O/E201 model."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.epic_cpg_logistic import (
    CPG_C_GRID,
    CpGLogisticConfig,
    KmerCpGCounts,
    _composition_prefixes,
    _count_oriented_ranges,
    _sparse_window_statistics,
    _window_statistics,
    aggregate_cpg_likelihood,
    cpg_bin_codes,
    cpg_enrichment_table,
    cpg_observed_expected,
)
import ml_project.epic_cpg_logistic as cpg_module
from ml_project.epic_kmer import _codes_from_state, _packed_suffix_state
from ml_project.epic_kmer_logistic import hierarchical_kmer_design
from ml_project.epic_gc_binned_logistic import gc_bin_codes
from ml_project.sequence_context import COMPLEMENT_CODE


def _toy_counts() -> KmerCpGCounts:
    totals = pd.DataFrame(
        {
            "contig": ["c1", "c1", "c2", "c2", "c2"],
            "strand": ["+", "-", "+", "-", "+"],
            "kmer_6_code": [0, 1, 0, 2, 3],
            "c_count": [20, 25, 30, 35, 0],
            "g_count": [20, 25, 35, 30, 0],
            "cpg_count": [4, 6, 12, 10, 0],
            "valid_count": [100, 100, 100, 100, 0],
            "gc_count": [40, 50, 65, 65, 0],
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
            "c_count": [20, 20, 25, 30, 30, 35, 0],
            "g_count": [20, 20, 25, 35, 35, 30, 0],
            "cpg_count": [4, 4, 6, 12, 12, 10, 0],
            "valid_count": [100, 100, 100, 100, 100, 100, 0],
            "gc_count": [40, 40, 50, 65, 65, 65, 0],
        }
    )
    positives["cpg_oe201"] = cpg_observed_expected(
        positives["c_count"],
        positives["g_count"],
        positives["cpg_count"],
        positives["valid_count"],
    )
    totals["gc_bin_code"] = gc_bin_codes(totals["gc_count"], totals["valid_count"])
    totals["cpg_bin_code"] = cpg_bin_codes(
        totals["c_count"], totals["g_count"], totals["cpg_count"], totals["valid_count"]
    )
    positives["gc_bin_code"] = gc_bin_codes(
        positives["gc_count"], positives["valid_count"]
    )
    positives["cpg_bin_code"] = cpg_bin_codes(
        positives["c_count"],
        positives["g_count"],
        positives["cpg_count"],
        positives["valid_count"],
    )
    return KmerCpGCounts("train", -100, 100, totals, positives)


class TestEpicCpGLogistic(unittest.TestCase):
    def setUp(self):
        self.config = CpGLogisticConfig()
        self.design = hierarchical_kmer_design(self.config)

    def test_formula_and_missing_policy(self):
        actual = cpg_observed_expected(
            np.array([20, 0, 10]),
            np.array([10, 20, 0]),
            np.array([4, 0, 0]),
            np.array([100, 100, 100]),
        )
        self.assertAlmostEqual(actual[0], 2.0)
        self.assertTrue(np.isnan(actual[1]))
        self.assertTrue(np.isnan(actual[2]))

    def test_cpg_statistic_is_reverse_complement_invariant(self):
        letters = np.array([0, 1, 2, 3, 1, 2, 0], dtype=np.uint8)
        reverse = COMPLEMENT_CODE[letters[::-1]]
        coordinate = np.array([3])
        forward_stats = _window_statistics(
            _composition_prefixes(letters),
            coordinate,
            sequence_length=len(letters),
            window_start=-3,
            window_end=3,
        )
        reverse_stats = _window_statistics(
            _composition_prefixes(reverse),
            coordinate,
            sequence_length=len(reverse),
            window_start=-3,
            window_end=3,
        )
        forward_oe = cpg_observed_expected(*forward_stats)[0]
        reverse_oe = cpg_observed_expected(*reverse_stats)[0]
        self.assertAlmostEqual(forward_oe, reverse_oe)
        self.assertEqual(int(forward_stats[2][0]), 2)

    def test_exact_counter_preserves_every_position(self):
        letters = np.array([0, 1, 2, 3, 1, 2, 0], dtype=np.uint8)
        counted = _count_oriented_ranges(letters, [(0, len(letters))], self.config)
        self.assertEqual(int(counted["total_positions"].sum()), len(letters))
        self.assertTrue((counted["gc_bin_code"] == 6).all())
        self.assertTrue((counted["cpg_bin_code"] == 6).all())

    def test_streaming_counter_matches_full_prefix_reference(self):
        rng = np.random.default_rng(42)
        letters = rng.integers(0, 5, size=1_000, dtype=np.uint8)
        config = CpGLogisticConfig(count_batch_size=25)
        actual = _count_oriented_ranges(letters, [(0, len(letters))], config)

        packed, unknown = _packed_suffix_state(letters, config.max_k)
        codes = _codes_from_state(
            packed, unknown, 0, len(letters), config.max_k
        )
        statistics = _window_statistics(
            _composition_prefixes(letters),
            np.arange(len(letters)),
            sequence_length=len(letters),
            window_start=config.window_start,
            window_end=config.window_end,
        )
        expected = pd.DataFrame(
            {
                "kmer_6_code": codes,
                "gc_bin_code": gc_bin_codes(
                    statistics[0] + statistics[1], statistics[3]
                ),
                "cpg_bin_code": cpg_bin_codes(*statistics),
                "total_positions": 1,
            }
        ).groupby(
            ["kmer_6_code", "gc_bin_code", "cpg_bin_code"],
            as_index=False,
        )["total_positions"].sum()
        columns = list(expected.columns)
        pd.testing.assert_frame_equal(
            actual.sort_values(columns[:-1]).reset_index(drop=True)[columns],
            expected.sort_values(columns[:-1]).reset_index(drop=True)[columns],
            check_dtype=False,
        )

    def test_streaming_counter_bounds_prefix_memory_to_one_chunk(self):
        letters = np.resize(np.array([0, 1, 2, 3], dtype=np.uint8), 1_000)
        config = CpGLogisticConfig(count_batch_size=25)
        observed_lengths: list[int] = []
        original = cpg_module._composition_prefixes

        def tracked(values):
            observed_lengths.append(len(values))
            return original(values)

        with patch.object(cpg_module, "_composition_prefixes", side_effect=tracked):
            _count_oriented_ranges(letters, [(0, len(letters))], config)
        self.assertTrue(observed_lengths)
        self.assertLess(max(observed_lengths), len(letters))
        self.assertLessEqual(
            max(observed_lengths), config.count_batch_size + config.window_size - 1
        )

    def test_sparse_positive_statistics_match_full_prefix_reference(self):
        rng = np.random.default_rng(7)
        letters = rng.integers(0, 5, size=500, dtype=np.uint8)
        coordinates = np.array([0, 1, 99, 100, 250, 498, 499])
        expected = _window_statistics(
            _composition_prefixes(letters),
            coordinates,
            sequence_length=len(letters),
            window_start=self.config.window_start,
            window_end=self.config.window_end,
        )
        actual = _sparse_window_statistics(
            letters, coordinates, self.config, batch_size=3
        )
        for expected_values, actual_values in zip(expected, actual):
            np.testing.assert_array_equal(actual_values, expected_values)

    def test_exact_aggregate_adds_seven_bin_columns_to_exp007(self):
        aggregate = aggregate_cpg_likelihood(
            _toy_counts(), self.design, config=self.config
        )
        self.assertEqual(aggregate.matrix.shape[1], 4386)
        self.assertEqual(int(aggregate.rows["population_count"].sum()), 300)
        weights = aggregate.rows.groupby("target")["sample_weight"].sum()
        self.assertAlmostEqual(weights.loc[0], weights.loc[1])

    def test_train_diagnostic_preserves_population(self):
        table = cpg_enrichment_table(_toy_counts())
        self.assertEqual(int(table["positions"].sum()), 300)
        self.assertEqual(int(table["positives"].sum()), 7)

    def test_grid_is_five_values_and_fifteen_fits(self):
        self.assertEqual(self.config.C_grid, CPG_C_GRID)
        self.assertEqual(len(self.config.C_grid) * self.config.cv_folds, 15)
        self.assertIn(3e-4, self.config.C_grid)

    def test_grid_cannot_be_changed(self):
        with self.assertRaises(ValueError):
            CpGLogisticConfig(C_grid=(1e-4, 3e-4))


if __name__ == "__main__":
    unittest.main()
