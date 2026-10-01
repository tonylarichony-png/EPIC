"""Contract tests for the EXP-010 regional dinucleotide screen."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.epic_experiments.exp_010_regionalnye_dinukleotidy_v_okne_201_bp import (
    DINUCLEOTIDE_PAIRS,
    FEATURE_READY,
    FIXED_C,
    KmerDinucleotideCounts,
    RegionalDinucleotideConfig,
    _count_oriented_ranges,
    aggregate_candidate,
    dinucleotide_bin_codes,
    dinucleotide_feature_table,
    dinucleotide_observed_expected,
    feature_contract,
)
from ml_project.epic_kmer_logistic import hierarchical_kmer_design


class TestExp010RegionalDinucleotides(unittest.TestCase):
    def setUp(self):
        self.config = RegionalDinucleotideConfig(count_batch_size=4)

    def test_protocol_is_frozen(self):
        self.assertTrue(FEATURE_READY)
        self.assertEqual(len(DINUCLEOTIDE_PAIRS), 15)
        self.assertNotIn("CG", DINUCLEOTIDE_PAIRS)
        self.assertEqual(FIXED_C, 1e-4)
        contract = feature_contract()
        self.assertEqual(contract["planned_fits"], 45)
        self.assertEqual(contract["total_model_columns"], 4393)
        self.assertEqual(contract["accumulation"], "forbidden")

    def test_observed_expected_and_missing(self):
        actual = dinucleotide_observed_expected(
            np.array([20, 0]), np.array([10, 10]),
            np.array([4, 0]), np.array([100, 100]))
        self.assertAlmostEqual(actual[0], 2.0)
        self.assertTrue(np.isnan(actual[1]))
        codes = dinucleotide_bin_codes([20, 0], [10, 10], [4, 0], [100, 100])
        self.assertEqual(int(codes[1]), 7)

    def test_each_pair_has_seven_reference_coded_columns(self):
        for pair in DINUCLEOTIDE_PAIRS:
            with self.subTest(pair=pair):
                table = dinucleotide_feature_table(pair)
                self.assertEqual(len(table), 7)
                self.assertNotIn(3, set(table["bin_code"]))
                self.assertTrue(table["feature_name"].str.startswith(pair).all())

    def test_streaming_counter_preserves_every_position(self):
        letters = np.array([0, 0, 3, 3, 0, 1, 2, 3, 4], dtype=np.uint8)
        for pair in ("AA", "TT", "TA", "GC"):
            with self.subTest(pair=pair):
                counted = _count_oriented_ranges(
                    letters, [(0, len(letters))], pair, self.config)
                self.assertEqual(int(counted["total_positions"].sum()), len(letters))
                self.assertTrue(counted["dinucleotide_bin_code"].between(0, 7).all())

    def test_exact_aggregate_adds_only_one_pair_feature(self):
        totals = pd.DataFrame({
            "contig": ["c1", "c1", "c2"], "strand": ["+", "-", "+"],
            "kmer_6_code": [0, 1, 2], "gc_bin_code": [2, 3, 4],
            "cpg_bin_code": [3, 4, 2], "dinucleotide_bin_code": [3, 5, 1],
            "total_positions": [100, 80, 60],
        })
        positives = pd.DataFrame({
            "contig": ["c1", "c1", "c2"], "strand": ["+", "-", "+"],
            "kmer_6_code": [0, 1, 2], "gc_bin_code": [2, 3, 4],
            "cpg_bin_code": [3, 4, 2], "dinucleotide_bin_code": [3, 5, 1],
            "count": [1, 2, 1],
        })
        counts = KmerDinucleotideCounts("train", "AA", -100, 100, totals, positives)
        aggregate = aggregate_candidate(
            counts, config=self.config, design=hierarchical_kmer_design(self.config))
        self.assertEqual(aggregate.matrix.shape[1], 4393)
        self.assertEqual(int(aggregate.rows["population_count"].sum()), 240)
        weights = aggregate.rows.groupby("target")["sample_weight"].sum()
        self.assertAlmostEqual(weights.loc[0], weights.loc[1])


if __name__ == "__main__":
    unittest.main()
