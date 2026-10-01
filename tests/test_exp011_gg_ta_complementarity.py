"""Stage-one contracts for EXP-011 GG+TA complementarity."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.epic_experiments.exp_011_gg_ta_complementarity import (
    CV_READY,
    FEATURE_READY,
    GGTAConfig,
    GGTAJointCounts,
    VariantSummary,
    _count_oriented_ranges,
    _decode_variant_keys,
    _fit_streaming_summary,
    _parameter_layout,
    _score_codes,
    count_candidate,
    count_diagnostics,
    feature_contract,
    cross_validate_candidate,
    summarize_cv_gate,
)
from ml_project.epic_kmer_logistic import hierarchical_kmer_design
from ml_project.epic_gc_binned_logistic import _gc_bin_matrix
from ml_project.epic_cpg_logistic import _cpg_bin_matrix
from ml_project.epic_experiments.exp_010_regionalnye_dinukleotidy_v_okne_201_bp import (
    _dinucleotide_matrix,
)


class TestExp011GgTaComplementarity(unittest.TestCase):
    def test_contract_is_additive_and_frozen(self):
        contract = feature_contract()
        self.assertTrue(FEATURE_READY)
        self.assertTrue(CV_READY)
        self.assertEqual(contract["pairs"], ("GG", "TA"))
        self.assertEqual(contract["combined_columns"], 4400)
        self.assertEqual(contract["interaction_columns"], 0)
        self.assertEqual(contract["new_cv_fits"], 9)

    def test_joint_streaming_counter_preserves_population(self):
        letters = np.array([2, 2, 3, 0, 3, 0, 1, 2, 4], dtype=np.uint8)
        config = GGTAConfig(count_batch_size=3)
        result = _count_oriented_ranges(
            letters, [(0, len(letters))], config
        )
        self.assertEqual(int(result["total_positions"].sum()), len(letters))
        self.assertTrue(result["gg_bin_code"].between(0, 7).all())
        self.assertTrue(result["ta_bin_code"].between(0, 7).all())

    def test_fold_level_diagnostics_preserve_totals(self):
        columns = {
            "kmer_6_code": [0], "gc_bin_code": [1], "cpg_bin_code": [2],
            "gg_bin_code": [3], "ta_bin_code": [4],
        }
        fold_zero = pd.DataFrame(columns | {"total_positions": [100]})
        fold_one = pd.DataFrame(columns | {"total_positions": [80]})
        positives = pd.DataFrame({"group": [0, 1, 1]})
        counts = GGTAJointCounts(
            "train", "fold", {0: fold_zero, 1: fold_one},
            pd.DataFrame({"contig": ["a", "b"], "fold": [0, 1]}), positives,
        )
        diagnostics = count_diagnostics(counts)
        total = diagnostics.loc[diagnostics["group"] == "TOTAL"].iloc[0]
        self.assertEqual(int(total["positions"]), 180)
        self.assertEqual(int(total["joint_rows"]), 2)
        self.assertEqual(int(total["positive_rows"]), 3)

    def test_train_count_streams_directly_into_frozen_folds(self):
        class TinySplit:
            def __init__(self):
                self._intervals = pd.DataFrame({
                    "contig": ["a", "b", "c", "d"],
                    "strand": ["+", "+", "+", "+"],
                    "start": [0, 0, 0, 0],
                    "end": [12, 12, 12, 12],
                })
                self._positives = pd.DataFrame({
                    "template_index": [0, 1, 2, 3],
                    "contig": ["a", "b", "c", "d"],
                    "coordinate_0based": [6, 6, 6, 6],
                    "strand": ["+", "+", "+", "+"],
                    "count": [1, 2, 3, 4],
                })

            def intervals(self, split_name):
                self.assert_name(split_name)
                return self._intervals

            def positive_targets(self, split_name):
                self.assert_name(split_name)
                return self._positives

            @staticmethod
            def assert_name(split_name):
                if split_name != "train":
                    raise AssertionError(split_name)

        sequences = {
            name: np.tile(np.arange(4, dtype=np.uint8), 3)
            for name in ("a", "b", "c", "d")
        }
        folds = pd.DataFrame({
            "contig": ["a", "b", "c", "d"], "fold": [0, 0, 1, 2],
        })
        counts = count_candidate(
            TinySplit(), sequences, "train", fold_assignment=folds,
            config=GGTAConfig(count_batch_size=5),
        )
        self.assertEqual(counts.group_kind, "fold")
        self.assertEqual(counts.represented_positions, 48)
        self.assertEqual(set(counts.totals_by_group), {0, 1, 2})
        self.assertEqual(len(counts.positives), 4)

    def test_streaming_optimizer_matches_sklearn_weighted_likelihood(self):
        config = GGTAConfig(max_iter=1000, tol=1e-11)
        design = hierarchical_kmer_design(config)
        kmer = np.arange(6, dtype=np.int64)
        gc = np.array([0, 1, 2, 3, 4, 8], dtype=np.int64)
        cpg = np.array([0, 1, 2, 3, 4, 7], dtype=np.int64)
        gg = np.array([0, 1, 2, 3, 4, 7], dtype=np.int64)
        keys = ((kmer * 9 + gc) * 8 + cpg) * 8 + gg
        totals = np.array([100, 120, 90, 130, 110, 140], dtype=np.int64)
        positives = np.array([1, 3, 1, 5, 2, 7], dtype=np.int64)
        summary = VariantSummary("EXP-009+GG", keys, totals, positives)
        streamed = _fit_streaming_summary(
            summary, config=config, design=design, batch_size=2,
        )
        self.assertEqual(len(streamed.coefficient_table), 4393)
        self.assertEqual(streamed.coefficient_table["feature_index"].nunique(), 4393)

        matrix = sparse.hstack([
            design.matrix[kmer], _gc_bin_matrix(gc, config),
            _cpg_bin_matrix(cpg, config), _dinucleotide_matrix(gg, "GG"),
        ], format="csr")
        row_index = np.repeat(np.arange(len(keys)), 2)
        target = np.tile(np.array([0, 1], dtype=np.int8), len(keys))
        population_count = np.column_stack((totals - positives, positives)).reshape(-1)
        population = totals.sum()
        multipliers = np.array([
            population / (2 * (population - positives.sum())),
            population / (2 * positives.sum()),
        ])
        sklearn_fit = LogisticRegression(
            C=config.candidate_C, solver="lbfgs", penalty="l2",
            max_iter=config.max_iter, tol=config.tol,
        ).fit(matrix[row_index], target, sample_weight=population_count * multipliers[target])
        layout = _parameter_layout(design, summary.variant, config)
        streamed_scores = _score_codes(
            streamed.coefficients, streamed.intercept, design, layout,
            _decode_variant_keys(keys, summary.variant), config,
        )
        np.testing.assert_allclose(
            streamed_scores, sklearn_fit.decision_function(matrix), atol=2e-5, rtol=2e-5,
        )

    def test_preregistered_cv_gate_accepts_clear_complementarity(self):
        rows = []
        values = {
            "EXP-009": ([1.00, 1.10, 0.90], [0.10, 0.11, 0.09]),
            "EXP-009+GG": ([1.10, 1.20, 1.00], [0.11, 0.12, 0.10]),
            "EXP-009+TA": ([1.05, 1.15, 0.95], [0.105, 0.115, 0.095]),
            "EXP-009+GG+TA": ([1.13, 1.23, 1.03], [0.12, 0.13, 0.11]),
        }
        for variant, (aps, correlations) in values.items():
            for fold, (ap, correlation) in enumerate(zip(aps, correlations)):
                rows.append({
                    "variant": variant, "fold": fold,
                    "average_precision": ap, "epic_spearman": correlation,
                    "converged": True, "fit_seconds": 1.0,
                })
        gate = summarize_cv_gate(pd.DataFrame(rows))
        self.assertTrue(gate.passed)
        self.assertEqual(gate.best_single_variant, "EXP-009+GG")
        self.assertTrue(gate.checks.all())


if __name__ == "__main__":
    unittest.main()
