"""Synthetic checks for leakage controls, weighting and interpretable coding."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile

import numpy as np
from sklearn.metrics import average_precision_score

SPEC = importlib.util.spec_from_file_location("logistic_eda", Path(__file__).parents[1]/"scripts/eda_logistic_context.py")
eda = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eda)


class TestLogisticContext(unittest.TestCase):
    def setUp(self):
        self.seqs = {"a": eda.eda.LUT[np.frombuffer(b"ACGTNacgt", dtype=np.uint8)],
                     "b": eda.eda.LUT[np.frombuffer(b"TTCACGTAA", dtype=np.uint8)],
                     "c": eda.eda.LUT[np.frombuffer(b"AACGTATGT", dtype=np.uint8)]}
        self.w = eda.eda.Whitelist([(name, 0, 9) for name in self.seqs], self.seqs)

    def test_reverse_complement_orients_about_coordinate(self):
        context = eda.eda.oriented_context(self.seqs, (np.array(["a", "a"]), np.array([2, 2]),
                                                     np.array([False, True])), [-1, 0, 1])
        np.testing.assert_array_equal(context, [[1, 2, 3], [0, 1, 2]])

    def test_edge_missing_composition_denominators(self):
        decoded = (np.array(["a", "a"]), np.array([0, 4]), np.array([False, True]))
        context = eda.eda.oriented_context(self.seqs, decoded, [-1, 0, 1])
        np.testing.assert_array_equal(context[0], [4, 0, 1])
        values = eda.composition_features(self.seqs, decoded, 3)
        self.assertAlmostEqual(values.loc[0, "gc_acgt_3"], .5)
        self.assertAlmostEqual(values.loc[0, "edge_fraction_3"], 1/3)
        self.assertAlmostEqual(values.loc[1, "unknown_fraction_3"], 1/3)
        self.assertAlmostEqual(values.loc[1, "lower_fraction_3"], 1/3)
        missing = eda.composition_features({"z": np.array([8], dtype=np.uint8)},
            (np.array(["z"]), np.array([0]), np.array([False])), 1)
        self.assertEqual(missing.loc[0, "gc_missing_1"], 1)
        self.assertEqual(missing.loc[0, "gc_acgt_1"], 0)

    def test_weighted_ap_matches_integer_expanded_population(self):
        y = np.array([1, 0, 1, 0])
        scores = np.array([.8, .7, .4, .1])
        weights = np.array([2, 20, 3, 40])
        weighted = average_precision_score(y, scores, sample_weight=weights)
        expanded = average_precision_score(np.repeat(y, weights), np.repeat(scores, weights))
        self.assertAlmostEqual(weighted, expanded)
        self.assertAlmostEqual(average_precision_score(y, scores, sample_weight=np.ones(4)),
                               average_precision_score(y, scores))
        self.assertAlmostEqual(average_precision_score(y, np.ones(4), sample_weight=weights),
                               np.average(y, weights=weights))

    def test_sampling_is_unique_grouped_reproducible_and_population_weighted(self):
        positive = np.array([0, 2, 9, 13, 18, 23])
        counts = np.arange(1, 7)
        lengths = {name: len(seq) for name, seq in self.seqs.items()}
        rows, contigs = eda.sample_positions(self.w, positive, counts, lengths, 42, 2, 5)
        again, _ = eda.sample_positions(self.w, positive, counts, lengths, 42, 2, 5)
        self.assertTrue(rows.equals(again))
        self.assertEqual(rows.template_index.nunique(), len(rows))
        self.assertEqual(rows.groupby("contig").fold.nunique().max(), 1)
        self.assertAlmostEqual(rows.population_weight.sum(), 2*self.w.total)
        self.assertAlmostEqual(np.average(rows.target, weights=rows.population_weight), len(positive)/(2*self.w.total))
        self.assertEqual(set(rows[rows.target == 0].template_index) & set(positive), set())
        for fold in range(3):
            self.assertFalse(set(rows[rows.fold == fold].contig) & set(rows[rows.fold != fold].contig))

    def test_fold_assignment_uses_lengths_only(self):
        self.assertEqual(eda.length_folds({"b": 9, "a": 9, "c": 5, "d": 2}),
                         {"a": 0, "b": 1, "c": 2, "d": 0})

    def test_reference_feature_and_interaction_coding(self):
        X, names = eda.reference_features(np.array([[3, 3], [0, 4], [1, 2]]), [-1, 0])
        self.assertEqual(X.shape, (3, 8))
        self.assertEqual(X[0].nnz, 0)
        self.assertEqual(names, ["base[-1]=A", "base[-1]=C", "base[-1]=G", "base[-1]=N/edge",
                                 "base[+0]=A", "base[+0]=C", "base[+0]=G", "base[+0]=N/edge"])
        np.testing.assert_array_equal(X.toarray()[1], [1, 0, 0, 0, 0, 0, 0, 1])
        baseline, names = eda.pair_baseline(np.array([[3, 3], [1, 0], [4, 0]]))
        self.assertEqual(baseline[0].nnz, 0)
        self.assertEqual(names[int(baseline[1].indices[0])], "pair[-1,0]=CA")
        self.assertEqual(names[int(baseline[2].indices[0])], "pair[-1,0]=N/edge")
        interactions, _ = eda.interaction_features(np.full((2, 11), 3))
        self.assertEqual(interactions.nnz, 0)

    def test_balancing_keeps_mean_one_and_equal_class_weight(self):
        y = np.array([1, 1, 0, 0, 0])
        w = eda.training_weights(y, np.array([10, 15, 100, 200, 400]))
        self.assertAlmostEqual(w.mean(), 1)
        self.assertAlmostEqual(w[y == 1].sum(), w[y == 0].sum())

    def test_dense_rank_metric_is_not_ordinary_spearman_with_ties(self):
        counts = np.array([0, 1, 1, 2, 4])
        scores = np.array([100, 4, 3, 2, 1])
        expected = np.corrcoef([1, 1, 2, 3], [4, 3, 2, 1])[0, 1]
        self.assertAlmostEqual(eda.dense_rank_corr(counts, scores), expected)

    def test_duplicate_audit_marks_only_cross_fold_equal_windows(self):
        import pandas as pd
        context = np.array([[0, 1], [0, 1], [2, 3], [2, 3], [3, 3]])
        report, mask = eda.duplicate_context_audit(context, pd.DataFrame({"fold": [0, 1, 2, 2, 1]}))
        self.assertEqual(report["cross_fold_duplicate_groups"], 1)
        np.testing.assert_array_equal(mask, [True, True, False, False, False])

    def test_grouped_fit_outputs_and_integrity_checked_cache(self):
        # Two candidates exercise sparse-only and train-standardized dense
        # composition paths without making the synthetic test expensive.
        positive = np.array([0, 2, 9, 13, 18, 23])
        counts = np.arange(1, 7)
        lengths = {name: len(seq) for name, seq in self.seqs.items()}
        rows, contigs = eda.sample_positions(self.w, positive, counts, lengths, 42, 2, 5)
        decoded = self.w.decode(rows.template_index.to_numpy())
        state = {"rows": rows, "contigs": contigs, "seqs": self.seqs, "decoded": decoded,
            "context": eda.eda.oriented_context(self.seqs, decoded, np.arange(-400, 401)),
            "seed": 42, "provenance": {"synthetic": True}, "composition": {}}
        selected = [eda.candidates()[0], eda.candidates()[8]]
        with tempfile.TemporaryDirectory() as temporary, patch.object(eda, "candidates", return_value=selected):
            result = eda.run_experiments(state, Path(temporary))
            self.assertEqual(len(result["scores"]), 6)
            self.assertTrue(result["scores"].converged.all())
            self.assertFalse(result["recommendation"]["champion_selected"])
            self.assertEqual(len(result["oof"]), len(rows))
            with patch.object(eda.LogisticRegression, "fit", side_effect=AssertionError("Must use verified cache")):
                cached = eda.run_experiments(state, Path(temporary))
            np.testing.assert_array_equal(cached["oof"].dinuc_baseline, result["oof"].dinuc_baseline)


if __name__ == "__main__":
    unittest.main()
