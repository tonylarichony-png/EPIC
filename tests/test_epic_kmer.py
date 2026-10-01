"""Focused tests for exact whole-k-mer lookup experiments."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.epic_data import EpicPaths, PreparedSplit, SplitConfig
from ml_project.epic_kmer import (
    KmerLookupConfig,
    context_kmer_codes,
    evaluate_kmer_lookup,
    fit_kmer_lookup,
    kmer_category_count,
    kmer_category_label,
    kmer_category_summary,
    oriented_kmer_codes,
    split_kmer_counts,
)
from ml_project.sequence_context import encode_fasta_sequence, extract_oriented_context


KS = (2, 4, 6, 8)


def _toy_split(root: Path, split_name: str) -> PreparedSplit:
    length = 13
    intervals = pd.DataFrame(
        {
            "contig": ["c1", "c1"],
            "start": [0, 0],
            "end": [length, length],
            "strand": ["+", "-"],
        }
    )
    positives = pd.DataFrame(
        {
            "template_index": [1, 3, 7, 15, 19],
            "contig": ["c1"] * 5,
            "coordinate_0based": [1, 3, 7, 2, 6],
            "strand": ["+", "+", "+", "-", "-"],
            "count": [1, 5, 2, 7, 3],
        }
    )
    empty = intervals.iloc[0:0].copy()
    paths = EpicPaths(
        project_root=root,
        assembly="toy",
        source=root,
        participants=root,
        output=root,
        inputs={},
    )
    return PreparedSplit(
        config=SplitConfig(assembly="toy"),
        paths=paths,
        contig_assignment=pd.DataFrame(
            {"contig": ["c1"], "split": [split_name]}
        ),
        summary=pd.DataFrame(),
        intervals_by_split={
            "train": intervals if split_name == "train" else empty,
            "validation": intervals if split_name == "validation" else empty,
            "test": empty,
        },
        positive_targets_by_split={
            "train": positives if split_name == "train" else positives.iloc[0:0],
            "validation": positives if split_name == "validation" else positives.iloc[0:0],
            "test": None,
        },
        manifest={},
    )


class TestEpicKmer(unittest.TestCase):
    def setUp(self):
        self.raw = encode_fasta_sequence("ACGTACGTNACGT")
        self.sequences = {"c1": self.raw}

    def test_category_encoding_and_strand_orientation_match_shared_extractor(self):
        self.assertEqual(kmer_category_label(4, 27), "ACGT")
        self.assertEqual(kmer_category_label(4, 4**4), "N/edge")
        rows = pd.DataFrame(
            {
                "contig": ["c1"] * (2 * len(self.raw)),
                "coordinate_0based": np.r_[
                    np.arange(len(self.raw)), np.arange(len(self.raw))
                ],
                "strand": ["+"] * len(self.raw) + ["-"] * len(self.raw),
            }
        )
        context = extract_oriented_context(
            self.sequences, rows, offsets=range(-3, 1)
        )
        expected = context_kmer_codes(context)
        plus, minus = oriented_kmer_codes(self.raw, 4)
        np.testing.assert_array_equal(np.r_[plus, minus], expected)
        self.assertTrue(np.all(plus[:3] == 4**4))
        self.assertEqual(kmer_category_label(4, int(plus[3])), "ACGT")

    def test_exact_split_counts_cover_full_population_for_every_k(self):
        with tempfile.TemporaryDirectory() as temporary:
            split = _toy_split(Path(temporary), "train")
            counts = split_kmer_counts(
                split, self.sequences, "train", ks=KS, batch_size=3
            )

        self.assertEqual(counts.ks, KS)
        self.assertEqual(
            counts.totals.groupby("k")["total_positions"].sum().to_dict(),
            {k: 2 * len(self.raw) for k in KS},
        )
        self.assertEqual(len(counts.positives), 5)
        for k in KS:
            plus, minus = oriented_kmer_codes(self.raw, k)
            expected = np.bincount(
                np.r_[plus, minus], minlength=kmer_category_count(k)
            )
            actual = (
                counts.totals.loc[counts.totals["k"] == k]
                .groupby("category_code")["total_positions"]
                .sum()
                .reindex(range(kmer_category_count(k)), fill_value=0)
                .to_numpy()
            )
            np.testing.assert_array_equal(actual, expected)
            summary = kmer_category_summary(counts, k)
            self.assertEqual(int(summary["total_positions"].sum()), 2 * len(self.raw))
            self.assertEqual(int(summary["positive_positions"].sum()), 5)

    def test_preregistered_smoothing_uses_one_expected_positive_and_suffixes(self):
        with tempfile.TemporaryDirectory() as temporary:
            train = _toy_split(Path(temporary), "train")
            counts = split_kmer_counts(train, self.sequences, "train", ks=KS)
        fit = fit_kmer_lookup(counts, KmerLookupConfig(ks=KS))

        expected_strength = 1.0 / (5 / (2 * len(self.raw)))
        np.testing.assert_allclose(
            fit.smoothing_report["strength"].to_numpy(), expected_strength
        )
        self.assertEqual(fit.support_summary["k"].tolist(), list(KS))
        self.assertTrue(
            fit.smoothing_report["optimizer_message"]
            .str.contains("one expected positive")
            .all()
        )

        k2 = fit.category_summary.loc[fit.category_summary["k"] == 2]
        nedge2 = k2.loc[k2["category"] == "N/edge"].iloc[0]
        self.assertAlmostEqual(nedge2["prior_score"], fit.global_prevalence)

        k4 = fit.category_summary.loc[fit.category_summary["k"] == 4]
        acgt = k4.loc[k4["category"] == "ACGT"].iloc[0]
        gt = k2.loc[k2["category"] == "GT"].iloc[0]
        self.assertAlmostEqual(acgt["prior_score"], gt["posterior_score"])
        nedge4 = k4.loc[k4["category"] == "N/edge"].iloc[0]
        self.assertEqual(int(nedge4["parent_category_code"]), -1)
        self.assertAlmostEqual(nedge4["prior_score"], fit.global_prevalence)

        unseen = k4.loc[k4["total_positions"] == 0].iloc[0]
        self.assertAlmostEqual(unseen["posterior_score"], unseen["prior_score"])

    def test_validation_metrics_are_exact_and_scores_remain_unquantized(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_counts = split_kmer_counts(
                _toy_split(root, "train"), self.sequences, "train", ks=KS
            )
            validation_counts = split_kmer_counts(
                _toy_split(root, "validation"),
                self.sequences,
                "validation",
                ks=KS,
            )
        fit = fit_kmer_lookup(train_counts, KmerLookupConfig(ks=KS))
        evaluation = evaluate_kmer_lookup(validation_counts, fit)

        self.assertEqual(
            evaluation.metric_summary["variant"].tolist(),
            ["constant_reference", "2mer_lookup", "4mer_lookup", "6mer_lookup", "8mer_lookup"],
        )
        self.assertEqual(len(evaluation.per_contig_metrics), 5)
        self.assertEqual(len(evaluation.positive_predictions), 5)
        self.assertEqual(set(evaluation.precision_recall["k"]), set(KS))

        for k in KS:
            total = (
                validation_counts.totals.loc[validation_counts.totals["k"] == k]
                .groupby("category_code")["total_positions"]
                .sum()
                .reindex(range(kmer_category_count(k)), fill_value=0)
                .to_numpy(dtype=np.int64)
            )
            positive = (
                validation_counts.positives[f"kmer_{k}_code"]
                .value_counts()
                .reindex(range(kmer_category_count(k)), fill_value=0)
                .to_numpy(dtype=np.int64)
            )
            targets = np.concatenate(
                [
                    np.r_[np.ones(pos, dtype=np.int8), np.zeros(size - pos, dtype=np.int8)]
                    for size, pos in zip(total, positive)
                ]
            )
            scores = np.concatenate(
                [np.full(size, fit.scores_by_k[k][code]) for code, size in enumerate(total)]
            )
            expected_ap = average_precision_score(targets, scores)
            row = evaluation.metric_summary.loc[
                evaluation.metric_summary["k"] == k
            ].iloc[0]
            self.assertAlmostEqual(row["average_precision"], expected_ap, places=15)

            positive_scores = evaluation.positive_predictions[f"kmer_{k}_score"].to_numpy()
            left = rankdata(
                evaluation.positive_predictions["count"].to_numpy(), method="dense"
            )
            right = rankdata(positive_scores, method="dense")
            expected_spearman = (
                0.0
                if right.std() == 0
                else float(np.corrcoef(left, right)[0, 1])
            )
            self.assertAlmostEqual(row["epic_spearman"], expected_spearman, places=15)


if __name__ == "__main__":
    unittest.main()
