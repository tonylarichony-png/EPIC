"""Focused tests for the exact aggregated EPIC dinucleotide baseline."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.epic_baseline import (
    DinucleotideBaselineConfig,
    _average_precision_from_categories,
    aggregate_training_data,
    category_summary,
    evaluate_dinucleotide_baseline,
    fit_dinucleotide_baseline,
    split_category_counts,
)
from ml_project.epic_data import EpicPaths, PreparedSplit, SplitConfig
from ml_project.sequence_context import encode_fasta_sequence
from ml_project.sequence_features import DINUCLEOTIDE_LABELS


def _toy_split(root: Path) -> PreparedSplit:
    intervals = pd.DataFrame(
        {
            "contig": ["c1", "c1"],
            "start": [0, 0],
            "end": [5, 5],
            "strand": ["+", "-"],
        }
    )
    positives = pd.DataFrame(
        {
            "template_index": [0, 1, 2, 3],
            "contig": ["c1"] * 4,
            "coordinate_0based": [1, 2, 0, 3],
            "strand": ["+", "-", "+", "+"],
            "count": [2, 5, 1, 4],
        }
    )
    empty_intervals = intervals.iloc[0:0].copy()
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
            {"contig": ["c1"], "split": ["train"]}
        ),
        summary=pd.DataFrame(),
        intervals_by_split={
            "train": intervals,
            "validation": intervals.copy(),
            "test": empty_intervals,
        },
        positive_targets_by_split={
            "train": positives,
            "validation": positives.copy(),
            "test": None,
        },
        manifest={},
    )


def _complete_summary() -> pd.DataFrame:
    total = np.array([3 + code % 4 for code in range(17)], dtype=np.int64)
    positive = np.array([1 + code % 2 for code in range(17)], dtype=np.int64)
    return pd.DataFrame(
        {
            "category_code": np.arange(17, dtype=np.int8),
            "category": DINUCLEOTIDE_LABELS,
            "total_positions": total,
            "positive_positions": positive,
            "negative_positions": total - positive,
        }
    )


class TestEpicBaseline(unittest.TestCase):
    def test_split_counts_cover_every_position_and_orient_both_strands(self):
        with tempfile.TemporaryDirectory() as temporary:
            split = _toy_split(Path(temporary))
            totals, positives = split_category_counts(
                split,
                {"c1": encode_fasta_sequence("ACGTN")},
                "train",
            )

        by_code = totals.groupby("category_code")["total_positions"].sum()
        self.assertEqual(int(by_code.sum()), 10)
        self.assertEqual(int(by_code.loc[1]), 2)   # AC
        self.assertEqual(int(by_code.loc[6]), 2)   # CG
        self.assertEqual(int(by_code.loc[11]), 2)  # GT
        self.assertEqual(int(by_code.loc[16]), 4)  # contig edge or N
        self.assertEqual(len(totals), 2 * len(DINUCLEOTIDE_LABELS))
        self.assertEqual(positives["category"].tolist(), ["AC", "AC", "N/edge", "GT"])

        summary = category_summary(totals, positives)
        self.assertEqual(int(summary["total_positions"].sum()), 10)
        self.assertEqual(int(summary["positive_positions"].sum()), 4)
        self.assertTrue((summary["negative_positions"] >= 0).all())

    def test_aggregate_rows_preserve_counts_and_exact_weighted_likelihood(self):
        summary = _complete_summary()
        config = DinucleotideBaselineConfig()
        aggregate, encoded = aggregate_training_data(summary, config)

        population = int(summary["total_positions"].sum())
        positives = int(summary["positive_positions"].sum())
        negatives = population - positives
        self.assertEqual(len(aggregate), 34)
        self.assertEqual(int(aggregate["population_count"].sum()), population)
        self.assertEqual(
            int(aggregate.loc[aggregate["target"] == 1, "population_count"].sum()),
            positives,
        )
        self.assertEqual(
            int(aggregate.loc[aggregate["target"] == 0, "population_count"].sum()),
            negatives,
        )
        weighted_by_class = aggregate.groupby("target")["sample_weight"].sum()
        np.testing.assert_allclose(weighted_by_class.to_numpy(), population / 2)
        self.assertEqual(encoded.matrix.shape, (34, 16))

        category_score = np.linspace(-1.25, 1.75, len(DINUCLEOTIDE_LABELS))
        aggregate_margin = category_score[
            aggregate["category_code"].to_numpy(dtype=np.int8)
        ]
        aggregate_target = aggregate["target"].to_numpy(dtype=np.int8)
        aggregate_loss = np.sum(
            aggregate["sample_weight"].to_numpy()
            * np.logaddexp(0.0, (1 - 2 * aggregate_target) * aggregate_margin)
        )

        expanded_loss = 0.0
        multipliers = {
            0: population / (2 * negatives),
            1: population / (2 * positives),
        }
        for row in summary.itertuples(index=False):
            for target, count in (
                (0, int(row.negative_positions)),
                (1, int(row.positive_positions)),
            ):
                expanded_loss += count * multipliers[target] * np.logaddexp(
                    0.0,
                    (1 - 2 * target) * category_score[int(row.category_code)],
                )
        self.assertAlmostEqual(float(aggregate_loss), float(expanded_loss), places=12)

    def test_category_average_precision_matches_expanded_sklearn_population(self):
        total = np.array([4, 3, 5], dtype=np.int64)
        positive = np.array([1, 2, 2], dtype=np.int64)
        scores = np.array([0.2, 0.8, 0.2], dtype=np.float64)

        exact, curve = _average_precision_from_categories(total, positive, scores)
        expanded_target = np.concatenate(
            [
                np.r_[np.ones(pos, dtype=np.int8), np.zeros(size - pos, dtype=np.int8)]
                for size, pos in zip(total, positive)
            ]
        )
        expanded_score = np.concatenate(
            [np.full(size, score) for size, score in zip(total, scores)]
        )
        expected = average_precision_score(expanded_target, expanded_score)

        self.assertAlmostEqual(exact, expected, places=15)
        self.assertEqual(len(curve), 2)  # equal 0.2 scores stay one threshold
        self.assertEqual(int(curve.iloc[-1]["predicted_positive"]), int(total.sum()))
        self.assertEqual(int(curve.iloc[-1]["true_positive"]), int(positive.sum()))

    def test_fit_and_evaluate_smoke_on_synthetic_prepared_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            split = _toy_split(Path(temporary))
            sequences = {"c1": encode_fasta_sequence("ACGTN")}
            train_totals, train_positives = split_category_counts(
                split, sequences, "train"
            )
            validation_totals, validation_positives = split_category_counts(
                split, sequences, "validation"
            )

        config = DinucleotideBaselineConfig(C=0.1, max_iter=500, tol=1e-8, threads=1)
        fit = fit_dinucleotide_baseline(
            category_summary(train_totals, train_positives), config
        )
        evaluation = evaluate_dinucleotide_baseline(
            validation_totals,
            validation_positives,
            fit,
            config,
        )

        self.assertEqual(fit.estimator.coef_.shape, (1, 16))
        self.assertEqual(int(fit.fit_report.iloc[0]["represented_positions"]), 10)
        self.assertEqual(int(fit.fit_report.iloc[0]["represented_positives"]), 4)
        self.assertEqual(evaluation.metric_summary["metric"].tolist(), [
            "Average Precision",
            "EPIC Spearman",
        ])
        self.assertTrue(np.isfinite(evaluation.metric_summary["candidate"]).all())
        self.assertEqual(len(evaluation.positive_predictions), 4)
        self.assertEqual(len(evaluation.per_contig_metrics), 2)
        self.assertEqual(
            set(evaluation.per_contig_metrics["variant"]),
            {"constant_reference", "dinucleotide_logistic"},
        )


if __name__ == "__main__":
    unittest.main()
