"""Exact full-population implementation of the first EPIC baseline.

The model has only 17 possible inputs: 16 canonical dinucleotides and one
N/edge category. Exact counts therefore reduce the full binary likelihood to
34 weighted rows (17 categories x 2 target classes) without sampling noise.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
import warnings
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .epic_data import PreparedSplit
from .sequence_context import (
    COMPLEMENT_CODE,
    UPPERCASE_BASE_CODE,
    extract_oriented_context,
    read_fasta,
)
from .sequence_features import (
    DINUCLEOTIDE_ENCODING_VERSION,
    DINUCLEOTIDE_LABELS,
    EncodedSequenceFeatures,
    dinucleotide_category_codes,
    encode_dinucleotide,
)


@dataclass(frozen=True)
class DinucleotideBaselineConfig:
    """Every setting capable of changing the baseline result."""

    seed: int = 42
    offsets: tuple[int, int] = (-1, 0)
    reference_pair: str = "TT"
    penalty: str = "l2"
    C: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 350
    tol: float = 1e-4
    threads: int = 2

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.offsets != (-1, 0):
            raise ValueError("the first baseline is fixed to offsets (-1, 0)")
        if self.penalty != "l2" or self.solver != "lbfgs":
            raise ValueError("the first baseline contract is L2 + lbfgs")
        if self.C <= 0 or self.max_iter <= 0 or self.tol <= 0 or self.threads <= 0:
            raise ValueError("C, max_iter, tol and threads must be positive")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["offsets"] = list(self.offsets)
        result["encoding_version"] = DINUCLEOTIDE_ENCODING_VERSION
        result["training_population"] = "all train whitelist positions"
        result["aggregation"] = "exact 17 categories x 2 target classes"
        return result


@dataclass(frozen=True)
class BaselineFit:
    estimator: LogisticRegression
    encoded: EncodedSequenceFeatures
    aggregate_training_rows: pd.DataFrame
    train_category_summary: pd.DataFrame
    fit_report: pd.DataFrame


@dataclass(frozen=True)
class BaselineEvaluation:
    metric_summary: pd.DataFrame
    per_contig_metrics: pd.DataFrame
    category_summary: pd.DataFrame
    positive_predictions: pd.DataFrame
    precision_recall: pd.DataFrame


def fasta_path(split: PreparedSplit) -> Path:
    return split.paths.source / "genome/genome.fa.gz"


def load_baseline_sequences(
    split: PreparedSplit,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Load FASTA and verify that every split contig is represented."""

    sequences, alphabet = read_fasta(fasta_path(split))
    required = set(split.contig_assignment["contig"].astype(str))
    missing = sorted(required - set(sequences))
    if missing:
        raise ValueError("Split contigs missing from FASTA: " + ", ".join(missing))
    return sequences, alphabet


def _pair_codes_for_contig(raw_sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the pair category at every coordinate for both strands."""

    letters = UPPERCASE_BASE_CODE[np.asarray(raw_sequence, dtype=np.int64)]
    plus_upstream = np.concatenate((np.array([4], dtype=np.uint8), letters[:-1]))
    plus = np.where(
        (plus_upstream < 4) & (letters < 4),
        4 * plus_upstream + letters,
        16,
    ).astype(np.int8)

    minus_letters = COMPLEMENT_CODE[letters]
    minus_upstream = np.concatenate((minus_letters[1:], np.array([4], dtype=np.uint8)))
    minus = np.where(
        (minus_upstream < 4) & (minus_letters < 4),
        4 * minus_upstream + minus_letters,
        16,
    ).astype(np.int8)
    return plus, minus


def split_category_counts(
    split: PreparedSplit,
    sequences: Mapping[str, np.ndarray],
    split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count every position and retain every positive for train or validation."""

    if split_name not in {"train", "validation"}:
        raise ValueError("Category counts are allowed only for train or validation")
    intervals = split.intervals(split_name)
    positives = split.positive_targets(split_name)
    if positives is None or positives.empty:
        raise ValueError(f"{split_name} positives are unavailable")

    total_records: list[dict[str, object]] = []
    for contig in sorted(intervals["contig"].unique()):
        plus_codes, minus_codes = _pair_codes_for_contig(sequences[str(contig)])
        local = intervals.loc[intervals["contig"] == contig]
        for strand, strand_codes in (("+", plus_codes), ("-", minus_codes)):
            counts = np.zeros(len(DINUCLEOTIDE_LABELS), dtype=np.int64)
            for row in local.loc[local["strand"] == strand].itertuples(index=False):
                counts += np.bincount(
                    strand_codes[int(row.start) : int(row.end)],
                    minlength=len(DINUCLEOTIDE_LABELS),
                )
            total_records.extend(
                {
                    "contig": str(contig),
                    "strand": strand,
                    "category_code": code,
                    "category": DINUCLEOTIDE_LABELS[code],
                    "total_positions": int(value),
                }
                for code, value in enumerate(counts)
            )

    positive_rows = positives[
        ["template_index", "contig", "coordinate_0based", "strand", "count"]
    ].copy()
    context = extract_oriented_context(sequences, positive_rows, (-1, 0))
    positive_rows["category_code"] = dinucleotide_category_codes(context)
    positive_rows["category"] = positive_rows["category_code"].map(
        dict(enumerate(DINUCLEOTIDE_LABELS))
    )
    return pd.DataFrame(total_records), positive_rows


def category_summary(
    totals: pd.DataFrame,
    positives: pd.DataFrame,
) -> pd.DataFrame:
    """Combine denominators, positives, target rates and enrichment."""

    total_by_category = (
        totals.groupby(["category_code", "category"], as_index=False)["total_positions"]
        .sum()
        .sort_values("category_code")
    )
    positive_by_category = (
        positives.groupby("category_code").agg(
            positive_positions=("category_code", "size"),
            read_count_sum=("count", "sum"),
        )
        .reset_index()
    )
    result = total_by_category.merge(
        positive_by_category, on="category_code", how="left"
    ).fillna({"positive_positions": 0, "read_count_sum": 0})
    result[["positive_positions", "read_count_sum"]] = result[
        ["positive_positions", "read_count_sum"]
    ].astype("int64")
    result["negative_positions"] = (
        result["total_positions"] - result["positive_positions"]
    )
    prevalence = result["positive_positions"].sum() / result["total_positions"].sum()
    result["positive_rate"] = result["positive_positions"] / result["total_positions"]
    result["enrichment"] = result["positive_rate"] / prevalence
    return result


def _contexts_for_category_codes(codes) -> np.ndarray:
    canonical = np.array(
        [[first, second] for first in range(4) for second in range(4)],
        dtype=np.uint8,
    )
    contexts = np.vstack((canonical, np.array([[4, 4]], dtype=np.uint8)))
    return contexts[np.asarray(codes, dtype=np.int8)]


def aggregate_training_data(
    summary: pd.DataFrame,
    config: DinucleotideBaselineConfig,
) -> tuple[pd.DataFrame, EncodedSequenceFeatures]:
    """Collapse the exact full-population binary likelihood to 34 rows."""

    records: list[dict[str, object]] = []
    total_population = int(summary["total_positions"].sum())
    positive_population = int(summary["positive_positions"].sum())
    negative_population = total_population - positive_population
    multipliers = {
        0: total_population / (2 * negative_population),
        1: total_population / (2 * positive_population),
    }
    for row in summary.itertuples(index=False):
        class_counts = {0: int(row.negative_positions), 1: int(row.positive_positions)}
        for target in (0, 1):
            records.append(
                {
                    "category_code": int(row.category_code),
                    "category": str(row.category),
                    "target": target,
                    "population_count": class_counts[target],
                    "class_multiplier": multipliers[target],
                    "sample_weight": class_counts[target] * multipliers[target],
                }
            )
    aggregate = pd.DataFrame(records)
    encoded = encode_dinucleotide(
        _contexts_for_category_codes(aggregate["category_code"]),
        offsets=config.offsets,
        reference=config.reference_pair,
    )
    if not np.isclose(aggregate["sample_weight"].sum(), total_population):
        raise AssertionError("Class-balanced weights must preserve total size")
    by_class = aggregate.groupby("target")["sample_weight"].sum()
    if not np.isclose(by_class.loc[0], by_class.loc[1]):
        raise AssertionError("Aggregate class weights must have equal totals")
    return aggregate, encoded


def fit_dinucleotide_baseline(
    train_summary: pd.DataFrame,
    config: DinucleotideBaselineConfig,
) -> BaselineFit:
    """Fit L2 logistic regression to the exact aggregate train likelihood."""

    aggregate, encoded = aggregate_training_data(train_summary, config)
    estimator = LogisticRegression(
        penalty=config.penalty,
        C=config.C,
        solver=config.solver,
        max_iter=config.max_iter,
        tol=config.tol,
        random_state=config.seed,
    )
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(
        limits=config.threads
    ):
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(
            encoded.matrix,
            aggregate["target"].to_numpy(dtype=np.int8),
            sample_weight=aggregate["sample_weight"].to_numpy(dtype=np.float64),
        )
    elapsed = time.monotonic() - started
    messages = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    iterations = int(estimator.n_iter_[0])
    fit_report = pd.DataFrame(
        [
            {
                "solver": config.solver,
                "penalty": config.penalty,
                "C": config.C,
                "tol": config.tol,
                "iterations": iterations,
                "max_iter": config.max_iter,
                "converged": not messages and iterations < config.max_iter,
                "fit_seconds": elapsed,
                "aggregate_rows": len(aggregate),
                "represented_positions": int(aggregate["population_count"].sum()),
                "represented_positives": int(
                    aggregate.loc[aggregate["target"] == 1, "population_count"].sum()
                ),
                "features": encoded.matrix.shape[1],
                "negative_weight_sum": float(
                    aggregate.loc[aggregate["target"] == 0, "sample_weight"].sum()
                ),
                "positive_weight_sum": float(
                    aggregate.loc[aggregate["target"] == 1, "sample_weight"].sum()
                ),
                "warnings": " | ".join(messages),
            }
        ]
    )
    return BaselineFit(estimator, encoded, aggregate, train_summary, fit_report)


def category_scores(
    estimator: LogisticRegression,
    config: DinucleotideBaselineConfig,
) -> np.ndarray:
    """Evaluate the fitted model once for each fixed category."""

    encoded = encode_dinucleotide(
        _contexts_for_category_codes(range(len(DINUCLEOTIDE_LABELS))),
        offsets=config.offsets,
        reference=config.reference_pair,
    )
    return estimator.decision_function(encoded.matrix).astype(np.float64)


def _dense_rank_correlation(counts, scores) -> float:
    left = rankdata(np.asarray(counts), method="dense")
    right = rankdata(np.asarray(scores), method="dense")
    if len(left) < 2 or left.std() == 0 or right.std() == 0:
        return 0.0
    return float(np.clip(np.corrcoef(left, right)[0, 1], -1.0, 1.0))


def _average_precision_from_categories(
    total: np.ndarray,
    positive: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    """Exact AP without expanding positions; equal scores remain one tie."""

    total = np.asarray(total, dtype=np.int64)
    positive = np.asarray(positive, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive_population = int(positive.sum())
    if positive_population == 0:
        return 0.0, pd.DataFrame(
            columns=["threshold", "true_positive", "predicted_positive", "precision", "recall"]
        )

    records = []
    cumulative_positive = 0
    cumulative_total = 0
    average_precision = 0.0
    for threshold in sorted(np.unique(scores), reverse=True):
        selected = scores == threshold
        group_positive = int(positive[selected].sum())
        group_total = int(total[selected].sum())
        cumulative_positive += group_positive
        cumulative_total += group_total
        precision = cumulative_positive / cumulative_total
        recall = cumulative_positive / positive_population
        average_precision += (group_positive / positive_population) * precision
        records.append(
            {
                "threshold": float(threshold),
                "true_positive": cumulative_positive,
                "predicted_positive": cumulative_total,
                "precision": precision,
                "recall": recall,
            }
        )
    return float(average_precision), pd.DataFrame(records)


def _segment_metrics(
    totals: pd.DataFrame,
    positives: pd.DataFrame,
    candidate_scores: np.ndarray,
    *,
    segment: str,
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    metric_rows: list[dict[str, object]] = []
    curves: list[pd.DataFrame] = []
    values = ["overall"] if segment == "overall" else sorted(totals[segment].unique())
    for value in values:
        total_local = totals if segment == "overall" else totals.loc[totals[segment] == value]
        positive_local = positives if segment == "overall" else positives.loc[positives[segment] == value]
        total_by_category = (
            total_local.groupby("category_code")["total_positions"]
            .sum()
            .reindex(range(len(DINUCLEOTIDE_LABELS)), fill_value=0)
            .to_numpy(dtype=np.int64)
        )
        positive_by_category = (
            positive_local.groupby("category_code").size()
            .reindex(range(len(DINUCLEOTIDE_LABELS)), fill_value=0)
            .to_numpy(dtype=np.int64)
        )
        population = int(total_by_category.sum())
        n_positive = int(positive_by_category.sum())
        prevalence = n_positive / population
        candidate_ap, curve = _average_precision_from_categories(
            total_by_category, positive_by_category, candidate_scores
        )
        positive_score = candidate_scores[
            positive_local["category_code"].to_numpy(dtype=np.int8)
        ]
        candidate_spearman = _dense_rank_correlation(
            positive_local["count"].to_numpy(dtype=np.int64), positive_score
        )
        label = str(value)
        metric_rows.extend(
            [
                {
                    "segment": segment,
                    "segment_value": label,
                    "variant": "constant_reference",
                    "average_precision": prevalence,
                    "epic_spearman": 0.0,
                    "positions": population,
                    "positives": n_positive,
                    "prevalence": prevalence,
                    "unique_score_levels": 1,
                    "metric_note": (
                        "constant score: AP equals prevalence; Spearman recorded as "
                        "0 by explicit no-ranking convention"
                    ),
                },
                {
                    "segment": segment,
                    "segment_value": label,
                    "variant": "dinucleotide_logistic",
                    "average_precision": candidate_ap,
                    "epic_spearman": candidate_spearman,
                    "positions": population,
                    "positives": n_positive,
                    "prevalence": prevalence,
                    "unique_score_levels": int(
                        np.unique(candidate_scores[total_by_category > 0]).size
                    ),
                    "metric_note": "exact full-whitelist evaluation",
                },
            ]
        )
        if segment == "overall":
            curve.insert(0, "variant", "dinucleotide_logistic")
            curves.append(curve)
    return metric_rows, curves


def evaluate_dinucleotide_baseline(
    validation_totals: pd.DataFrame,
    validation_positives: pd.DataFrame,
    fit: BaselineFit,
    config: DinucleotideBaselineConfig,
) -> BaselineEvaluation:
    """Evaluate exact AP on all validation rows and Spearman on all positives."""

    scores = category_scores(fit.estimator, config)
    positives = validation_positives.copy()
    positives["candidate_score"] = scores[
        positives["category_code"].to_numpy(dtype=np.int8)
    ]
    positives["reference_score"] = 0.0

    overall_rows, curves = _segment_metrics(
        validation_totals, positives, scores, segment="overall"
    )
    contig_rows, _ = _segment_metrics(
        validation_totals, positives, scores, segment="contig"
    )
    metrics = pd.DataFrame(overall_rows)
    candidate = metrics.loc[metrics["variant"] == "dinucleotide_logistic"].iloc[0]
    reference = metrics.loc[metrics["variant"] == "constant_reference"].iloc[0]
    metric_summary = pd.DataFrame(
        [
            {
                "metric": "Average Precision",
                "direction": "maximize",
                "reference": reference["average_precision"],
                "candidate": candidate["average_precision"],
                "delta": candidate["average_precision"] - reference["average_precision"],
                "support": int(candidate["positions"]),
                "positive_support": int(candidate["positives"]),
            },
            {
                "metric": "EPIC Spearman",
                "direction": "maximize",
                "reference": reference["epic_spearman"],
                "candidate": candidate["epic_spearman"],
                "delta": candidate["epic_spearman"] - reference["epic_spearman"],
                "support": int(candidate["positives"]),
                "positive_support": int(candidate["positives"]),
            },
        ]
    )
    validation_summary = category_summary(validation_totals, positives)
    validation_summary["candidate_score"] = scores[
        validation_summary["category_code"].to_numpy(dtype=np.int8)
    ]
    return BaselineEvaluation(
        metric_summary=metric_summary,
        per_contig_metrics=pd.DataFrame(contig_rows),
        category_summary=validation_summary,
        positive_predictions=positives,
        precision_recall=pd.concat(curves, ignore_index=True),
    )


def coefficient_table(
    fit: BaselineFit,
    config: DinucleotideBaselineConfig,
) -> pd.DataFrame:
    """Return effects including an explicit zero coefficient for TT."""

    coefficients = dict(zip(fit.encoded.names, fit.estimator.coef_[0]))
    prefix = f"pair[{config.offsets[0]},{config.offsets[1]}]="
    records = []
    for category in DINUCLEOTIDE_LABELS:
        feature = prefix + category
        coefficient = 0.0 if category == config.reference_pair else float(coefficients[feature])
        records.append(
            {
                "category": category,
                "feature": feature,
                "coefficient": coefficient,
                "odds_ratio": float(np.exp(np.clip(coefficient, -700, 700))),
                "is_reference": category == config.reference_pair,
            }
        )
    return pd.DataFrame(records)


__all__ = [
    "BaselineEvaluation",
    "BaselineFit",
    "DinucleotideBaselineConfig",
    "aggregate_training_data",
    "category_scores",
    "category_summary",
    "coefficient_table",
    "evaluate_dinucleotide_baseline",
    "fasta_path",
    "fit_dinucleotide_baseline",
    "load_baseline_sequences",
    "split_category_counts",
]
