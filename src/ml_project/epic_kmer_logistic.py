"""Sparse hierarchical k-mer logistic regression for EPIC experiments.

The module contains the reusable mechanics for EXP-005.  Importing it never
fits a model.  Training happens only through explicit function calls from the
experiment notebook.

The design has one active column at every requested suffix level.  For the
default ``(2, 4, 6)`` hierarchy a canonical 6-mer activates its 2-mer suffix,
4-mer suffix and full 6-mer.  A 6-mer ``N/edge`` activates the dedicated
``N/edge`` column at every level, matching the conservative ambiguity policy
used by the lookup experiments.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time
import warnings
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .epic_baseline import (
    _average_precision_from_categories,
    _dense_rank_correlation,
)
from .epic_kmer import (
    KmerCounts,
    SUPPORTED_K,
    kmer_category_count,
    kmer_category_labels,
)


DEFAULT_C_GRID = (
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    3e-1,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
)


@dataclass(frozen=True)
class HierarchicalKmerLogisticConfig:
    """Every setting capable of changing the EXP-005 logistic result."""

    ks: tuple[int, ...] = (2, 4, 6)
    C_grid: tuple[float, ...] = DEFAULT_C_GRID
    penalty: str = "l2"
    solver: str = "lbfgs"
    max_iter: int = 500
    tol: float = 1e-8
    threads: int = 2
    seed: int = 42
    balance_classes: bool = True
    cv_folds: int = 3
    near_best_relative_ap: float = 0.01

    def __post_init__(self) -> None:
        if not self.ks or tuple(sorted(set(self.ks))) != self.ks:
            raise ValueError("ks must be strictly increasing")
        if any(k not in SUPPORTED_K for k in self.ks):
            raise ValueError(f"ks must be drawn from {SUPPORTED_K}")
        if self.ks[-1] != 6:
            raise ValueError("EXP-005 is fixed to the 6-mer champion")
        if not self.C_grid or any(not np.isfinite(c) or c <= 0 for c in self.C_grid):
            raise ValueError("C_grid values must be finite and positive")
        if tuple(sorted(set(self.C_grid))) != self.C_grid:
            raise ValueError("C_grid must be strictly increasing")
        if self.penalty != "l2" or self.solver != "lbfgs":
            raise ValueError("EXP-005 is fixed to L2 logistic regression with lbfgs")
        if self.max_iter <= 0 or self.tol <= 0 or self.threads <= 0:
            raise ValueError("max_iter, tol and threads must be positive")
        if self.seed < 0 or self.cv_folds < 2:
            raise ValueError("seed must be non-negative and cv_folds at least two")
        if not 0 <= self.near_best_relative_ap < 1:
            raise ValueError("near_best_relative_ap must be in [0, 1)")

    @property
    def max_k(self) -> int:
        return self.ks[-1]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["ks"] = list(self.ks)
        result["C_grid"] = list(self.C_grid)
        result["feature_encoding"] = "full one-hot at every suffix level"
        result["ambiguous_policy"] = "max-k N/edge activates N/edge at every level"
        result["training_population"] = "all train whitelist positions"
        result["aggregation"] = "exact category x binary-class sufficient statistics"
        return result


@dataclass(frozen=True)
class HierarchicalKmerDesign:
    """Fixed sparse features for every possible max-k category."""

    matrix: sparse.csr_matrix
    feature_table: pd.DataFrame
    category_table: pd.DataFrame
    ks: tuple[int, ...]
    max_k: int


@dataclass(frozen=True)
class AggregateBinaryLikelihood:
    """Exact repeated-row binary likelihood represented by sample weights."""

    matrix: sparse.csr_matrix
    target: np.ndarray
    sample_weight: np.ndarray
    rows: pd.DataFrame


@dataclass(frozen=True)
class HierarchicalKmerLogisticFit:
    """One fitted C candidate plus transparent diagnostics."""

    estimator: LogisticRegression
    design: HierarchicalKmerDesign
    aggregate: AggregateBinaryLikelihood
    category_scores: np.ndarray
    coefficient_table: pd.DataFrame
    fit_report: pd.DataFrame


@dataclass(frozen=True)
class CategoryScoreEvaluation:
    """Exact full-population metrics for one vector of category scores."""

    metric_summary: pd.DataFrame
    precision_recall: pd.DataFrame
    positive_predictions: pd.DataFrame


def hierarchical_kmer_design(
    config: HierarchicalKmerLogisticConfig = HierarchicalKmerLogisticConfig(),
) -> HierarchicalKmerDesign:
    """Create the fixed suffix-hierarchy one-hot matrix without fitting."""

    feature_rows: list[dict[str, object]] = []
    offsets: dict[int, int] = {}
    column_offset = 0
    for k in config.ks:
        offsets[k] = column_offset
        for code, label in enumerate(kmer_category_labels(k)):
            feature_rows.append(
                {
                    "feature_index": column_offset + code,
                    "level_k": k,
                    "category_code": code,
                    "category": label,
                    "feature_name": f"k{k}={label}",
                }
            )
        column_offset += kmer_category_count(k)

    max_categories = kmer_category_count(config.max_k)
    max_n_edge = max_categories - 1
    row_index = np.repeat(np.arange(max_categories, dtype=np.int32), len(config.ks))
    column_index = np.empty(max_categories * len(config.ks), dtype=np.int32)
    category_records: list[dict[str, object]] = []
    cursor = 0
    for max_code in range(max_categories):
        record: dict[str, object] = {
            "max_category_code": max_code,
            "max_category": kmer_category_labels(config.max_k)[max_code],
        }
        for k in config.ks:
            if max_code == max_n_edge:
                local_code = kmer_category_count(k) - 1
            else:
                local_code = max_code % (4**k)
            column_index[cursor] = offsets[k] + local_code
            cursor += 1
            record[f"kmer_{k}_code"] = local_code
            record[f"kmer_{k}"] = kmer_category_labels(k)[local_code]
        category_records.append(record)

    matrix = sparse.csr_matrix(
        (
            np.ones(len(row_index), dtype=np.float64),
            (row_index, column_index),
        ),
        shape=(max_categories, column_offset),
    )
    return HierarchicalKmerDesign(
        matrix=matrix,
        feature_table=pd.DataFrame(feature_rows),
        category_table=pd.DataFrame(category_records),
        ks=config.ks,
        max_k=config.max_k,
    )


def _normalise_contigs(contigs: Iterable[str] | None) -> tuple[str, ...] | None:
    if contigs is None:
        return None
    result = tuple(sorted({str(value) for value in contigs}))
    if not result:
        raise ValueError("contigs cannot be empty")
    return result


def aggregate_category_arrays(
    counts: KmerCounts,
    k: int,
    *,
    contigs: Iterable[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact total/positive counts, optionally for selected contigs."""

    if k not in counts.ks:
        raise ValueError(f"k={k} is absent from the count bundle")
    selected = _normalise_contigs(contigs)
    total_rows = counts.totals.loc[counts.totals["k"] == k]
    positive_rows = counts.positives
    if selected is not None:
        available = set(counts.totals["contig"].astype(str))
        missing = sorted(set(selected) - available)
        if missing:
            raise ValueError("unknown contigs: " + ", ".join(missing))
        total_rows = total_rows.loc[total_rows["contig"].astype(str).isin(selected)]
        positive_rows = positive_rows.loc[
            positive_rows["contig"].astype(str).isin(selected)
        ]
    size = kmer_category_count(k)
    total = (
        total_rows.groupby("category_code")["total_positions"]
        .sum()
        .reindex(range(size), fill_value=0)
        .to_numpy(dtype=np.int64)
    )
    code_column = f"kmer_{k}_code"
    positive = (
        positive_rows[code_column]
        .value_counts()
        .reindex(range(size), fill_value=0)
        .to_numpy(dtype=np.int64)
    )
    if np.any(positive > total):
        raise AssertionError("positive category count exceeds its denominator")
    return total, positive


def aggregate_binary_likelihood(
    counts: KmerCounts,
    design: HierarchicalKmerDesign,
    *,
    contigs: Iterable[str] | None = None,
    balance_classes: bool = True,
) -> AggregateBinaryLikelihood:
    """Collapse the full binary likelihood to category-by-class rows exactly."""

    total, positive = aggregate_category_arrays(
        counts, design.max_k, contigs=contigs
    )
    negative = total - positive
    population = int(total.sum())
    positive_population = int(positive.sum())
    negative_population = population - positive_population
    if population <= 0 or positive_population <= 0 or negative_population <= 0:
        raise ValueError("selected population must contain both classes")

    category_codes = np.repeat(np.arange(len(total), dtype=np.int32), 2)
    targets = np.tile(np.array([0, 1], dtype=np.int8), len(total))
    population_counts = np.column_stack((negative, positive)).reshape(-1)
    keep = population_counts > 0
    category_codes = category_codes[keep]
    targets = targets[keep]
    population_counts = population_counts[keep]

    if balance_classes:
        multipliers = np.array(
            [
                population / (2 * negative_population),
                population / (2 * positive_population),
            ],
            dtype=np.float64,
        )
    else:
        multipliers = np.ones(2, dtype=np.float64)
    weights = population_counts.astype(np.float64) * multipliers[targets]
    rows = pd.DataFrame(
        {
            "category_code": category_codes,
            "category": np.asarray(kmer_category_labels(design.max_k))[category_codes],
            "target": targets,
            "population_count": population_counts,
            "class_multiplier": multipliers[targets],
            "sample_weight": weights,
        }
    )
    matrix = design.matrix[category_codes]
    if int(rows["population_count"].sum()) != population:
        raise AssertionError("aggregate rows do not preserve the population")
    if balance_classes:
        sums = rows.groupby("target")["sample_weight"].sum()
        if not np.isclose(sums.loc[0], sums.loc[1]):
            raise AssertionError("balanced class weights must have equal sums")
    return AggregateBinaryLikelihood(
        matrix=matrix,
        target=targets,
        sample_weight=weights,
        rows=rows,
    )


def make_balanced_contig_folds(
    counts: KmerCounts,
    *,
    k: int = 6,
    n_splits: int = 3,
) -> pd.DataFrame:
    """Assign whole train contigs greedily by full population size."""

    if counts.split_name != "train":
        raise ValueError("CV folds can only be constructed from train counts")
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    total_rows = counts.totals.loc[counts.totals["k"] == k]
    population = (
        total_rows.groupby("contig")["total_positions"].sum().astype(np.int64)
    )
    positive = counts.positives.groupby("contig").size().astype(np.int64)
    summary = pd.DataFrame(
        {
            "contig": population.index.astype(str),
            "positions": population.to_numpy(),
            "positives": positive.reindex(population.index, fill_value=0).to_numpy(),
        }
    ).sort_values(["positions", "contig"], ascending=[False, True])
    if len(summary) < n_splits:
        raise ValueError("n_splits exceeds the number of train contigs")

    fold_positions = np.zeros(n_splits, dtype=np.int64)
    assignments: list[int] = []
    for row in summary.itertuples(index=False):
        fold = int(np.argmin(fold_positions))
        assignments.append(fold)
        fold_positions[fold] += int(row.positions)
    summary["fold"] = assignments
    summary["prevalence"] = summary["positives"] / summary["positions"]
    return summary.sort_values(["fold", "contig"]).reset_index(drop=True)


def fit_hierarchical_kmer_logistic(
    counts: KmerCounts,
    *,
    C: float,
    config: HierarchicalKmerLogisticConfig = HierarchicalKmerLogisticConfig(),
    contigs: Iterable[str] | None = None,
    design: HierarchicalKmerDesign | None = None,
) -> HierarchicalKmerLogisticFit:
    """Fit one explicit L2 candidate; calling this function trains a model."""

    if counts.split_name != "train":
        raise ValueError("fit requires the train count bundle")
    if C not in config.C_grid:
        raise ValueError("C must be one of the preregistered C_grid values")
    design = design or hierarchical_kmer_design(config)
    aggregate = aggregate_binary_likelihood(
        counts,
        design,
        contigs=contigs,
        balance_classes=config.balance_classes,
    )
    estimator = LogisticRegression(
        penalty=config.penalty,
        C=float(C),
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
            aggregate.matrix,
            aggregate.target,
            sample_weight=aggregate.sample_weight,
        )
    elapsed = time.monotonic() - started
    messages = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    iterations = int(estimator.n_iter_[0])
    scores = estimator.decision_function(design.matrix).astype(np.float64)
    coefficients = design.feature_table.copy()
    coefficients["coefficient"] = estimator.coef_[0]
    coefficients["absolute_coefficient"] = np.abs(estimator.coef_[0])
    fit_report = pd.DataFrame(
        [
            {
                "C": float(C),
                "solver": config.solver,
                "penalty": config.penalty,
                "tol": config.tol,
                "iterations": iterations,
                "max_iter": config.max_iter,
                "converged": not messages and iterations < config.max_iter,
                "fit_seconds": elapsed,
                "aggregate_rows": len(aggregate.rows),
                "features": design.matrix.shape[1],
                "represented_positions": int(
                    aggregate.rows["population_count"].sum()
                ),
                "represented_positives": int(
                    aggregate.rows.loc[
                        aggregate.rows["target"] == 1, "population_count"
                    ].sum()
                ),
                "intercept": float(estimator.intercept_[0]),
                "warnings": " | ".join(messages),
            }
        ]
    )
    return HierarchicalKmerLogisticFit(
        estimator=estimator,
        design=design,
        aggregate=aggregate,
        category_scores=scores,
        coefficient_table=coefficients,
        fit_report=fit_report,
    )


def evaluate_category_scores(
    counts: KmerCounts,
    scores: Sequence[float],
    *,
    k: int = 6,
    contigs: Iterable[str] | None = None,
    variant: str = "hierarchical_2_4_6mer_lr",
) -> CategoryScoreEvaluation:
    """Evaluate one category-score vector on an exact population slice."""

    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (kmer_category_count(k),) or not np.all(np.isfinite(values)):
        raise ValueError("scores have the wrong shape or contain non-finite values")
    selected = _normalise_contigs(contigs)
    total, positive = aggregate_category_arrays(counts, k, contigs=selected)
    represented = total > 0
    average_precision, curve = _average_precision_from_categories(
        total[represented], positive[represented], values[represented]
    )
    positive_rows = counts.positives
    if selected is not None:
        positive_rows = positive_rows.loc[
            positive_rows["contig"].astype(str).isin(selected)
        ]
    positive_scores = values[
        positive_rows[f"kmer_{k}_code"].to_numpy(dtype=np.int32)
    ]
    spearman = _dense_rank_correlation(
        positive_rows["count"].to_numpy(dtype=np.int64), positive_scores
    )
    positive_loss = np.logaddexp(0.0, -values)
    negative_loss = np.logaddexp(0.0, values)
    population_log_loss = float(
        (positive @ positive_loss + (total - positive) @ negative_loss)
        / total.sum()
    )
    balanced_log_loss = float(
        0.5
        * (
            (positive @ positive_loss) / positive.sum()
            + ((total - positive) @ negative_loss) / (total - positive).sum()
        )
    )
    segment = "overall" if selected is None else ",".join(selected)
    metric = pd.DataFrame(
        [
            {
                "segment": segment,
                "variant": variant,
                "average_precision": average_precision,
                "epic_spearman": spearman,
                "population_log_loss": population_log_loss,
                "balanced_log_loss": balanced_log_loss,
                "positions": int(total.sum()),
                "positives": int(positive.sum()),
                "prevalence": float(positive.sum() / total.sum()),
                "unique_score_levels": int(np.unique(values[represented]).size),
            }
        ]
    )
    curve.insert(0, "variant", variant)
    predictions = positive_rows[
        ["template_index", "contig", "coordinate_0based", "strand", "count"]
    ].copy()
    predictions["score"] = positive_scores
    predictions["variant"] = variant
    return CategoryScoreEvaluation(metric, curve, predictions)


def cross_validate_hierarchical_kmer_logistic(
    train_counts: KmerCounts,
    fold_assignment: pd.DataFrame,
    *,
    config: HierarchicalKmerLogisticConfig = HierarchicalKmerLogisticConfig(),
) -> pd.DataFrame:
    """Run the explicit train-contig CV grid; calling this trains models."""

    required = {"contig", "fold"}
    if not required.issubset(fold_assignment.columns):
        raise ValueError("fold_assignment must contain contig and fold")
    expected = set(train_counts.totals["contig"].astype(str))
    actual = set(fold_assignment["contig"].astype(str))
    if expected != actual:
        raise ValueError("fold_assignment must contain every train contig exactly once")
    if fold_assignment["contig"].astype(str).duplicated().any():
        raise ValueError("each contig must occur exactly once")

    design = hierarchical_kmer_design(config)
    records: list[dict[str, object]] = []
    folds = sorted(fold_assignment["fold"].astype(int).unique())
    for C in config.C_grid:
        for fold in folds:
            validation_contigs = tuple(
                fold_assignment.loc[
                    fold_assignment["fold"].astype(int) == fold, "contig"
                ].astype(str)
            )
            fitting_contigs = tuple(sorted(expected - set(validation_contigs)))
            fitted = fit_hierarchical_kmer_logistic(
                train_counts,
                C=C,
                config=config,
                contigs=fitting_contigs,
                design=design,
            )
            evaluated = evaluate_category_scores(
                train_counts,
                fitted.category_scores,
                k=config.max_k,
                contigs=validation_contigs,
            )
            records.append(
                {
                    "C": float(C),
                    "fold": int(fold),
                    "validation_contigs": ",".join(validation_contigs),
                    **evaluated.metric_summary.iloc[0].to_dict(),
                    **fitted.fit_report.iloc[0][
                        ["iterations", "converged", "fit_seconds", "warnings"]
                    ].to_dict(),
                }
            )
    return pd.DataFrame(records)


def summarise_cv(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold results without choosing C."""

    required = {
        "C",
        "fold",
        "average_precision",
        "epic_spearman",
        "balanced_log_loss",
        "converged",
        "iterations",
        "fit_seconds",
    }
    if not required.issubset(cv_results.columns):
        raise ValueError("cv_results is missing required columns")
    return (
        cv_results.groupby("C", as_index=False)
        .agg(
            mean_average_precision=("average_precision", "mean"),
            std_average_precision=("average_precision", "std"),
            mean_epic_spearman=("epic_spearman", "mean"),
            std_epic_spearman=("epic_spearman", "std"),
            mean_balanced_log_loss=("balanced_log_loss", "mean"),
            all_folds_converged=("converged", "all"),
            max_iterations=("iterations", "max"),
            total_fit_seconds=("fit_seconds", "sum"),
        )
        .sort_values("C")
        .reset_index(drop=True)
    )


def select_regularization(
    cv_summary: pd.DataFrame,
    *,
    relative_ap_tolerance: float = 0.01,
) -> pd.Series:
    """Select the strongest converged C within 1% AP and Spearman guardrail."""

    converged = cv_summary.loc[cv_summary["all_folds_converged"].astype(bool)].copy()
    if converged.empty:
        raise RuntimeError("no C converged on every fold")
    best_index = converged["mean_average_precision"].idxmax()
    best = converged.loc[best_index]
    eligible = converged.loc[
        (converged["mean_average_precision"] >= (1 - relative_ap_tolerance) * best["mean_average_precision"])
        & (converged["mean_epic_spearman"] >= best["mean_epic_spearman"])
    ]
    if eligible.empty:
        raise AssertionError("the best-AP candidate must always be eligible")
    return eligible.sort_values("C").iloc[0]


__all__ = [
    "DEFAULT_C_GRID",
    "AggregateBinaryLikelihood",
    "CategoryScoreEvaluation",
    "HierarchicalKmerDesign",
    "HierarchicalKmerLogisticConfig",
    "HierarchicalKmerLogisticFit",
    "aggregate_binary_likelihood",
    "aggregate_category_arrays",
    "cross_validate_hierarchical_kmer_logistic",
    "evaluate_category_scores",
    "fit_hierarchical_kmer_logistic",
    "hierarchical_kmer_design",
    "make_balanced_contig_folds",
    "select_regularization",
    "summarise_cv",
]
