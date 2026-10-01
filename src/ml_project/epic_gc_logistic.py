"""Exact GC201 feature and hierarchical k-mer logistic model for EXP-006.

The module is intentionally inert on import.  Counting and fitting happen only
when the experiment notebook calls the explicit public functions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
import warnings
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .epic_baseline import _dense_rank_correlation
from .epic_data import PreparedSplit
from .epic_kmer import (
    KmerCounts,
    _codes_from_state,
    _packed_suffix_state,
    _uppercase_letters,
    context_kmer_codes,
    kmer_category_count,
    kmer_category_labels,
)
from .epic_kmer_logistic import (
    DEFAULT_C_GRID,
    HierarchicalKmerDesign,
    hierarchical_kmer_design,
)
from .sequence_composition import (
    gc_counts_from_prefix,
    gc_fraction,
    gc_prefix_from_base_codes,
    gc_window_counts,
)
from .sequence_context import COMPLEMENT_CODE, extract_oriented_context


@dataclass(frozen=True)
class GCWindowLogisticConfig:
    """Every setting capable of changing the EXP-006 candidate."""

    ks: tuple[int, ...] = (2, 4, 6)
    window_start: int = -100
    window_end: int = 100
    C_grid: tuple[float, ...] = DEFAULT_C_GRID
    penalty: str = "l2"
    solver: str = "lbfgs"
    max_iter: int = 2000
    tol: float = 1e-8
    threads: int = 2
    seed: int = 42
    balance_classes: bool = True
    cv_folds: int = 3
    near_best_relative_ap: float = 0.01
    count_batch_size: int = 1_000_000

    def __post_init__(self) -> None:
        if self.ks != (2, 4, 6):
            raise ValueError("EXP-006 is fixed to the hierarchical 2/4/6-mer base")
        if self.window_start != -100 or self.window_end != 100:
            raise ValueError("EXP-006 is fixed to the inclusive GC window [-100, +100]")
        if not self.C_grid or tuple(sorted(set(self.C_grid))) != self.C_grid:
            raise ValueError("C_grid must be non-empty, unique, and increasing")
        if any(not np.isfinite(value) or value <= 0 for value in self.C_grid):
            raise ValueError("every C must be finite and positive")
        if self.penalty != "l2" or self.solver != "lbfgs":
            raise ValueError("EXP-006 is fixed to L2 logistic regression with lbfgs")
        if self.max_iter <= 0 or self.tol <= 0 or self.threads <= 0:
            raise ValueError("max_iter, tol, and threads must be positive")
        if self.seed < 0 or self.cv_folds < 2 or self.count_batch_size <= 0:
            raise ValueError("seed, cv_folds, and count_batch_size are invalid")
        if not 0 <= self.near_best_relative_ap < 1:
            raise ValueError("near_best_relative_ap must be in [0, 1)")

    @property
    def max_k(self) -> int:
        return self.ks[-1]

    @property
    def window_size(self) -> int:
        return self.window_end - self.window_start + 1

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["ks"] = list(self.ks)
        result["C_grid"] = list(self.C_grid)
        result["gc_feature"] = "GC / valid ACGT in inclusive offsets [-100, +100]"
        result["gc_missing_policy"] = "train-mean imputation when valid_count == 0"
        result["gc_scaling"] = "weighted mean/std fitted inside each train fold"
        result["feature_encoding"] = "hierarchical 2/4/6-mer one-hot + standardized GC201"
        result["aggregation"] = (
            "exact contig x strand x 6-mer x gc_count x valid_count sufficient statistics"
        )
        result["training_population"] = "all train whitelist positions"
        return result


@dataclass(frozen=True)
class KmerGCCounts:
    """Exact joint denominators and sparse positives for k-mer plus GC201."""

    split_name: str
    window_start: int
    window_end: int
    totals: pd.DataFrame
    positives: pd.DataFrame


@dataclass(frozen=True)
class AggregateGCBinaryLikelihood:
    """Exact repeated-row likelihood with train-only GC scaling."""

    matrix: sparse.csr_matrix
    target: np.ndarray
    sample_weight: np.ndarray
    rows: pd.DataFrame
    gc_mean: float
    gc_std: float


@dataclass(frozen=True)
class GCWindowLogisticFit:
    """One fitted candidate and all interpretation metadata."""

    estimator: LogisticRegression
    design: HierarchicalKmerDesign
    aggregate: AggregateGCBinaryLikelihood
    coefficient_table: pd.DataFrame
    fit_report: pd.DataFrame


@dataclass(frozen=True)
class GCWindowEvaluation:
    """Exact full-population evaluation without expanding negatives."""

    metric_summary: pd.DataFrame
    per_contig_metrics: pd.DataFrame
    precision_recall: pd.DataFrame
    positive_predictions: pd.DataFrame


def _normalise_contigs(contigs: Iterable[str] | None) -> tuple[str, ...] | None:
    if contigs is None:
        return None
    result = tuple(sorted({str(value) for value in contigs}))
    if not result:
        raise ValueError("contigs cannot be empty")
    return result


def _reduce_packed_counts(keys: list[np.ndarray], counts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not keys:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    all_keys = np.concatenate(keys)
    all_counts = np.concatenate(counts)
    order = np.argsort(all_keys, kind="mergesort")
    ordered_keys = all_keys[order]
    ordered_counts = all_counts[order]
    first = np.r_[True, ordered_keys[1:] != ordered_keys[:-1]]
    starts = np.flatnonzero(first)
    return ordered_keys[starts], np.add.reduceat(ordered_counts, starts)


def _count_oriented_ranges(
    letters: np.ndarray,
    ranges: Sequence[tuple[int, int]],
    config: GCWindowLogisticConfig,
) -> pd.DataFrame:
    """Count exact feature tuples for one oriented contig-strand."""

    packed, unknown_prefix = _packed_suffix_state(letters, config.max_k)
    prefix = gc_prefix_from_base_codes(letters)
    base = config.window_size + 1
    unique_parts: list[np.ndarray] = []
    count_parts: list[np.ndarray] = []
    buffered_keys: list[np.ndarray] = []
    buffered = 0

    def flush() -> None:
        nonlocal buffered_keys, buffered
        if not buffered_keys:
            return
        values = buffered_keys[0] if len(buffered_keys) == 1 else np.concatenate(buffered_keys)
        local_keys, local_counts = np.unique(values, return_counts=True)
        unique_parts.append(local_keys.astype(np.int64, copy=False))
        count_parts.append(local_counts.astype(np.int64, copy=False))
        buffered_keys = []
        buffered = 0

    for interval_start, interval_end in ranges:
        cursor = int(interval_start)
        stop = int(interval_end)
        if not 0 <= cursor < stop <= len(letters):
            raise ValueError("whitelist interval is outside its FASTA contig")
        while cursor < stop:
            take = min(stop - cursor, config.count_batch_size - buffered)
            end = cursor + take
            coordinates = np.arange(cursor, end, dtype=np.int64)
            codes = _codes_from_state(
                packed,
                unknown_prefix,
                cursor,
                end,
                config.max_k,
            ).astype(np.int64, copy=False)
            gc_count, valid_count = gc_counts_from_prefix(
                prefix,
                coordinates,
                window_start=config.window_start,
                window_end=config.window_end,
            )
            keys = (
                (codes * base + valid_count.astype(np.int64)) * base
                + gc_count.astype(np.int64)
            )
            buffered_keys.append(keys)
            buffered += take
            cursor = end
            if buffered == config.count_batch_size:
                flush()
    flush()

    packed_keys, population_counts = _reduce_packed_counts(unique_parts, count_parts)
    gc_count = packed_keys % base
    remainder = packed_keys // base
    valid_count = remainder % base
    kmer_code = remainder // base
    return pd.DataFrame(
        {
            "kmer_6_code": kmer_code.astype(np.int32),
            "gc_count": gc_count.astype(np.int16),
            "valid_count": valid_count.astype(np.int16),
            "total_positions": population_counts.astype(np.int64),
        }
    )


def _positive_feature_rows(
    sequences: Mapping[str, np.ndarray],
    positives: pd.DataFrame,
    config: GCWindowLogisticConfig,
) -> pd.DataFrame:
    result = positives[
        ["template_index", "contig", "coordinate_0based", "strand", "count"]
    ].copy()
    context = extract_oriented_context(
        sequences,
        result,
        offsets=range(-(config.max_k - 1), 1),
    )
    result["kmer_6_code"] = context_kmer_codes(context)
    result["gc_count"] = np.int16(0)
    result["valid_count"] = np.int16(0)
    for contig, group in result.groupby("contig", sort=False):
        coordinates = group["coordinate_0based"].to_numpy(dtype=np.int64)
        gc_count, valid_count = gc_window_counts(
            sequences[str(contig)],
            coordinates,
            window_start=config.window_start,
            window_end=config.window_end,
        )
        result.loc[group.index, "gc_count"] = gc_count
        result.loc[group.index, "valid_count"] = valid_count
    result["gc_count"] = result["gc_count"].astype("int16")
    result["valid_count"] = result["valid_count"].astype("int16")
    result["gc_fraction"] = gc_fraction(
        result["gc_count"].to_numpy(), result["valid_count"].to_numpy()
    )
    return result


def split_kmer_gc_counts(
    split: PreparedSplit,
    sequences: Mapping[str, np.ndarray],
    split_name: str,
    *,
    config: GCWindowLogisticConfig = GCWindowLogisticConfig(),
) -> KmerGCCounts:
    """Count every whitelist position by exact 6-mer and GC201 statistics."""

    if split_name not in {"train", "validation"}:
        raise ValueError("GC counts require train or validation labels")
    intervals = split.intervals(split_name)
    positives = split.positive_targets(split_name)
    if positives is None or positives.empty:
        raise ValueError(f"{split_name} positives are unavailable")
    missing = sorted(set(intervals["contig"].astype(str)) - set(sequences))
    if missing:
        raise ValueError("split contigs missing from FASTA: " + ", ".join(missing))

    records: list[pd.DataFrame] = []
    for contig in sorted(intervals["contig"].astype(str).unique()):
        letters = _uppercase_letters(sequences[contig])
        local = intervals.loc[intervals["contig"].astype(str) == contig]
        for strand in ("+", "-"):
            strand_intervals = local.loc[local["strand"] == strand, ["start", "end"]]
            if strand_intervals.empty:
                continue
            ranges = [tuple(map(int, row)) for row in strand_intervals.to_numpy()]
            if strand == "+":
                oriented = letters
                oriented_ranges = ranges
            else:
                oriented = COMPLEMENT_CODE[letters[::-1]]
                length = len(letters)
                oriented_ranges = [(length - end, length - start) for start, end in ranges]
            counted = _count_oriented_ranges(oriented, oriented_ranges, config)
            counted.insert(0, "strand", strand)
            counted.insert(0, "contig", contig)
            records.append(counted)

    totals = pd.concat(records, ignore_index=True)
    positives_with_features = _positive_feature_rows(sequences, positives, config)
    expected_population = int((intervals["end"] - intervals["start"]).sum())
    if int(totals["total_positions"].sum()) != expected_population:
        raise AssertionError("joint GC counts do not cover the complete split")

    keys = ["contig", "strand", "kmer_6_code", "gc_count", "valid_count"]
    total_by_key = totals.groupby(keys)["total_positions"].sum()
    positive_by_key = positives_with_features.groupby(keys).size()
    aligned = total_by_key.reindex(positive_by_key.index, fill_value=0)
    if np.any(positive_by_key.to_numpy() > aligned.to_numpy()):
        raise AssertionError("positive tuple count exceeds its denominator")
    return KmerGCCounts(
        split_name,
        config.window_start,
        config.window_end,
        totals,
        positives_with_features,
    )


def kmer_counts_from_gc_counts(
    counts: KmerGCCounts,
    *,
    ks: Sequence[int] = (2, 4, 6),
) -> KmerCounts:
    """Recover exact k-mer marginals without rescanning the FASTA."""

    normalized = tuple(int(k) for k in ks)
    if normalized != tuple(sorted(set(normalized))) or normalized[-1] != 6:
        raise ValueError("ks must be increasing and end at 6")
    max_n_edge = kmer_category_count(6) - 1
    total_frames: list[pd.DataFrame] = []
    positive_rows = counts.positives.copy()
    max_codes_total = counts.totals["kmer_6_code"].to_numpy(dtype=np.int32)
    max_codes_positive = positive_rows["kmer_6_code"].to_numpy(dtype=np.int32)
    for k in normalized:
        n_edge = kmer_category_count(k) - 1
        total_codes = np.where(
            max_codes_total == max_n_edge,
            n_edge,
            max_codes_total % (4**k),
        ).astype(np.int32)
        local = counts.totals[["contig", "strand", "total_positions"]].copy()
        local["k"] = np.int8(k)
        local["category_code"] = total_codes
        local = (
            local.groupby(["k", "contig", "strand", "category_code"], as_index=False)[
                "total_positions"
            ]
            .sum()
        )
        total_frames.append(local)
        positive_rows[f"kmer_{k}_code"] = np.where(
            max_codes_positive == max_n_edge,
            n_edge,
            max_codes_positive % (4**k),
        ).astype(np.int32)
    totals = pd.concat(total_frames, ignore_index=True)
    return KmerCounts(counts.split_name, normalized, totals, positive_rows)


def _joint_summary(
    counts: KmerGCCounts,
    *,
    contigs: Iterable[str] | None = None,
) -> pd.DataFrame:
    selected = _normalise_contigs(contigs)
    totals = counts.totals
    positives = counts.positives
    if selected is not None:
        available = set(totals["contig"].astype(str))
        missing = sorted(set(selected) - available)
        if missing:
            raise ValueError("unknown contigs: " + ", ".join(missing))
        totals = totals.loc[totals["contig"].astype(str).isin(selected)]
        positives = positives.loc[positives["contig"].astype(str).isin(selected)]
    keys = ["kmer_6_code", "gc_count", "valid_count"]
    total = totals.groupby(keys, as_index=False)["total_positions"].sum()
    positive = positives.groupby(keys).size().rename("positive_positions").reset_index()
    result = total.merge(positive, on=keys, how="left", validate="one_to_one")
    result["positive_positions"] = result["positive_positions"].fillna(0).astype("int64")
    result["negative_positions"] = (
        result["total_positions"] - result["positive_positions"]
    )
    if (result["negative_positions"] < 0).any():
        raise AssertionError("positive tuple count exceeds total")
    result["gc_fraction"] = gc_fraction(
        result["gc_count"].to_numpy(), result["valid_count"].to_numpy()
    )
    return result


def _gc_scaler(summary: pd.DataFrame) -> tuple[float, float]:
    raw = summary["gc_fraction"].to_numpy(dtype=np.float64)
    weight = summary["total_positions"].to_numpy(dtype=np.float64)
    observed = np.isfinite(raw)
    if not observed.any() or weight[observed].sum() <= 0:
        raise ValueError("GC201 has no observed values")
    mean = float(np.average(raw[observed], weights=weight[observed]))
    imputed = np.where(observed, raw, mean)
    variance = float(np.average((imputed - mean) ** 2, weights=weight))
    std = float(np.sqrt(variance))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("GC201 has zero or invalid train variance")
    return mean, std


def _matrix_for_summary(
    summary: pd.DataFrame,
    design: HierarchicalKmerDesign,
    *,
    gc_mean: float,
    gc_std: float,
) -> sparse.csr_matrix:
    raw = summary["gc_fraction"].to_numpy(dtype=np.float64)
    imputed = np.where(np.isfinite(raw), raw, gc_mean)
    standardized = (imputed - gc_mean) / gc_std
    kmer_matrix = design.matrix[
        summary["kmer_6_code"].to_numpy(dtype=np.int32)
    ]
    return sparse.hstack(
        [kmer_matrix, sparse.csr_matrix(standardized[:, None])],
        format="csr",
    )


def aggregate_gc_binary_likelihood(
    counts: KmerGCCounts,
    design: HierarchicalKmerDesign,
    *,
    contigs: Iterable[str] | None = None,
    balance_classes: bool = True,
) -> AggregateGCBinaryLikelihood:
    """Create exact category/GC/class rows and fit GC scaling on train only."""

    if counts.split_name != "train":
        raise ValueError("training aggregate requires train counts")
    summary = _joint_summary(counts, contigs=contigs)
    gc_mean, gc_std = _gc_scaler(summary)
    population = int(summary["total_positions"].sum())
    positive_population = int(summary["positive_positions"].sum())
    negative_population = population - positive_population
    if population <= 0 or positive_population <= 0 or negative_population <= 0:
        raise ValueError("selected population must contain both classes")

    tuple_indices = np.repeat(np.arange(len(summary), dtype=np.int64), 2)
    targets = np.tile(np.array([0, 1], dtype=np.int8), len(summary))
    population_counts = np.column_stack(
        (
            summary["negative_positions"].to_numpy(dtype=np.int64),
            summary["positive_positions"].to_numpy(dtype=np.int64),
        )
    ).reshape(-1)
    keep = population_counts > 0
    tuple_indices = tuple_indices[keep]
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

    base_rows = summary.iloc[tuple_indices].reset_index(drop=True)
    rows = base_rows.assign(
        target=targets,
        population_count=population_counts,
        class_multiplier=multipliers[targets],
        sample_weight=weights,
    )
    tuple_matrix = _matrix_for_summary(
        summary, design, gc_mean=gc_mean, gc_std=gc_std
    )
    matrix = tuple_matrix[tuple_indices]
    if int(rows["population_count"].sum()) != population:
        raise AssertionError("aggregate rows do not preserve the population")
    if balance_classes:
        sums = rows.groupby("target")["sample_weight"].sum()
        if not np.isclose(sums.loc[0], sums.loc[1]):
            raise AssertionError("balanced class weights must have equal sums")
    return AggregateGCBinaryLikelihood(
        matrix=matrix,
        target=targets,
        sample_weight=weights,
        rows=rows,
        gc_mean=gc_mean,
        gc_std=gc_std,
    )


def _coefficient_table(
    estimator: LogisticRegression,
    design: HierarchicalKmerDesign,
) -> pd.DataFrame:
    table = design.feature_table.copy()
    table["feature_group"] = table["level_k"].map(lambda value: f"{int(value)}-mer")
    gc_row = pd.DataFrame(
        [
            {
                "feature_index": len(table),
                "level_k": pd.NA,
                "category_code": pd.NA,
                "category": "GC / valid ACGT in [-100,+100]",
                "feature_name": "gc201_z",
                "feature_group": "GC201",
            }
        ]
    )
    table = pd.concat([table, gc_row], ignore_index=True)
    table["coefficient"] = estimator.coef_[0]
    table["absolute_coefficient"] = np.abs(estimator.coef_[0])
    return table


def _fit_from_aggregate(
    aggregate: AggregateGCBinaryLikelihood,
    design: HierarchicalKmerDesign,
    *,
    C: float,
    config: GCWindowLogisticConfig,
) -> GCWindowLogisticFit:
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
    coefficients = _coefficient_table(estimator, design)
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
                "features": aggregate.matrix.shape[1],
                "represented_positions": int(aggregate.rows["population_count"].sum()),
                "represented_positives": int(
                    aggregate.rows.loc[
                        aggregate.rows["target"] == 1, "population_count"
                    ].sum()
                ),
                "gc_mean": aggregate.gc_mean,
                "gc_std": aggregate.gc_std,
                "gc_coefficient": float(estimator.coef_[0, -1]),
                "intercept": float(estimator.intercept_[0]),
                "warnings": " | ".join(messages),
            }
        ]
    )
    return GCWindowLogisticFit(
        estimator=estimator,
        design=design,
        aggregate=aggregate,
        coefficient_table=coefficients,
        fit_report=fit_report,
    )


def fit_gc_logistic(
    counts: KmerGCCounts,
    *,
    C: float,
    config: GCWindowLogisticConfig = GCWindowLogisticConfig(),
    contigs: Iterable[str] | None = None,
    design: HierarchicalKmerDesign | None = None,
) -> GCWindowLogisticFit:
    """Fit one explicit L2 candidate; this is an actual training call."""

    if counts.split_name != "train":
        raise ValueError("fit_gc_logistic requires train counts")
    if C not in config.C_grid:
        raise ValueError("C must belong to the preregistered grid")
    design = design or hierarchical_kmer_design(config)
    aggregate = aggregate_gc_binary_likelihood(
        counts,
        design,
        contigs=contigs,
        balance_classes=config.balance_classes,
    )
    return _fit_from_aggregate(aggregate, design, C=C, config=config)


def _average_precision_from_rows(
    total: np.ndarray,
    positive: np.ndarray,
    scores: np.ndarray,
    *,
    include_curve: bool,
) -> tuple[float, pd.DataFrame]:
    total = np.asarray(total, dtype=np.int64)
    positive = np.asarray(positive, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if not (total.shape == positive.shape == scores.shape):
        raise ValueError("metric arrays must have identical shapes")
    positive_population = int(positive.sum())
    if positive_population == 0:
        return 0.0, pd.DataFrame(
            columns=["threshold", "true_positive", "predicted_positive", "precision", "recall"]
        )
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_total = total[order]
    sorted_positive = positive[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    group_total = np.add.reduceat(sorted_total, starts)
    group_positive = np.add.reduceat(sorted_positive, starts)
    cumulative_total = np.cumsum(group_total)
    cumulative_positive = np.cumsum(group_positive)
    precision = cumulative_positive / cumulative_total
    recall = cumulative_positive / positive_population
    average_precision = float(
        np.sum((group_positive / positive_population) * precision)
    )
    if not include_curve:
        return average_precision, pd.DataFrame()
    curve = pd.DataFrame(
        {
            "threshold": sorted_scores[starts],
            "true_positive": cumulative_positive,
            "predicted_positive": cumulative_total,
            "precision": precision,
            "recall": recall,
        }
    )
    return average_precision, curve


def _evaluate_segment(
    counts: KmerGCCounts,
    fit: GCWindowLogisticFit,
    *,
    contigs: Iterable[str] | None,
    variant: str,
    include_curve: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = _normalise_contigs(contigs)
    summary = _joint_summary(counts, contigs=selected)
    matrix = _matrix_for_summary(
        summary,
        fit.design,
        gc_mean=fit.aggregate.gc_mean,
        gc_std=fit.aggregate.gc_std,
    )
    scores = fit.estimator.decision_function(matrix).astype(np.float64)
    total = summary["total_positions"].to_numpy(dtype=np.int64)
    positive = summary["positive_positions"].to_numpy(dtype=np.int64)
    ap, curve = _average_precision_from_rows(
        total, positive, scores, include_curve=include_curve
    )

    score_table = summary[
        ["kmer_6_code", "gc_count", "valid_count"]
    ].copy()
    score_table["score"] = scores
    positive_rows = counts.positives
    if selected is not None:
        positive_rows = positive_rows.loc[
            positive_rows["contig"].astype(str).isin(selected)
        ]
    predictions = positive_rows.merge(
        score_table,
        on=["kmer_6_code", "gc_count", "valid_count"],
        how="left",
        validate="many_to_one",
    )
    if predictions["score"].isna().any():
        raise AssertionError("positive row has no candidate score")
    spearman = _dense_rank_correlation(
        predictions["count"].to_numpy(dtype=np.int64),
        predictions["score"].to_numpy(dtype=np.float64),
    )
    positive_loss = np.logaddexp(0.0, -scores)
    negative_loss = np.logaddexp(0.0, scores)
    population = int(total.sum())
    n_positive = int(positive.sum())
    n_negative = population - n_positive
    population_loss = float(
        (positive @ positive_loss + (total - positive) @ negative_loss) / population
    )
    balanced_loss = float(
        0.5
        * (
            (positive @ positive_loss) / n_positive
            + ((total - positive) @ negative_loss) / n_negative
        )
    )
    segment = "overall" if selected is None else ",".join(selected)
    metric = pd.DataFrame(
        [
            {
                "segment": segment,
                "variant": variant,
                "average_precision": ap,
                "epic_spearman": spearman,
                "population_log_loss": population_loss,
                "balanced_log_loss": balanced_loss,
                "positions": population,
                "positives": n_positive,
                "prevalence": n_positive / population,
                "unique_score_levels": int(np.unique(scores).size),
            }
        ]
    )
    if include_curve:
        curve.insert(0, "variant", variant)
    predictions["variant"] = variant
    return metric, curve, predictions


def evaluate_gc_logistic(
    counts: KmerGCCounts,
    fit: GCWindowLogisticFit,
    *,
    contigs: Iterable[str] | None = None,
    variant: str = "hierarchical_2_4_6mer_lr_plus_gc201",
    include_curve: bool = True,
    include_per_contig: bool = True,
) -> GCWindowEvaluation:
    """Evaluate exact AP and positive-row EPIC Spearman."""

    metric, curve, predictions = _evaluate_segment(
        counts,
        fit,
        contigs=contigs,
        variant=variant,
        include_curve=include_curve,
    )
    per_contig_rows: list[pd.DataFrame] = []
    if include_per_contig:
        selected = _normalise_contigs(contigs)
        available = (
            list(selected)
            if selected is not None
            else sorted(counts.totals["contig"].astype(str).unique())
        )
        for contig in available:
            local, _, _ = _evaluate_segment(
                counts,
                fit,
                contigs=(contig,),
                variant=variant,
                include_curve=False,
            )
            local.insert(0, "contig", contig)
            per_contig_rows.append(local)
    per_contig = (
        pd.concat(per_contig_rows, ignore_index=True)
        if per_contig_rows
        else pd.DataFrame()
    )
    return GCWindowEvaluation(metric, per_contig, curve, predictions)


def cross_validate_gc_logistic(
    train_counts: KmerGCCounts,
    fold_assignment: pd.DataFrame,
    *,
    config: GCWindowLogisticConfig = GCWindowLogisticConfig(),
) -> pd.DataFrame:
    """Run the preregistered train-contig grid; calling this trains models."""

    if train_counts.split_name != "train":
        raise ValueError("cross-validation requires train counts")
    if not {"contig", "fold"}.issubset(fold_assignment.columns):
        raise ValueError("fold_assignment must contain contig and fold")
    expected = set(train_counts.totals["contig"].astype(str))
    actual = set(fold_assignment["contig"].astype(str))
    if expected != actual or fold_assignment["contig"].astype(str).duplicated().any():
        raise ValueError("fold_assignment must contain every train contig exactly once")

    design = hierarchical_kmer_design(config)
    records: list[dict[str, object]] = []
    folds = sorted(fold_assignment["fold"].astype(int).unique())
    for fold in folds:
        validation_contigs = tuple(
            fold_assignment.loc[
                fold_assignment["fold"].astype(int) == fold, "contig"
            ].astype(str)
        )
        fitting_contigs = tuple(sorted(expected - set(validation_contigs)))
        aggregate = aggregate_gc_binary_likelihood(
            train_counts,
            design,
            contigs=fitting_contigs,
            balance_classes=config.balance_classes,
        )
        for C in config.C_grid:
            fitted = _fit_from_aggregate(
                aggregate, design, C=float(C), config=config
            )
            evaluated = evaluate_gc_logistic(
                train_counts,
                fitted,
                contigs=validation_contigs,
                include_curve=False,
                include_per_contig=False,
            )
            records.append(
                {
                    "C": float(C),
                    "fold": int(fold),
                    "validation_contigs": ",".join(validation_contigs),
                    **evaluated.metric_summary.iloc[0].to_dict(),
                    **fitted.fit_report.iloc[0][
                        [
                            "iterations",
                            "converged",
                            "fit_seconds",
                            "aggregate_rows",
                            "gc_mean",
                            "gc_std",
                            "gc_coefficient",
                            "warnings",
                        ]
                    ].to_dict(),
                }
            )
    return pd.DataFrame(records).sort_values(["C", "fold"]).reset_index(drop=True)


def gc_enrichment_table(
    counts: KmerGCCounts,
    *,
    bin_width: float = 0.05,
) -> pd.DataFrame:
    """Describe train-only target enrichment by fixed GC bins."""

    if not 0 < bin_width <= 0.5 or not np.isclose(1 / bin_width, round(1 / bin_width)):
        raise ValueError("bin_width must divide [0, 1] into equal bins")
    summary = _joint_summary(counts)
    edges = np.linspace(0.0, 1.0, int(round(1 / bin_width)) + 1)
    observed = summary.loc[summary["gc_fraction"].notna()].copy()
    observed["gc_bin"] = pd.cut(
        observed["gc_fraction"],
        bins=edges,
        include_lowest=True,
        right=True,
    )
    observed["gc_weighted_sum"] = (
        observed["gc_fraction"] * observed["total_positions"]
    )
    grouped = observed.groupby("gc_bin", observed=False).agg(
        positions=("total_positions", "sum"),
        positives=("positive_positions", "sum"),
        gc_weighted_sum=("gc_weighted_sum", "sum"),
    ).reset_index()
    grouped["mean_gc"] = grouped["gc_weighted_sum"] / grouped["positions"]
    grouped = grouped.drop(columns="gc_weighted_sum")
    grouped["positive_rate"] = grouped["positives"] / grouped["positions"]
    prevalence = summary["positive_positions"].sum() / summary["total_positions"].sum()
    grouped["enrichment"] = grouped["positive_rate"] / prevalence
    grouped["gc_bin"] = grouped["gc_bin"].astype(str)
    return grouped


def gc_count_diagnostics(counts: KmerGCCounts) -> pd.DataFrame:
    """Compact integrity and memory report before any model is fitted."""

    total_positions = int(counts.totals["total_positions"].sum())
    missing_positions = int(
        counts.totals.loc[counts.totals["valid_count"] == 0, "total_positions"].sum()
    )
    return pd.DataFrame(
        [
            {
                "split": counts.split_name,
                "aggregate_rows": len(counts.totals),
                "positions": total_positions,
                "positives": len(counts.positives),
                "unique_gc_fractions": int(
                    counts.totals[["gc_count", "valid_count"]].drop_duplicates().shape[0]
                ),
                "zero_valid_positions": missing_positions,
                "zero_valid_fraction": missing_positions / total_positions,
                "totals_memory_mib": float(
                    counts.totals.memory_usage(index=True, deep=True).sum() / 2**20
                ),
                "positives_memory_mib": float(
                    counts.positives.memory_usage(index=True, deep=True).sum() / 2**20
                ),
            }
        ]
    )


__all__ = [
    "AggregateGCBinaryLikelihood",
    "GCWindowEvaluation",
    "GCWindowLogisticConfig",
    "GCWindowLogisticFit",
    "KmerGCCounts",
    "aggregate_gc_binary_likelihood",
    "cross_validate_gc_logistic",
    "evaluate_gc_logistic",
    "fit_gc_logistic",
    "gc_count_diagnostics",
    "gc_enrichment_table",
    "kmer_counts_from_gc_counts",
    "split_kmer_gc_counts",
]
