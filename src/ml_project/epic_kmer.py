"""Exact whole-k-mer lookup models for the EPIC sequence experiments.

The feature at a genomic position is one strand-oriented word whose last base
is at offset zero.  Canonical words are stored as base-4 integer codes and all
words containing an unknown base or a contig edge share one ``N/edge`` code.

The lookup target is binary target prevalence, never mean read count.  Sparse
positive BED rows are combined with exact denominators from every whitelist
position.  Scores use a parent-centred Beta-Binomial posterior mean with the
suffix hierarchy ``8 -> 6 -> 4 -> 2 -> global train prevalence``.  The default
preregistered prior strength is ``1 / train prevalence`` at every level (one
expected positive when centred on global prevalence); train-only
empirical-Bayes and explicit fixed strengths are also supported for later
sensitivity checks and fully reported.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import betaln

from .epic_baseline import (
    _average_precision_from_categories,
    _dense_rank_correlation,
)
from .epic_data import PreparedSplit
from .sequence_context import (
    COMPLEMENT_CODE,
    UPPERCASE_BASE_CODE,
    extract_oriented_context,
)


SUPPORTED_K = (2, 4, 6, 8)
BASES = "ACGT"


def _code_column(k: int) -> str:
    return f"kmer_{k}_code"


def _score_column(k: int) -> str:
    return f"kmer_{k}_score"


def _validate_ks(ks: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(k) for k in ks)
    if not result or result != tuple(sorted(set(result))):
        raise ValueError("ks must be a non-empty, strictly increasing sequence")
    unsupported = sorted(set(result) - set(SUPPORTED_K))
    if unsupported:
        raise ValueError(f"unsupported k values: {unsupported}")
    return result


@dataclass(frozen=True)
class KmerLookupConfig:
    """Settings which can change a whole-k-mer lookup result.

    ``one_expected_positive`` uses the preregistered strength
    ``1 / train_prevalence`` at every level: a prior centred on global
    prevalence contributes exactly one expected positive.  ``empirical_bayes``
    estimates a shared strength at each level from train only.  ``fixed``
    requires one positive strength for every requested k in
    ``fixed_strengths``.
    """

    ks: tuple[int, ...] = SUPPORTED_K
    smoothing: str = "one_expected_positive"
    fixed_strengths: tuple[tuple[int, float], ...] = ()
    strength_bounds: tuple[float, float] = (1e-2, 1e8)
    optimizer_xatol: float = 1e-5
    count_batch_size: int = 2_000_000

    def __post_init__(self) -> None:
        normalized = _validate_ks(self.ks)
        if normalized != self.ks:
            raise ValueError("ks must contain Python integers in increasing order")
        if self.smoothing not in {
            "one_expected_positive",
            "empirical_bayes",
            "fixed",
        }:
            raise ValueError(
                "smoothing must be 'one_expected_positive', "
                "'empirical_bayes' or 'fixed'"
            )
        lower, upper = self.strength_bounds
        if not (0 < lower < upper):
            raise ValueError("strength_bounds must satisfy 0 < lower < upper")
        if self.optimizer_xatol <= 0 or self.count_batch_size <= 0:
            raise ValueError("optimizer_xatol and count_batch_size must be positive")
        supplied = dict(self.fixed_strengths)
        if len(supplied) != len(self.fixed_strengths):
            raise ValueError("fixed_strengths contains duplicate k values")
        if any(k not in self.ks for k in supplied):
            raise ValueError("fixed_strengths contains a k absent from ks")
        if any(not np.isfinite(value) or value <= 0 for value in supplied.values()):
            raise ValueError("every fixed smoothing strength must be finite and positive")
        if self.smoothing == "fixed" and set(supplied) != set(self.ks):
            raise ValueError("fixed smoothing requires a strength for every requested k")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["ks"] = list(self.ks)
        result["fixed_strengths"] = {
            str(k): float(value) for k, value in self.fixed_strengths
        }
        result["window"] = "strand-oriented offsets [-(k-1), ..., 0]"
        result["target"] = "binary count > 0"
        result["backoff"] = "suffix hierarchy ending at offset 0"
        return result


@dataclass(frozen=True)
class KmerCounts:
    """Exact full-population totals and sparse positive k-mer assignments."""

    split_name: str
    ks: tuple[int, ...]
    totals: pd.DataFrame
    positives: pd.DataFrame


@dataclass(frozen=True)
class KmerLookupFit:
    """Fitted train-only lookup tables and their complete diagnostics."""

    config: KmerLookupConfig
    scores_by_k: Mapping[int, np.ndarray]
    global_prevalence: float
    category_summary: pd.DataFrame
    support_summary: pd.DataFrame
    smoothing_report: pd.DataFrame


@dataclass(frozen=True)
class KmerEvaluation:
    """Full validation metrics without expanding the negative population."""

    metric_summary: pd.DataFrame
    per_contig_metrics: pd.DataFrame
    category_summary: pd.DataFrame
    positive_predictions: pd.DataFrame
    precision_recall: pd.DataFrame


def kmer_category_count(k: int) -> int:
    """Number of canonical words plus the combined ``N/edge`` category."""

    _validate_ks((k,))
    return 4**k + 1


def kmer_category_label(k: int, code: int) -> str:
    """Decode one base-4 category code to a human-readable word."""

    category_count = kmer_category_count(k)
    if not isinstance(code, (int, np.integer)) or not 0 <= int(code) < category_count:
        raise ValueError(f"category code for k={k} must be in [0, {category_count})")
    value = int(code)
    if value == category_count - 1:
        return "N/edge"
    letters = ["A"] * k
    for index in range(k - 1, -1, -1):
        value, digit = divmod(value, 4)
        letters[index] = BASES[digit]
    return "".join(letters)


@lru_cache(maxsize=None)
def kmer_category_labels(k: int) -> tuple[str, ...]:
    """All labels in code order; cached because the 8-mer table is sizeable."""

    return tuple(kmer_category_label(k, code) for code in range(kmer_category_count(k)))


def context_kmer_codes(context: np.ndarray) -> np.ndarray:
    """Encode an oriented A/C/G/T/N context matrix as one whole word."""

    values = np.asarray(context)
    if values.ndim != 2 or values.shape[1] not in SUPPORTED_K:
        raise ValueError(f"context must have shape (n, k) with k in {SUPPORTED_K}")
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("context codes must be integers")
    if np.any(values < 0) or np.any(values > 4):
        raise ValueError("context codes must be A=0, C=1, G=2, T=3 or N/edge=4")
    k = values.shape[1]
    canonical = np.where(values < 4, values, 0).astype(np.uint32, copy=False)
    powers = (4 ** np.arange(k - 1, -1, -1)).astype(np.uint32)
    result = canonical @ powers
    result = result.astype(np.int32, copy=False)
    result[np.any(values == 4, axis=1)] = 4**k
    return result


def _uppercase_letters(raw_sequence: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_sequence)
    if raw.ndim != 1 or not len(raw):
        raise ValueError("FASTA sequence must be a non-empty vector")
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("FASTA sequence codes must be integers")
    if np.any(raw < 0) or np.any(raw >= len(UPPERCASE_BASE_CODE)):
        raise ValueError("unknown case-preserving FASTA code")
    return UPPERCASE_BASE_CODE[raw.astype(np.int64, copy=False)]


def _packed_suffix_state(letters: np.ndarray, max_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Pack the last max_k bases and retain an inclusive unknown-base prefix."""

    canonical = np.where(letters < 4, letters, 0).astype(np.uint32, copy=False)
    packed = canonical.copy()
    factor = 4
    for lag in range(1, max_k):
        packed[lag:] += canonical[:-lag] * factor
        factor *= 4
    unknown_prefix = np.cumsum(letters >= 4, dtype=np.uint32)
    return packed, unknown_prefix


def _codes_from_state(
    packed: np.ndarray,
    unknown_prefix: np.ndarray,
    start: int,
    end: int,
    k: int,
) -> np.ndarray:
    """Materialize codes for endpoint coordinates in one half-open slice."""

    if not 0 <= start <= end <= len(packed):
        raise ValueError("requested code slice is outside the contig")
    result = (packed[start:end] & np.uint32(4**k - 1)).astype(np.int32)
    if not len(result):
        return result
    invalid = unknown_prefix[start:end].copy()
    subtract_from = max(start, k)
    if subtract_from < end:
        invalid[subtract_from - start :] -= unknown_prefix[
            subtract_from - k : end - k
        ]
    edge_stop = min(end, k - 1)
    if start < edge_stop:
        invalid[: edge_stop - start] += 1
    result[invalid > 0] = 4**k
    return result


def oriented_kmer_codes(raw_sequence: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the whole-k-mer category at every coordinate on both strands."""

    _validate_ks((k,))
    letters = _uppercase_letters(raw_sequence)
    plus_state = _packed_suffix_state(letters, k)
    plus = _codes_from_state(*plus_state, 0, len(letters), k)
    reverse_complement = COMPLEMENT_CODE[letters[::-1]]
    minus_state = _packed_suffix_state(reverse_complement, k)
    minus = _codes_from_state(*minus_state, 0, len(letters), k)[::-1].copy()
    return plus, minus


def _interval_counts(
    packed: np.ndarray,
    unknown_prefix: np.ndarray,
    intervals: Sequence[tuple[int, int]],
    k: int,
    batch_size: int,
) -> np.ndarray:
    counts = np.zeros(kmer_category_count(k), dtype=np.int64)
    pieces: list[np.ndarray] = []
    buffered = 0

    def flush() -> None:
        nonlocal pieces, buffered
        if not pieces:
            return
        values = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        local = np.bincount(values, minlength=len(counts))
        counts[:] += local
        pieces = []
        buffered = 0

    for interval_start, interval_end in intervals:
        cursor = int(interval_start)
        stop = int(interval_end)
        if not 0 <= cursor < stop <= len(packed):
            raise ValueError("whitelist interval is outside its FASTA contig")
        while cursor < stop:
            take = min(stop - cursor, batch_size - buffered)
            pieces.append(
                _codes_from_state(
                    packed,
                    unknown_prefix,
                    cursor,
                    cursor + take,
                    k,
                )
            )
            cursor += take
            buffered += take
            if buffered == batch_size:
                flush()
    flush()
    return counts


def _positive_codes(
    sequences: Mapping[str, np.ndarray],
    positives: pd.DataFrame,
    ks: tuple[int, ...],
) -> pd.DataFrame:
    result = positives.copy()
    max_k = max(ks)
    context = extract_oriented_context(
        sequences,
        result,
        offsets=range(-(max_k - 1), 1),
    )
    for k in ks:
        result[_code_column(k)] = context_kmer_codes(context[:, -k:])
    return result


def split_kmer_counts(
    split: PreparedSplit,
    sequences: Mapping[str, np.ndarray],
    split_name: str,
    *,
    ks: Sequence[int] = SUPPORTED_K,
    batch_size: int = 2_000_000,
) -> KmerCounts:
    """Count all whitelist positions and assign every positive exactly once."""

    normalized_ks = _validate_ks(ks)
    if split_name not in {"train", "validation"}:
        raise ValueError("k-mer lookup counts require train or validation labels")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    intervals = split.intervals(split_name)
    positives = split.positive_targets(split_name)
    if positives is None:
        raise ValueError(f"{split_name} positives are unavailable")
    missing = sorted(set(intervals["contig"].astype(str)) - set(sequences))
    if missing:
        raise ValueError("split contigs missing from FASTA: " + ", ".join(missing))

    records: list[pd.DataFrame] = []
    max_k = max(normalized_ks)
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
            packed, unknown_prefix = _packed_suffix_state(oriented, max_k)
            for k in normalized_ks:
                counts = _interval_counts(
                    packed,
                    unknown_prefix,
                    oriented_ranges,
                    k,
                    batch_size,
                )
                present = np.flatnonzero(counts)
                records.append(
                    pd.DataFrame(
                        {
                            "k": k,
                            "contig": contig,
                            "strand": strand,
                            "category_code": present.astype(np.int32),
                            "total_positions": counts[present],
                        }
                    )
                )

    totals = pd.concat(records, ignore_index=True) if records else pd.DataFrame(
        columns=["k", "contig", "strand", "category_code", "total_positions"]
    )
    totals = totals.astype(
        {"k": "int8", "category_code": "int32", "total_positions": "int64"}
    )
    positive_codes = _positive_codes(sequences, positives, normalized_ks)

    expected_population = int((intervals["end"] - intervals["start"]).sum())
    by_k = totals.groupby("k")["total_positions"].sum()
    for k in normalized_ks:
        if int(by_k.get(k, 0)) != expected_population:
            raise AssertionError(f"k={k} counts do not cover the complete split")
        total_by_code = (
            totals.loc[totals["k"] == k]
            .groupby("category_code")["total_positions"]
            .sum()
        )
        positive_by_code = positive_codes[_code_column(k)].value_counts()
        aligned = total_by_code.reindex(positive_by_code.index, fill_value=0)
        if np.any(positive_by_code.to_numpy() > aligned.to_numpy()):
            raise AssertionError(f"k={k} positive count exceeds its denominator")
    return KmerCounts(split_name, normalized_ks, totals, positive_codes)


def _aggregate_arrays(
    counts: KmerCounts,
    k: int,
    *,
    contig: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    size = kmer_category_count(k)
    total_rows = counts.totals.loc[counts.totals["k"] == k]
    positives = counts.positives
    if contig is not None:
        total_rows = total_rows.loc[total_rows["contig"].astype(str) == str(contig)]
        positives = positives.loc[positives["contig"].astype(str) == str(contig)]
    total = (
        total_rows.groupby("category_code")["total_positions"]
        .sum()
        .reindex(range(size), fill_value=0)
        .to_numpy(dtype=np.int64)
    )
    positive = (
        positives[_code_column(k)]
        .value_counts()
        .reindex(range(size), fill_value=0)
        .to_numpy(dtype=np.int64)
    )
    return total, positive


def kmer_category_summary(counts: KmerCounts, k: int) -> pd.DataFrame:
    """Dense category table with exact binary rates and enrichments."""

    if k not in counts.ks:
        raise ValueError(f"k={k} is absent from this count bundle")
    total, positive = _aggregate_arrays(counts, k)
    population = int(total.sum())
    positive_population = int(positive.sum())
    if population <= 0 or positive_population <= 0:
        raise ValueError("category summary requires a non-empty split with positives")
    rate = np.divide(
        positive,
        total,
        out=np.zeros_like(positive, dtype=np.float64),
        where=total > 0,
    )
    prevalence = positive_population / population
    return pd.DataFrame(
        {
            "k": k,
            "category_code": np.arange(len(total), dtype=np.int32),
            "category": kmer_category_labels(k),
            "total_positions": total,
            "positive_positions": positive,
            "negative_positions": total - positive,
            "positive_rate": rate,
            "enrichment": rate / prevalence,
            "observed": total > 0,
        }
    )


def _parent_codes(k: int, parent_k: int | None) -> np.ndarray:
    size = kmer_category_count(k)
    if parent_k is None:
        return np.full(size, -1, dtype=np.int32)
    canonical = np.arange(size - 1, dtype=np.int32) & (4**parent_k - 1)
    # N/edge has no unique suffix.  -1 records that its prior is global.
    return np.r_[canonical, np.int32(-1)]


def _estimate_beta_strength(
    total: np.ndarray,
    positive: np.ndarray,
    prior_mean: np.ndarray,
    bounds: tuple[float, float],
    xatol: float,
) -> tuple[float, float, bool, str]:
    observed = total > 0
    n = total[observed].astype(np.float64)
    y = positive[observed].astype(np.float64)
    q = np.clip(prior_mean[observed].astype(np.float64), 1e-12, 1 - 1e-12)
    log_bounds = tuple(np.log(bounds))

    def negative_log_marginal(log_strength: float) -> float:
        strength = float(np.exp(log_strength))
        alpha = strength * q
        beta = strength * (1 - q)
        value = np.sum(
            betaln(y + alpha, n - y + beta) - betaln(alpha, beta)
        )
        return -float(value)

    result = minimize_scalar(
        negative_log_marginal,
        method="bounded",
        bounds=log_bounds,
        options={"xatol": xatol},
    )
    strength = float(np.exp(result.x))
    return strength, float(-result.fun), bool(result.success), str(result.message)


def fit_kmer_lookup(
    train_counts: KmerCounts,
    config: KmerLookupConfig = KmerLookupConfig(),
) -> KmerLookupFit:
    """Fit suffix-backoff posterior rates from binary train counts only."""

    if train_counts.split_name != "train":
        raise ValueError("fit_kmer_lookup requires a train count bundle")
    if tuple(train_counts.ks) != tuple(config.ks):
        raise ValueError("count bundle ks differ from config ks")
    first_total, first_positive = _aggregate_arrays(train_counts, config.ks[0])
    population = int(first_total.sum())
    positive_population = int(first_positive.sum())
    if population <= 0 or not 0 < positive_population < population:
        raise ValueError("train population must contain both binary classes")
    global_prevalence = positive_population / population
    fixed = dict(config.fixed_strengths)

    scores_by_k: dict[int, np.ndarray] = {}
    summaries: list[pd.DataFrame] = []
    reports: list[dict[str, object]] = []
    supports: list[dict[str, object]] = []
    parent_k: int | None = None
    for k in config.ks:
        summary = kmer_category_summary(train_counts, k)
        total = summary["total_positions"].to_numpy(dtype=np.int64)
        positive = summary["positive_positions"].to_numpy(dtype=np.int64)
        parent_codes = _parent_codes(k, parent_k)
        prior = np.full(len(total), global_prevalence, dtype=np.float64)
        if parent_k is not None:
            prior[:-1] = scores_by_k[parent_k][parent_codes[:-1]]
        if config.smoothing == "fixed":
            strength = float(fixed[k])
            log_marginal = np.nan
            optimizer_success = True
            optimizer_message = "fixed by configuration"
        elif config.smoothing == "one_expected_positive":
            strength = 1.0 / global_prevalence
            log_marginal = np.nan
            optimizer_success = True
            optimizer_message = (
                "preregistered tau = 1 / train prevalence; one expected positive "
                "at the global prior mean"
            )
        else:
            strength, log_marginal, optimizer_success, optimizer_message = (
                _estimate_beta_strength(
                    total,
                    positive,
                    prior,
                    config.strength_bounds,
                    config.optimizer_xatol,
                )
            )
            if not optimizer_success or not np.isfinite(strength):
                raise RuntimeError(f"empirical-Bayes fit failed for k={k}: {optimizer_message}")
        scores = (positive + strength * prior) / (total + strength)
        scores_by_k[k] = scores
        data_weight = total / (total + strength)

        summary["parent_k"] = "global" if parent_k is None else str(parent_k)
        summary["parent_category_code"] = parent_codes
        summary["prior_score"] = prior
        summary["smoothing_strength"] = strength
        summary["prior_positive_pseudocount"] = strength * prior
        summary["prior_negative_pseudocount"] = strength * (1 - prior)
        summary["data_weight"] = data_weight
        summary["posterior_score"] = scores
        summaries.append(summary)
        lower, upper = config.strength_bounds
        reports.append(
            {
                "k": k,
                "parent": (
                    "global prevalence"
                    if parent_k is None
                    else f"{parent_k}-mer suffix; N/edge uses global prevalence"
                ),
                "method": config.smoothing,
                "strength": strength,
                "strength_lower_bound": lower,
                "strength_upper_bound": upper,
                "at_lower_bound": bool(strength <= lower * 1.001),
                "at_upper_bound": bool(strength >= upper / 1.001),
                "log_marginal_likelihood": log_marginal,
                "optimizer_success": optimizer_success,
                "optimizer_message": optimizer_message,
                "formula": "(positive + strength * parent_score) / (total + strength)",
            }
        )
        observed_support = total[total > 0]
        supports.append(
            {
                "k": k,
                "categories": len(total),
                "observed_categories": int(np.count_nonzero(total)),
                "unseen_categories": int(np.count_nonzero(total == 0)),
                "categories_with_positive": int(np.count_nonzero(positive)),
                "zero_positive_categories": int(np.count_nonzero((total > 0) & (positive == 0))),
                "total_positions": int(total.sum()),
                "positive_positions": int(positive.sum()),
                "min_observed_support": int(observed_support.min()),
                "median_observed_support": float(np.median(observed_support)),
                "p10_observed_support": float(np.quantile(observed_support, 0.10)),
                "p90_observed_support": float(np.quantile(observed_support, 0.90)),
                "max_observed_support": int(observed_support.max()),
                "smoothing_strength": strength,
            }
        )
        parent_k = k

    return KmerLookupFit(
        config=config,
        scores_by_k=scores_by_k,
        global_prevalence=global_prevalence,
        category_summary=pd.concat(summaries, ignore_index=True),
        support_summary=pd.DataFrame(supports),
        smoothing_report=pd.DataFrame(reports),
    )


def _segment_metrics(
    validation_counts: KmerCounts,
    fit: KmerLookupFit,
    *,
    contig: str | None,
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    positives = validation_counts.positives
    if contig is not None:
        positives = positives.loc[positives["contig"].astype(str) == str(contig)]
    rows: list[dict[str, object]] = []
    curves: list[pd.DataFrame] = []
    first_total, first_positive = _aggregate_arrays(
        validation_counts, fit.config.ks[0], contig=contig
    )
    population = int(first_total.sum())
    n_positive = int(first_positive.sum())
    prevalence = n_positive / population
    segment_value = "overall" if contig is None else str(contig)
    rows.append(
        {
            "segment": "overall" if contig is None else "contig",
            "segment_value": segment_value,
            "k": 0,
            "variant": "constant_reference",
            "average_precision": prevalence,
            "epic_spearman": 0.0,
            "positions": population,
            "positives": n_positive,
            "prevalence": prevalence,
            "unique_score_levels": 1,
            "metric_note": (
                "constant score: AP equals prevalence; Spearman is 0 by the "
                "explicit no-ranking convention"
            ),
        }
    )
    for k in fit.config.ks:
        total, positive = _aggregate_arrays(validation_counts, k, contig=contig)
        scores = fit.scores_by_k[k]
        # Unseen validation categories have zero mass and must not introduce
        # empty leading thresholds into the exact aggregated PR curve.
        represented = total > 0
        average_precision, curve = _average_precision_from_categories(
            total[represented], positive[represented], scores[represented]
        )
        positive_scores = scores[
            positives[_code_column(k)].to_numpy(dtype=np.int32)
        ]
        spearman = _dense_rank_correlation(
            positives["count"].to_numpy(dtype=np.int64), positive_scores
        )
        variant = f"{k}mer_lookup"
        rows.append(
            {
                "segment": "overall" if contig is None else "contig",
                "segment_value": segment_value,
                "k": k,
                "variant": variant,
                "average_precision": average_precision,
                "epic_spearman": spearman,
                "positions": int(total.sum()),
                "positives": int(positive.sum()),
                "prevalence": prevalence,
                "unique_score_levels": int(np.unique(scores[total > 0]).size),
                "metric_note": "exact full-whitelist evaluation with raw train-only scores",
            }
        )
        if contig is None:
            curve.insert(0, "k", k)
            curve.insert(1, "variant", variant)
            curves.append(curve)
    return rows, curves


def evaluate_kmer_lookup(
    validation_counts: KmerCounts,
    fit: KmerLookupFit,
) -> KmerEvaluation:
    """Evaluate exact AP on all validation rows and Spearman on positives."""

    if validation_counts.split_name != "validation":
        raise ValueError("evaluate_kmer_lookup requires a validation count bundle")
    if tuple(validation_counts.ks) != tuple(fit.config.ks):
        raise ValueError("validation ks differ from the fitted config")
    overall_rows, curves = _segment_metrics(validation_counts, fit, contig=None)
    contig_rows: list[dict[str, object]] = []
    for contig in sorted(validation_counts.totals["contig"].astype(str).unique()):
        local_rows, _ = _segment_metrics(validation_counts, fit, contig=contig)
        contig_rows.extend(local_rows)

    predictions = validation_counts.positives.copy()
    validation_summaries: list[pd.DataFrame] = []
    for k in fit.config.ks:
        scores = fit.scores_by_k[k]
        predictions[_score_column(k)] = scores[
            predictions[_code_column(k)].to_numpy(dtype=np.int32)
        ]
        summary = kmer_category_summary(validation_counts, k)
        train = fit.category_summary.loc[
            fit.category_summary["k"] == k,
            [
                "category_code",
                "total_positions",
                "positive_positions",
                "prior_score",
                "data_weight",
                "posterior_score",
            ],
        ].rename(
            columns={
                "total_positions": "train_total_positions",
                "positive_positions": "train_positive_positions",
                "prior_score": "train_prior_score",
                "data_weight": "train_data_weight",
                "posterior_score": "candidate_score",
            }
        )
        summary = summary.merge(train, on="category_code", how="left", validate="one_to_one")
        validation_summaries.append(summary)

    return KmerEvaluation(
        metric_summary=pd.DataFrame(overall_rows),
        per_contig_metrics=pd.DataFrame(contig_rows),
        category_summary=pd.concat(validation_summaries, ignore_index=True),
        positive_predictions=predictions,
        precision_recall=pd.concat(curves, ignore_index=True),
    )


__all__ = [
    "BASES",
    "SUPPORTED_K",
    "KmerCounts",
    "KmerEvaluation",
    "KmerLookupConfig",
    "KmerLookupFit",
    "context_kmer_codes",
    "evaluate_kmer_lookup",
    "fit_kmer_lookup",
    "kmer_category_count",
    "kmer_category_label",
    "kmer_category_labels",
    "kmer_category_summary",
    "oriented_kmer_codes",
    "split_kmer_counts",
]
