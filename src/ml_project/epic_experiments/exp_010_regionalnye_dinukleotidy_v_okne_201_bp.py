"""EXP-010: screen one regional dinucleotide O/E201 feature at a time.

Each candidate is exactly EXP-009 plus seven reference-coded bins for one
dinucleotide.  Candidates never accumulate.  Importing this module performs no
counting or fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
import warnings
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from ml_project.epic_baseline import _dense_rank_correlation
from ml_project.epic_cpg_logistic import (
    CPG_DIAGNOSTIC_EDGES,
    CPG_DIAGNOSTIC_LABELS,
    CPG_REFERENCE_BIN,
    CpGLogisticConfig,
    _add_count_part,
    _collapse_count_parts,
    _composition_prefixes,
    _matrix_for_summary as _champion_matrix,
    _oriented_letters,
    _sparse_window_statistics,
    _window_statistics,
    cpg_bin_codes,
)
from ml_project.epic_data import PreparedSplit
from ml_project.epic_gc_binned_logistic import GCWindowEvaluation, gc_bin_codes
from ml_project.epic_gc_logistic import _average_precision_from_rows, _normalise_contigs
from ml_project.epic_kmer import _codes_from_state, _packed_suffix_state, context_kmer_codes
from ml_project.epic_kmer_logistic import HierarchicalKmerDesign, hierarchical_kmer_design
from ml_project.sequence_context import extract_oriented_context


EXPERIMENT_ID = "EXP-010"
CHAMPION_EXPERIMENT = "EXP-009"
CHAMPION_RUN = "run_001"
HYPOTHESIS = (
    "Другие соседние пары букв содержат дополнительный сигнал о старте "
    "транскрипции сверх локальных k-mer, GC201 и CpG O/E201"
)
CHANGED_VARIABLE = "EXP-009 + один региональный dinucleotide O/E201 bin feature"
DINUCLEOTIDE_PAIRS = (
    "AA", "TT", "TA", "GG", "CC", "AT", "GC", "CA", "TG",
    "AG", "CT", "GA", "TC", "AC", "GT",
)
BASE_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}
DINUCLEOTIDE_EDGES = CPG_DIAGNOSTIC_EDGES
DINUCLEOTIDE_LABELS = (*CPG_DIAGNOSTIC_LABELS[:-1], "missing left/right")
DINUCLEOTIDE_REFERENCE_BIN = CPG_REFERENCE_BIN
FIXED_C = 1e-4
FEATURE_READY = True


@dataclass(frozen=True)
class RegionalDinucleotideConfig(CpGLogisticConfig):
    """Frozen settings for the 15-way EXP-010 screen."""

    candidate_C: float = FIXED_C

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.candidate_C != FIXED_C:
            raise ValueError("EXP-010 fixes C=0.0001 from champion EXP-009")

    def to_dict(self) -> dict[str, object]:
        result = super().to_dict()
        result.update({
            "candidate_pairs": list(DINUCLEOTIDE_PAIRS),
            "candidate_C": self.candidate_C,
            "dinucleotide_oe_edges": list(DINUCLEOTIDE_EDGES),
            "dinucleotide_reference_bin": DINUCLEOTIDE_REFERENCE_BIN,
            "screening_rule": "one pair at a time; features never accumulate",
            "planned_candidate_fits": len(DINUCLEOTIDE_PAIRS) * self.cv_folds,
        })
        return result


@dataclass(frozen=True)
class KmerDinucleotideCounts:
    split_name: str
    pair: str
    window_start: int
    window_end: int
    totals: pd.DataFrame
    positives: pd.DataFrame


@dataclass(frozen=True)
class DinucleotideAggregate:
    matrix: sparse.csr_matrix
    target: np.ndarray
    sample_weight: np.ndarray
    rows: pd.DataFrame


@dataclass(frozen=True)
class DinucleotideFit:
    estimator: LogisticRegression
    design: HierarchicalKmerDesign
    pair: str
    aggregate: DinucleotideAggregate
    coefficient_table: pd.DataFrame
    fit_report: pd.DataFrame


def _validate_pair(pair: str) -> str:
    value = str(pair).upper()
    if value not in DINUCLEOTIDE_PAIRS:
        raise ValueError(f"pair must be one of {DINUCLEOTIDE_PAIRS}")
    return value


def feature_contract() -> dict[str, Any]:
    return {
        "champion": f"{CHAMPION_EXPERIMENT}/{CHAMPION_RUN}",
        "candidates_in_priority_order": DINUCLEOTIDE_PAIRS,
        "formula": "pair_count * valid_count / (left_count * right_count)",
        "window": "strand-oriented inclusive [-100,+100]",
        "bins": DINUCLEOTIDE_LABELS,
        "reference_bin": DINUCLEOTIDE_LABELS[DINUCLEOTIDE_REFERENCE_BIN],
        "candidate_columns": 7,
        "total_model_columns": 4393,
        "aggregation_key": "contig x strand x 6-mer x GC-bin x CpG-bin x pair-bin",
        "missing_edge_policy": "clipped window; missing bin if denominator is zero",
        "leakage_check": "fixed edges; folds and validation inherited from EXP-009",
        "candidate_C": FIXED_C,
        "planned_fits": 45,
        "accumulation": "forbidden",
    }


def dinucleotide_observed_expected(left_count: Any, right_count: Any,
                                    pair_count: Any, valid_count: Any) -> np.ndarray:
    left = np.asarray(left_count, dtype=np.float64)
    right = np.asarray(right_count, dtype=np.float64)
    observed = np.asarray(pair_count, dtype=np.float64)
    valid = np.asarray(valid_count, dtype=np.float64)
    if not (left.shape == right.shape == observed.shape == valid.shape):
        raise ValueError("dinucleotide statistic arrays must have identical shapes")
    denominator = left * right
    return np.divide(observed * valid, denominator,
                     out=np.full(left.shape, np.nan),
                     where=(denominator > 0) & (valid > 0))


def dinucleotide_bin_codes(left_count: Any, right_count: Any,
                           pair_count: Any, valid_count: Any) -> np.ndarray:
    values = dinucleotide_observed_expected(
        left_count, right_count, pair_count, valid_count)
    result = np.searchsorted(np.asarray(DINUCLEOTIDE_EDGES), values,
                             side="right").astype(np.int8)
    result[~np.isfinite(values)] = len(DINUCLEOTIDE_LABELS) - 1
    return result


def dinucleotide_feature_table(pair: str) -> pd.DataFrame:
    pair = _validate_pair(pair)
    rows = []
    for code, label in enumerate(DINUCLEOTIDE_LABELS):
        if code == DINUCLEOTIDE_REFERENCE_BIN:
            continue
        rows.append({"bin_code": code, "bin_label": label,
                     "feature_index_within_pair": len(rows),
                     "feature_name": f"{pair}_oe201_bin={label}",
                     "feature_group": f"{pair} O/E201 bin"})
    return pd.DataFrame(rows)


def _dinucleotide_matrix(codes: np.ndarray, pair: str) -> sparse.csr_matrix:
    codes = np.asarray(codes, dtype=np.int8)
    if ((codes < 0) | (codes >= len(DINUCLEOTIDE_LABELS))).any():
        raise ValueError("dinucleotide bin outside the fixed vocabulary")
    table = dinucleotide_feature_table(pair)
    mapping = np.full(len(DINUCLEOTIDE_LABELS), -1, dtype=np.int8)
    mapping[table["bin_code"].to_numpy(dtype=np.int8)] = table[
        "feature_index_within_pair"].to_numpy(dtype=np.int8)
    columns = mapping[codes]
    active = columns >= 0
    return sparse.csr_matrix((np.ones(active.sum()),
                              (np.flatnonzero(active), columns[active])),
                             shape=(len(codes), len(table)))


def _pair_prefixes(letters: np.ndarray, pair: str):
    left, right = map(BASE_CODE.__getitem__, pair)
    def prefix(values: np.ndarray) -> np.ndarray:
        out = np.empty(len(values) + 1, dtype=np.int32)
        out[0] = 0
        np.cumsum(values, dtype=np.int32, out=out[1:])
        return out
    return (prefix(letters == left), prefix(letters == right),
            prefix((letters[:-1] == left) & (letters[1:] == right)),
            prefix(letters < 4))


def _pair_window_statistics(prefixes, coordinates: np.ndarray, *,
                            sequence_length: int, window_start: int,
                            window_end: int):
    coordinates = np.asarray(coordinates, dtype=np.int64)
    lo = np.maximum(0, coordinates + window_start)
    hi = np.minimum(sequence_length, coordinates + window_end + 1)
    left, right, pairs, valid = prefixes
    pair_hi = np.maximum(lo, hi - 1)
    return ((left[hi] - left[lo]).astype(np.int16),
            (right[hi] - right[lo]).astype(np.int16),
            (pairs[pair_hi] - pairs[lo]).astype(np.int16),
            (valid[hi] - valid[lo]).astype(np.int16))


def _feature_keys_for_batch(letters: np.ndarray, start: int, end: int,
                            pair: str, config: RegionalDinucleotideConfig) -> np.ndarray:
    halo_start = max(0, start + min(config.window_start, -(config.max_k - 1)))
    halo_end = min(len(letters), end + config.window_end)
    local = letters[halo_start:halo_end]
    local_start, local_end = start - halo_start, end - halo_start
    packed, unknown = _packed_suffix_state(local, config.max_k)
    kmer = _codes_from_state(packed, unknown, local_start, local_end,
                             config.max_k).astype(np.int64, copy=False)
    coordinates = np.arange(local_start, local_end, dtype=np.int64)
    c, g, cpg, valid = _window_statistics(
        _composition_prefixes(local), coordinates, sequence_length=len(local),
        window_start=config.window_start, window_end=config.window_end)
    left, right, observed, pair_valid = _pair_window_statistics(
        _pair_prefixes(local, pair), coordinates, sequence_length=len(local),
        window_start=config.window_start, window_end=config.window_end)
    if not np.array_equal(valid, pair_valid):
        raise AssertionError("window validity counters disagree")
    gc_code = gc_bin_codes(c + g, valid)
    cpg_code = cpg_bin_codes(c, g, cpg, valid)
    pair_code = dinucleotide_bin_codes(left, right, observed, valid)
    keys = ((kmer * 9 + gc_code) * 8 + cpg_code) * 8 + pair_code
    return keys


def _count_oriented_ranges(letters: np.ndarray, ranges: Sequence[tuple[int, int]],
                           pair: str, config: RegionalDinucleotideConfig) -> pd.DataFrame:
    levels = []
    for interval_start, interval_end in ranges:
        cursor, stop = int(interval_start), int(interval_end)
        if not 0 <= cursor < stop <= len(letters):
            raise ValueError("whitelist interval is outside FASTA")
        while cursor < stop:
            end = min(stop, cursor + config.count_batch_size)
            keys, counts = np.unique(
                _feature_keys_for_batch(letters, cursor, end, pair, config),
                return_counts=True)
            _add_count_part(levels, keys.astype(np.int64), counts.astype(np.int64))
            cursor = end
    packed, counts = _collapse_count_parts(levels)
    remainder = packed.copy()
    pair_code = (remainder % 8).astype(np.int8); remainder //= 8
    cpg_code = (remainder % 8).astype(np.int8); remainder //= 8
    gc_code = (remainder % 9).astype(np.int8); remainder //= 9
    return pd.DataFrame({"kmer_6_code": remainder.astype(np.int32),
                         "gc_bin_code": gc_code, "cpg_bin_code": cpg_code,
                         "dinucleotide_bin_code": pair_code,
                         "total_positions": counts.astype(np.int64)})


def _sparse_pair_statistics(letters: np.ndarray, coordinates: np.ndarray,
                            pair: str, config: RegionalDinucleotideConfig,
                            batch_size: int = 20_000):
    left_code, right_code = map(BASE_CODE.__getitem__, pair)
    coordinates = np.asarray(coordinates, dtype=np.int64)
    result = [np.empty(len(coordinates), dtype=np.int16) for _ in range(4)]
    offsets = np.arange(config.window_start, config.window_end + 1)
    for start in range(0, len(coordinates), batch_size):
        end = min(len(coordinates), start + batch_size)
        indices = coordinates[start:end, None] + offsets
        inside = (indices >= 0) & (indices < len(letters))
        window = np.full(indices.shape, 4, dtype=np.uint8)
        window[inside] = letters[indices[inside]]
        result[0][start:end] = np.count_nonzero(window == left_code, axis=1)
        result[1][start:end] = np.count_nonzero(window == right_code, axis=1)
        result[2][start:end] = np.count_nonzero(
            (window[:, :-1] == left_code) & (window[:, 1:] == right_code), axis=1)
        result[3][start:end] = np.count_nonzero(window < 4, axis=1)
    return tuple(result)


def _positive_rows(sequences: Mapping[str, np.ndarray], positives: pd.DataFrame,
                   pair: str, config: RegionalDinucleotideConfig) -> pd.DataFrame:
    result = positives[["template_index", "contig", "coordinate_0based",
                        "strand", "count"]].copy()
    context = extract_oriented_context(
        sequences, result, offsets=range(-(config.max_k - 1), 1))
    result["kmer_6_code"] = context_kmer_codes(context)
    columns = ("c_count", "g_count", "cpg_count", "valid_count",
               "left_count", "right_count", "pair_count")
    for column in columns:
        result[column] = np.int16(0)
    for (contig, strand), group in result.groupby(["contig", "strand"], sort=False):
        letters = _oriented_letters(sequences[str(contig)], str(strand))
        coordinates = group["coordinate_0based"].to_numpy(dtype=np.int64)
        if strand == "-":
            coordinates = len(letters) - 1 - coordinates
        base_stats = _sparse_window_statistics(letters, coordinates, config)
        pair_stats = _sparse_pair_statistics(letters, coordinates, pair, config)
        for column, values in zip(("c_count", "g_count", "cpg_count", "valid_count"), base_stats):
            result.loc[group.index, column] = values
        for column, values in zip(("left_count", "right_count", "pair_count", "pair_valid"), pair_stats):
            if column != "pair_valid":
                result.loc[group.index, column] = values
        if not np.array_equal(base_stats[3], pair_stats[3]):
            raise AssertionError("positive window validity counters disagree")
    result["gc_bin_code"] = gc_bin_codes(result.c_count + result.g_count, result.valid_count)
    result["cpg_bin_code"] = cpg_bin_codes(
        result.c_count, result.g_count, result.cpg_count, result.valid_count)
    result["dinucleotide_oe201"] = dinucleotide_observed_expected(
        result.left_count, result.right_count, result.pair_count, result.valid_count)
    result["dinucleotide_bin_code"] = dinucleotide_bin_codes(
        result.left_count, result.right_count, result.pair_count, result.valid_count)
    return result


def count_candidate(split: PreparedSplit, sequences: Mapping[str, np.ndarray],
                    split_name: str, *, pair: str | None = None,
                    parameters: Mapping[str, Any] | None = None,
                    config: RegionalDinucleotideConfig = RegionalDinucleotideConfig(),
                    verbose: bool = False) -> KmerDinucleotideCounts:
    pair = _validate_pair(pair or (parameters or {}).get("pair", ""))
    if split_name not in {"train", "validation"}:
        raise ValueError("labels are available only for train or validation")
    intervals = split.intervals(split_name)
    positives = split.positive_targets(split_name)
    if positives is None or positives.empty:
        raise ValueError(f"{split_name} positives are unavailable")
    contigs = sorted(intervals.contig.astype(str).unique())
    records = []
    total_parts = sum(not intervals.loc[(intervals.contig.astype(str) == contig) &
        (intervals.strand == strand)].empty for contig in contigs for strand in ("+", "-"))
    completed = 0; started = time.monotonic()
    for contig_code, contig in enumerate(contigs):
        local = intervals.loc[intervals.contig.astype(str) == contig]
        for strand_code, strand in enumerate(("+", "-")):
            frame = local.loc[local.strand == strand, ["start", "end"]]
            if frame.empty:
                continue
            raw = sequences[contig]
            letters = _oriented_letters(raw, strand)
            ranges = [tuple(map(int, row)) for row in frame.to_numpy()]
            if strand == "-":
                ranges = [(len(letters) - end, len(letters) - start) for start, end in ranges]
            part = _count_oriented_ranges(letters, ranges, pair, config)
            part.insert(0, "strand_code", np.int8(strand_code))
            part.insert(0, "contig_code", np.int16(contig_code))
            records.append(part); completed += 1
            if verbose:
                print(f"[{completed:02d}/{total_parts:02d}] pair={pair} {contig} "
                      f"strand={strand} positions={part.total_positions.sum():,} "
                      f"rows={len(part):,} elapsed={time.monotonic()-started:.1f}s", flush=True)
    totals = pd.concat(records, ignore_index=True, copy=False)
    contig_code = totals.pop("contig_code").to_numpy(dtype=np.int16)
    strand_code = totals.pop("strand_code").to_numpy(dtype=np.int8)
    totals.insert(0, "strand", pd.Categorical.from_codes(strand_code, ["+", "-"]))
    totals.insert(0, "contig", pd.Categorical.from_codes(contig_code, contigs))
    positive_rows = _positive_rows(sequences, positives, pair, config)
    expected = int((intervals.end - intervals.start).sum())
    if int(totals.total_positions.sum()) != expected:
        raise AssertionError("candidate counts do not preserve the split population")
    return KmerDinucleotideCounts(split_name, pair, config.window_start,
                                  config.window_end, totals, positive_rows)


def _summary(counts: KmerDinucleotideCounts, contigs: Iterable[str] | None = None):
    selected = _normalise_contigs(contigs)
    totals, positives = counts.totals, counts.positives
    if selected is not None:
        totals = totals.loc[totals.contig.astype(str).isin(selected)]
        positives = positives.loc[positives.contig.astype(str).isin(selected)]
    keys = ["kmer_6_code", "gc_bin_code", "cpg_bin_code", "dinucleotide_bin_code"]
    total = totals.groupby(keys, as_index=False, observed=True).total_positions.sum()
    positive = positives.groupby(keys, observed=True).size().rename("positive_positions").reset_index()
    result = total.merge(positive, on=keys, how="left", validate="one_to_one")
    result["positive_positions"] = result.positive_positions.fillna(0).astype(np.int64)
    result["negative_positions"] = result.total_positions - result.positive_positions
    if (result.negative_positions < 0).any():
        raise AssertionError("positive tuple count exceeds total")
    return result


def _matrix(summary: pd.DataFrame, design: HierarchicalKmerDesign,
            pair: str, config: RegionalDinucleotideConfig):
    champion = _champion_matrix(summary, design, config)
    extra = _dinucleotide_matrix(summary.dinucleotide_bin_code.to_numpy(), pair)
    return sparse.hstack([champion, extra], format="csr")


def aggregate_candidate(counts: KmerDinucleotideCounts, *,
                        config: RegionalDinucleotideConfig = RegionalDinucleotideConfig(),
                        contigs: Iterable[str] | None = None,
                        design: HierarchicalKmerDesign | None = None) -> DinucleotideAggregate:
    if counts.split_name != "train":
        raise ValueError("training aggregate requires train counts")
    design = design or hierarchical_kmer_design(config)
    summary = _summary(counts, contigs)
    population = int(summary.total_positions.sum())
    positives = int(summary.positive_positions.sum()); negatives = population - positives
    indices = np.repeat(np.arange(len(summary)), 2)
    target = np.tile(np.array([0, 1], dtype=np.int8), len(summary))
    population_count = np.column_stack((summary.negative_positions, summary.positive_positions)).reshape(-1)
    keep = population_count > 0
    indices, target, population_count = indices[keep], target[keep], population_count[keep]
    multipliers = np.array([population / (2 * negatives), population / (2 * positives)])
    weights = population_count * multipliers[target]
    rows = summary.iloc[indices].reset_index(drop=True).assign(
        target=target, population_count=population_count,
        class_multiplier=multipliers[target], sample_weight=weights)
    return DinucleotideAggregate(_matrix(summary, design, counts.pair, config)[indices],
                                 target, weights, rows)


def _coefficient_table(estimator, design, pair, config):
    from ml_project.epic_cpg_logistic import _coefficient_table as champion_coefficients
    # champion_coefficients cannot consume the longer estimator directly; rebuild
    champion_estimator = type("View", (), {"coef_": estimator.coef_[:, :4386]})()
    champion = champion_coefficients(champion_estimator, design, config)
    extra = dinucleotide_feature_table(pair)
    extra["feature_index"] = 4386 + extra.feature_index_within_pair
    extra["level_k"] = pd.NA; extra["category_code"] = extra.bin_code
    extra["category"] = extra.bin_label
    extra["coefficient"] = estimator.coef_[0, 4386:]
    extra["absolute_coefficient"] = np.abs(extra.coefficient)
    return pd.concat([champion, extra[champion.columns]], ignore_index=True)


def _fit(aggregate, design, pair, config):
    estimator = LogisticRegression(penalty=config.penalty, C=config.candidate_C,
        solver=config.solver, max_iter=config.max_iter, tol=config.tol,
        random_state=config.seed)
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(aggregate.matrix, aggregate.target, sample_weight=aggregate.sample_weight)
    messages = [str(x.message) for x in caught if issubclass(x.category, ConvergenceWarning)]
    iterations = int(estimator.n_iter_[0])
    report = pd.DataFrame([{"pair": pair, "C": config.candidate_C,
        "iterations": iterations, "max_iter": config.max_iter,
        "converged": not messages and iterations < config.max_iter,
        "fit_seconds": time.monotonic() - started,
        "aggregate_rows": len(aggregate.rows), "features": aggregate.matrix.shape[1],
        "represented_positions": int(aggregate.rows.population_count.sum()),
        "warnings": " | ".join(messages)}])
    return DinucleotideFit(estimator, design, pair, aggregate,
                           _coefficient_table(estimator, design, pair, config), report)


def fit_candidate(counts: KmerDinucleotideCounts, *,
                  config: RegionalDinucleotideConfig = RegionalDinucleotideConfig(),
                  contigs: Iterable[str] | None = None, **_: Any) -> DinucleotideFit:
    design = hierarchical_kmer_design(config)
    return _fit(aggregate_candidate(counts, config=config, contigs=contigs, design=design),
                design, counts.pair, config)


def _evaluate(counts, fit, config, contigs, include_curve, variant):
    selected = _normalise_contigs(contigs)
    summary = _summary(counts, selected)
    scores = fit.estimator.decision_function(_matrix(summary, fit.design, fit.pair, config))
    total = summary.total_positions.to_numpy(dtype=np.int64)
    positive = summary.positive_positions.to_numpy(dtype=np.int64)
    ap, curve = _average_precision_from_rows(total, positive, scores, include_curve=include_curve)
    keys = ["kmer_6_code", "gc_bin_code", "cpg_bin_code", "dinucleotide_bin_code"]
    score_table = summary[keys].copy(); score_table["score"] = scores
    rows = counts.positives
    if selected is not None:
        rows = rows.loc[rows.contig.astype(str).isin(selected)]
    predictions = rows.merge(score_table, on=keys, how="left", validate="many_to_one")
    spearman = _dense_rank_correlation(predictions["count"].to_numpy(), predictions.score.to_numpy())
    segment = "overall" if selected is None else ",".join(selected)
    metric = pd.DataFrame([{"segment": segment, "variant": variant,
        "average_precision": ap, "epic_spearman": spearman,
        "positions": int(total.sum()), "positives": int(positive.sum()),
        "prevalence": positive.sum() / total.sum()}])
    if not curve.empty:
        curve["segment"] = segment; curve["variant"] = variant
    return metric, curve, predictions.assign(segment=segment, variant=variant)


def evaluate_candidate(counts: KmerDinucleotideCounts, fit: DinucleotideFit, *,
                       config: RegionalDinucleotideConfig = RegionalDinucleotideConfig(),
                       contigs: Iterable[str] | None = None,
                       include_curve: bool = True,
                       include_per_contig: bool = True, **_: Any) -> GCWindowEvaluation:
    variant = f"EXP-010 EXP-009 + {counts.pair} O/E201"
    overall, curve, predictions = _evaluate(
        counts, fit, config, contigs, include_curve, variant)
    selected = _normalise_contigs(contigs)
    if include_per_contig:
        selected = selected or tuple(sorted(counts.totals.contig.astype(str).unique()))
        per_contig = pd.concat([_evaluate(counts, fit, config, [c], False, variant)[0]
                                for c in selected], ignore_index=True).rename(columns={"segment": "contig"})
    else:
        per_contig = pd.DataFrame()
    return GCWindowEvaluation(overall, per_contig, curve, predictions)


def cross_validate_candidate(train_counts: KmerDinucleotideCounts,
                             fold_assignment: pd.DataFrame, *,
                             config: RegionalDinucleotideConfig = RegionalDinucleotideConfig(),
                             verbose: bool = False,
                             progress: Callable[[dict[str, Any]], None] | None = None,
                             **_: Any) -> pd.DataFrame:
    expected = set(train_counts.totals.contig.astype(str))
    if set(fold_assignment.contig.astype(str)) != expected:
        raise ValueError("fold_assignment must contain every train contig")
    design = hierarchical_kmer_design(config); records = []
    folds = sorted(fold_assignment.fold.astype(int).unique())
    if len(folds) != 3:
        raise AssertionError("EXP-010 requires exactly three folds per candidate")
    for index, fold in enumerate(folds, 1):
        held_out = tuple(fold_assignment.loc[fold_assignment.fold == fold, "contig"].astype(str))
        fitting = tuple(sorted(expected - set(held_out)))
        fitted = _fit(aggregate_candidate(train_counts, config=config,
            contigs=fitting, design=design), design, train_counts.pair, config)
        evaluated = evaluate_candidate(train_counts, fitted, config=config,
            contigs=held_out, include_curve=False, include_per_contig=False)
        report = fitted.fit_report.iloc[0]
        record = {"pair": train_counts.pair, "fold": fold,
            "validation_contigs": ",".join(held_out),
            **evaluated.metric_summary.iloc[0].to_dict(), **report.to_dict()}
        records.append(record)
        event = {"completed": index, "total": 3, "pair": train_counts.pair,
                 "fold": fold, "average_precision": record["average_precision"],
                 "converged": report["converged"], "fit_seconds": report["fit_seconds"]}
        if verbose:
            print(f"[{index}/3] pair={train_counts.pair} fold={fold} C={FIXED_C:g} "
                  f"AP={event['average_precision']:.9f} time={event['fit_seconds']:.2f}s "
                  f"converged={event['converged']}", flush=True)
        if progress:
            progress(event)
    return pd.DataFrame(records).sort_values("fold").reset_index(drop=True)


def dinucleotide_enrichment_table(counts: KmerDinucleotideCounts) -> pd.DataFrame:
    summary = _summary(counts)
    grouped = summary.groupby("dinucleotide_bin_code", as_index=False).agg(
        positions=("total_positions", "sum"), positives=("positive_positions", "sum"))
    full = pd.DataFrame({"dinucleotide_bin_code": np.arange(8)})
    grouped = full.merge(grouped, how="left").fillna(0)
    grouped["bin"] = np.asarray(DINUCLEOTIDE_LABELS)[grouped.dinucleotide_bin_code]
    grouped["positive_rate"] = grouped.positives / grouped.positions.replace(0, np.nan)
    prevalence = summary.positive_positions.sum() / summary.total_positions.sum()
    grouped["enrichment"] = grouped.positive_rate / prevalence
    grouped.insert(0, "pair", counts.pair)
    return grouped


def diagnose_feature(counts: KmerDinucleotideCounts | PreparedSplit,
                     sequences: Mapping[str, np.ndarray] | None = None, *,
                     pair: str | None = None,
                     parameters: Mapping[str, Any] | None = None,
                     config: RegionalDinucleotideConfig = RegionalDinucleotideConfig(),
                     max_positions: int = 2_000_000, **_: Any) -> pd.DataFrame:
    """Run a bounded target-free pilot, or summarize already counted rows."""
    if isinstance(counts, PreparedSplit):
        if sequences is None:
            raise ValueError("sequences are required for a raw pilot")
        pair = _validate_pair(pair or (parameters or {}).get("pair", ""))
        intervals = counts.intervals("train").copy()
        intervals["length"] = intervals.end - intervals.start
        choices = (intervals.groupby(["contig", "strand"], observed=True)
                   .length.sum().sort_values(ascending=False))
        contig, strand = choices.index[0]
        local = intervals.loc[(intervals.contig.astype(str) == str(contig)) &
                              (intervals.strand == strand), ["start", "end"]]
        remaining = int(max_positions); ranges = []
        for start, end in local.to_numpy():
            if remaining <= 0:
                break
            take = min(int(end - start), remaining)
            ranges.append((int(start), int(start) + take)); remaining -= take
        letters = _oriented_letters(sequences[str(contig)], str(strand))
        if strand == "-":
            ranges = [(len(letters) - end, len(letters) - start)
                      for start, end in ranges]
        started = time.monotonic()
        totals = _count_oriented_ranges(letters, ranges, pair, config)
        positions = int(totals.total_positions.sum())
        return pd.DataFrame([{"pair": pair, "pilot_contig": str(contig),
            "pilot_strand": str(strand), "positions": positions,
            "joint_rows": len(totals),
            "unique_rows_per_position": len(totals) / positions,
            "totals_memory_mib": totals.memory_usage(deep=True).sum() / 2**20,
            "seconds": time.monotonic() - started}])

    counts = counts
    return pd.DataFrame([{"pair": counts.pair,
        "positions": int(counts.totals.total_positions.sum()),
        "joint_rows": len(counts.totals),
        "unique_rows_per_position": len(counts.totals) / counts.totals.total_positions.sum(),
        "totals_memory_mib": counts.totals.memory_usage(deep=True).sum() / 2**20,
        "positive_rows": len(counts.positives),
        "missing_positive_rows": int((counts.positives.dinucleotide_bin_code == 7).sum())}])


__all__ = ["DINUCLEOTIDE_PAIRS", "FEATURE_READY", "FIXED_C",
    "RegionalDinucleotideConfig", "KmerDinucleotideCounts", "DinucleotideFit",
    "aggregate_candidate", "count_candidate", "cross_validate_candidate",
    "diagnose_feature", "dinucleotide_bin_codes", "dinucleotide_enrichment_table",
    "dinucleotide_feature_table", "dinucleotide_observed_expected",
    "evaluate_candidate", "feature_contract", "fit_candidate"]
