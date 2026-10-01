"""Exact fixed-bin CpG observed/expected feature for EXP-009.

The accepted EXP-007 main effects stay unchanged. EXP-009 computes CpG O/E in
the same inclusive strand-oriented 201 bp window:

``CpG O/E = CpG_count * valid_ACGT / (C_count * G_count)``.

The raw ratio is mapped to fixed, pre-registered bins.  This keeps the full
population likelihood exact and compact; no position sampling or hidden
continuous-value quantization is used.  Counting and fitting are explicit
calls; importing this module does no project work.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
import warnings
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .epic_baseline import _dense_rank_correlation
from .epic_data import PreparedSplit
from .epic_gc_binned_logistic import (
    GC_BIN_LABELS,
    GCBinnedLogisticConfig,
    _gc_bin_matrix,
    gc_bin_codes,
    gc_bin_feature_table,
)
from .epic_gc_logistic import (
    GCWindowEvaluation,
    _average_precision_from_rows,
    _normalise_contigs,
)
from .epic_kmer import (
    KmerCounts,
    _codes_from_state,
    _packed_suffix_state,
    _uppercase_letters,
    context_kmer_codes,
    kmer_category_count,
)
from .epic_kmer_logistic import HierarchicalKmerDesign, hierarchical_kmer_design
from .sequence_context import extract_oriented_context


CPG_C_GRID: tuple[float, ...] = (1e-4, 2e-4, 3e-4, 5e-4, 1e-3)
CPG_DIAGNOSTIC_EDGES: tuple[float, ...] = (
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
)
CPG_DIAGNOSTIC_LABELS: tuple[str, ...] = (
    "<0.25",
    "[0.25,0.50)",
    "[0.50,0.75)",
    "[0.75,1.00)",
    "[1.00,1.25)",
    "[1.25,1.50)",
    ">=1.50",
    "missing C/G",
)
CPG_REFERENCE_BIN = 3
CPG_MISSING_BIN = 7


@dataclass(frozen=True)
class CpGLogisticConfig(GCBinnedLogisticConfig):
    """Every setting capable of changing the EXP-009 candidate."""

    C_grid: tuple[float, ...] = CPG_C_GRID
    # CpG needs three composition counters in addition to the k-mer state.
    # Keep the transient chunk deliberately smaller than in EXP-006.
    count_batch_size: int = 250_000
    cpg_reference_bin: int = CPG_REFERENCE_BIN

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.C_grid != CPG_C_GRID:
            raise ValueError("EXP-009 uses five preregistered C values (15 CV fits)")
        if self.cpg_reference_bin != CPG_REFERENCE_BIN:
            raise ValueError("EXP-009 CpG reference must be [0.75,1.00)")

    def to_dict(self) -> dict[str, object]:
        result = super().to_dict()
        result["C_grid"] = list(self.C_grid)
        result["cpg_feature"] = (
            "CpG_count * valid_ACGT / (C_count * G_count) in inclusive "
            "strand-oriented offsets [-100,+100]"
        )
        result["cpg_bin_edges"] = list(CPG_DIAGNOSTIC_EDGES)
        result["cpg_bin_labels"] = list(CPG_DIAGNOSTIC_LABELS)
        result["cpg_reference_bin"] = self.cpg_reference_bin
        result["cpg_reference_bin_label"] = CPG_DIAGNOSTIC_LABELS[
            self.cpg_reference_bin
        ]
        result["cpg_missing_policy"] = "separate categorical bin when C_count*G_count is zero"
        result["cpg_scaling"] = "none; fixed train-independent bin boundaries"
        result["cpg_clipping"] = "none"
        result["feature_encoding"] = (
            "EXP-007 main effects + seven reference-coded fixed CpG O/E201 bins"
        )
        result["aggregation"] = (
            "exact contig x strand x 6-mer x GC201-bin x CpG-O/E201-bin statistics"
        )
        result["thread_control"] = (
            "notebook sets MKL_THREADING_LAYER=SEQUENTIAL before numeric imports; "
            "threadpoolctl is not called during fit"
        )
        return result


@dataclass(frozen=True)
class KmerCpGCounts:
    split_name: str
    window_start: int
    window_end: int
    totals: pd.DataFrame
    positives: pd.DataFrame


@dataclass(frozen=True)
class AggregateCpGLikelihood:
    matrix: sparse.csr_matrix
    target: np.ndarray
    sample_weight: np.ndarray
    rows: pd.DataFrame


@dataclass(frozen=True)
class CpGLogisticFit:
    estimator: LogisticRegression
    design: HierarchicalKmerDesign
    aggregate: AggregateCpGLikelihood
    coefficient_table: pd.DataFrame
    fit_report: pd.DataFrame


def cpg_observed_expected(
    c_count: np.ndarray | pd.Series,
    g_count: np.ndarray | pd.Series,
    cpg_count: np.ndarray | pd.Series,
    valid_count: np.ndarray | pd.Series,
) -> np.ndarray:
    """Compute un-clipped CpG O/E; return NaN when expectation is undefined."""

    c = np.asarray(c_count, dtype=np.float64)
    g = np.asarray(g_count, dtype=np.float64)
    cpg = np.asarray(cpg_count, dtype=np.float64)
    valid = np.asarray(valid_count, dtype=np.float64)
    if not (c.shape == g.shape == cpg.shape == valid.shape):
        raise ValueError("CpG statistic arrays must have identical shapes")
    denominator = c * g
    return np.divide(
        cpg * valid,
        denominator,
        out=np.full(c.shape, np.nan, dtype=np.float64),
        where=(denominator > 0) & (valid > 0),
    )


def cpg_bin_codes(
    c_count: np.ndarray | pd.Series,
    g_count: np.ndarray | pd.Series,
    cpg_count: np.ndarray | pd.Series,
    valid_count: np.ndarray | pd.Series,
) -> np.ndarray:
    """Map the raw CpG O/E ratio to the fixed EXP-009 vocabulary."""

    values = cpg_observed_expected(c_count, g_count, cpg_count, valid_count)
    result = np.searchsorted(
        np.asarray(CPG_DIAGNOSTIC_EDGES), values, side="right"
    ).astype(np.int8)
    result[~np.isfinite(values)] = CPG_MISSING_BIN
    return result


def cpg_bin_feature_table(
    config: CpGLogisticConfig = CpGLogisticConfig(),
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    feature_index = 0
    for code, label in enumerate(CPG_DIAGNOSTIC_LABELS):
        if code == config.cpg_reference_bin:
            continue
        records.append(
            {
                "bin_code": code,
                "bin_label": label,
                "feature_index_within_cpg": feature_index,
                "feature_name": f"cpg_oe201_bin={label}",
                "feature_group": "CpG O/E201 bin",
            }
        )
        feature_index += 1
    return pd.DataFrame(records)


def _cpg_bin_matrix(
    codes: np.ndarray,
    config: CpGLogisticConfig,
) -> sparse.csr_matrix:
    codes = np.asarray(codes, dtype=np.int8)
    if ((codes < 0) | (codes >= len(CPG_DIAGNOSTIC_LABELS))).any():
        raise ValueError("CpG O/E bin code outside the fixed vocabulary")
    table = cpg_bin_feature_table(config)
    column_for_code = np.full(len(CPG_DIAGNOSTIC_LABELS), -1, dtype=np.int16)
    column_for_code[table["bin_code"].to_numpy(dtype=np.int8)] = table[
        "feature_index_within_cpg"
    ].to_numpy(dtype=np.int16)
    columns = column_for_code[codes]
    active = columns >= 0
    return sparse.csr_matrix(
        (
            np.ones(int(active.sum()), dtype=np.float64),
            (np.flatnonzero(active), columns[active]),
        ),
        shape=(len(codes), len(table)),
    )


def _prefix(values: np.ndarray) -> np.ndarray:
    result = np.empty(len(values) + 1, dtype=np.int32)
    result[0] = 0
    np.cumsum(values, dtype=np.int32, out=result[1:])
    return result


def _composition_prefixes(
    letters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c_prefix = _prefix(letters == 1)
    g_prefix = _prefix(letters == 2)
    valid_prefix = _prefix(letters < 4)
    cpg_starts = (letters[:-1] == 1) & (letters[1:] == 2)
    cpg_prefix = _prefix(cpg_starts)
    return c_prefix, g_prefix, valid_prefix, cpg_prefix


def _window_statistics(
    prefixes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    coordinates: np.ndarray,
    *,
    sequence_length: int,
    window_start: int,
    window_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.asarray(coordinates, dtype=np.int64)
    lo = np.maximum(0, coordinates + window_start)
    hi = np.minimum(sequence_length, coordinates + window_end + 1)
    c_prefix, g_prefix, valid_prefix, cpg_prefix = prefixes
    c_count = c_prefix[hi] - c_prefix[lo]
    g_count = g_prefix[hi] - g_prefix[lo]
    valid_count = valid_prefix[hi] - valid_prefix[lo]
    pair_hi = np.maximum(lo, hi - 1)
    cpg_count = cpg_prefix[pair_hi] - cpg_prefix[lo]
    return (
        c_count.astype(np.int16),
        g_count.astype(np.int16),
        cpg_count.astype(np.int16),
        valid_count.astype(np.int16),
    )


def _merge_sorted_counts(
    existing_keys: np.ndarray,
    existing_counts: np.ndarray,
    new_keys: np.ndarray,
    new_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge two sorted unique counter arrays without a global re-sort."""

    if not len(existing_keys):
        return (
            new_keys.astype(np.int64, copy=False),
            new_counts.astype(np.int64, copy=False),
        )
    if not len(new_keys):
        return existing_keys, existing_counts

    positions = np.searchsorted(existing_keys, new_keys)
    matched = positions < len(existing_keys)
    matched_indices = np.flatnonzero(matched)
    matched[matched_indices] = (
        existing_keys[positions[matched_indices]] == new_keys[matched_indices]
    )
    if matched.any():
        existing_counts[positions[matched]] += new_counts[matched]

    unseen_keys = new_keys[~matched]
    if not len(unseen_keys):
        return existing_keys, existing_counts
    unseen_counts = new_counts[~matched]
    insertion_points = positions[~matched]
    final_positions = insertion_points + np.arange(len(unseen_keys), dtype=np.int64)
    final_length = len(existing_keys) + len(unseen_keys)
    result_keys = np.empty(final_length, dtype=np.int64)
    result_counts = np.empty(final_length, dtype=np.int64)
    inserted = np.zeros(final_length, dtype=bool)
    inserted[final_positions] = True
    result_keys[final_positions] = unseen_keys
    result_counts[final_positions] = unseen_counts
    result_keys[~inserted] = existing_keys
    result_counts[~inserted] = existing_counts
    return result_keys, result_counts


def _add_count_part(
    levels: list[tuple[np.ndarray, np.ndarray] | None],
    keys: np.ndarray,
    counts: np.ndarray,
) -> None:
    """Binary-compaction merge: each batch participates in O(log batches)."""

    level = 0
    while True:
        if level == len(levels):
            levels.append((keys, counts))
            return
        current = levels[level]
        if current is None:
            levels[level] = (keys, counts)
            return
        levels[level] = None
        keys, counts = _merge_sorted_counts(
            current[0], current[1], keys, counts
        )
        level += 1


def _collapse_count_parts(
    levels: list[tuple[np.ndarray, np.ndarray] | None],
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.array([], dtype=np.int64)
    counts = np.array([], dtype=np.int64)
    for part in reversed(levels):
        if part is not None:
            keys, counts = _merge_sorted_counts(
                keys, counts, part[0], part[1]
            )
    return keys, counts


def _feature_keys_for_batch(
    letters: np.ndarray,
    start: int,
    end: int,
    config: CpGLogisticConfig,
) -> np.ndarray:
    """Build packed keys with only a bounded sequence halo in memory."""

    halo_start = max(
        0,
        start + min(config.window_start, -(config.max_k - 1)),
    )
    halo_end = min(len(letters), end + config.window_end)
    local_letters = letters[halo_start:halo_end]
    local_start = start - halo_start
    local_end = end - halo_start

    packed, unknown_prefix = _packed_suffix_state(local_letters, config.max_k)
    kmer_code = _codes_from_state(
        packed,
        unknown_prefix,
        local_start,
        local_end,
        config.max_k,
    ).astype(np.int64, copy=False)
    coordinates = np.arange(local_start, local_end, dtype=np.int64)
    c_count, g_count, cpg_count, valid_count = _window_statistics(
        _composition_prefixes(local_letters),
        coordinates,
        sequence_length=len(local_letters),
        window_start=config.window_start,
        window_end=config.window_end,
    )

    gc_codes = gc_bin_codes(c_count + g_count, valid_count)
    cpg_codes = cpg_bin_codes(c_count, g_count, cpg_count, valid_count)
    keys = kmer_code
    keys *= len(GC_BIN_LABELS)
    np.add(keys, gc_codes, out=keys, casting="unsafe")
    keys *= len(CPG_DIAGNOSTIC_LABELS)
    np.add(keys, cpg_codes, out=keys, casting="unsafe")
    return keys


def _count_oriented_ranges(
    letters: np.ndarray,
    ranges: Sequence[tuple[int, int]],
    config: CpGLogisticConfig,
) -> pd.DataFrame:
    """Count exact EXP-009 tuples for one oriented contig-strand."""

    count_levels: list[tuple[np.ndarray, np.ndarray] | None] = []

    for interval_start, interval_end in ranges:
        cursor = int(interval_start)
        stop = int(interval_end)
        if not 0 <= cursor < stop <= len(letters):
            raise ValueError("whitelist interval is outside its FASTA contig")
        while cursor < stop:
            take = min(stop - cursor, config.count_batch_size)
            end = cursor + take
            local_keys, local_counts = np.unique(
                _feature_keys_for_batch(letters, cursor, end, config),
                return_counts=True,
            )
            _add_count_part(
                count_levels,
                local_keys.astype(np.int64, copy=False),
                local_counts.astype(np.int64, copy=False),
            )
            cursor = end

    packed_keys, population_counts = _collapse_count_parts(count_levels)
    remainder = packed_keys
    cpg_bin_code = (remainder % len(CPG_DIAGNOSTIC_LABELS)).astype(np.int8)
    remainder //= len(CPG_DIAGNOSTIC_LABELS)
    gc_bin_code = (remainder % len(GC_BIN_LABELS)).astype(np.int8)
    kmer_code = remainder // len(GC_BIN_LABELS)
    return pd.DataFrame(
        {
            "kmer_6_code": kmer_code.astype(np.int32),
            "gc_bin_code": gc_bin_code,
            "cpg_bin_code": cpg_bin_code,
            "total_positions": population_counts.astype(np.int64),
        }
    )


def _oriented_letters(raw_sequence: np.ndarray, strand: str) -> np.ndarray:
    """Materialize one strand with one full uint8 allocation."""

    raw = np.asarray(raw_sequence)
    if strand == "+":
        return _uppercase_letters(raw)
    if strand != "-":
        raise ValueError("strand must be '+' or '-'")
    letters = _uppercase_letters(raw[::-1])
    np.subtract(3, letters, out=letters, where=letters < 4)
    return letters


def _sparse_window_statistics(
    letters: np.ndarray,
    coordinates: np.ndarray,
    config: CpGLogisticConfig,
    *,
    batch_size: int = 20_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute positive-row statistics without full-contig prefix arrays."""

    coordinates = np.asarray(coordinates, dtype=np.int64)
    result = [np.empty(len(coordinates), dtype=np.int16) for _ in range(4)]
    offsets = np.arange(
        config.window_start,
        config.window_end + 1,
        dtype=np.int32,
    )
    for start in range(0, len(coordinates), batch_size):
        end = min(len(coordinates), start + batch_size)
        local_coordinates = coordinates[start:end]
        if np.any(local_coordinates < 0) or np.any(local_coordinates >= len(letters)):
            raise ValueError("positive coordinate is outside its FASTA contig")
        indices = local_coordinates.astype(np.int32)[:, None] + offsets
        in_bounds = (indices >= 0) & (indices < len(letters))
        window = np.full(indices.shape, 4, dtype=np.uint8)
        window[in_bounds] = letters[indices[in_bounds]]
        result[0][start:end] = np.count_nonzero(window == 1, axis=1)
        result[1][start:end] = np.count_nonzero(window == 2, axis=1)
        result[2][start:end] = np.count_nonzero(
            (window[:, :-1] == 1) & (window[:, 1:] == 2),
            axis=1,
        )
        result[3][start:end] = np.count_nonzero(window < 4, axis=1)
    return tuple(result)  # type: ignore[return-value]


def _positive_feature_rows(
    sequences: Mapping[str, np.ndarray],
    positives: pd.DataFrame,
    config: CpGLogisticConfig,
) -> pd.DataFrame:
    result = positives[
        ["template_index", "contig", "coordinate_0based", "strand", "count"]
    ].copy()
    context = extract_oriented_context(
        sequences, result, offsets=range(-(config.max_k - 1), 1)
    )
    result["kmer_6_code"] = context_kmer_codes(context)
    for column in ("c_count", "g_count", "cpg_count", "valid_count"):
        result[column] = np.int16(0)
    for (contig, strand), group in result.groupby(["contig", "strand"], sort=False):
        letters = _oriented_letters(sequences[str(contig)], str(strand))
        coordinates = group["coordinate_0based"].to_numpy(dtype=np.int64)
        if strand == "-":
            coordinates = len(letters) - 1 - coordinates
        stats = _sparse_window_statistics(letters, coordinates, config)
        for column, values in zip(
            ("c_count", "g_count", "cpg_count", "valid_count"), stats
        ):
            result.loc[group.index, column] = values
    for column in ("c_count", "g_count", "cpg_count", "valid_count"):
        result[column] = result[column].astype("int16")
    result["gc_count"] = (
        result["c_count"].astype(np.int16) + result["g_count"].astype(np.int16)
    )
    result["cpg_oe201"] = cpg_observed_expected(
        result["c_count"],
        result["g_count"],
        result["cpg_count"],
        result["valid_count"],
    )
    result["gc_bin_code"] = gc_bin_codes(
        result["gc_count"], result["valid_count"]
    )
    result["cpg_bin_code"] = cpg_bin_codes(
        result["c_count"],
        result["g_count"],
        result["cpg_count"],
        result["valid_count"],
    )
    return result


def split_kmer_cpg_counts(
    split: PreparedSplit,
    sequences: Mapping[str, np.ndarray],
    split_name: str,
    *,
    config: CpGLogisticConfig = CpGLogisticConfig(),
    verbose: bool = False,
) -> KmerCpGCounts:
    """Count every labelled whitelist position by exact EXP-009 statistics."""

    if split_name not in {"train", "validation"}:
        raise ValueError("CpG counts require train or validation labels")
    intervals = split.intervals(split_name)
    positives = split.positive_targets(split_name)
    if positives is None or positives.empty:
        raise ValueError(f"{split_name} positives are unavailable")
    missing = sorted(set(intervals["contig"].astype(str)) - set(sequences))
    if missing:
        raise ValueError("split contigs missing from FASTA: " + ", ".join(missing))

    records: list[pd.DataFrame] = []
    contigs = sorted(intervals["contig"].astype(str).unique())
    total_parts = sum(
        int(not intervals.loc[
            (intervals["contig"].astype(str) == contig)
            & (intervals["strand"] == strand)
        ].empty)
        for contig in contigs
        for strand in ("+", "-")
    )
    completed_parts = 0
    counting_started = time.monotonic()
    for contig_code, contig in enumerate(contigs):
        raw_sequence = sequences[contig]
        local = intervals.loc[intervals["contig"].astype(str) == contig]
        for strand_code, strand in enumerate(("+", "-")):
            strand_intervals = local.loc[
                local["strand"] == strand, ["start", "end"]
            ]
            if strand_intervals.empty:
                continue
            ranges = [tuple(map(int, row)) for row in strand_intervals.to_numpy()]
            oriented = _oriented_letters(raw_sequence, strand)
            if strand == "-":
                length = len(oriented)
                oriented_ranges = [
                    (length - end, length - start) for start, end in ranges
                ]
            else:
                oriented_ranges = ranges
            part_started = time.monotonic()
            counted = _count_oriented_ranges(oriented, oriented_ranges, config)
            counted.insert(0, "strand_code", np.int8(strand_code))
            counted.insert(0, "contig_code", np.int16(contig_code))
            records.append(counted)
            completed_parts += 1
            if verbose:
                represented = int(counted["total_positions"].sum())
                print(
                    f"[{completed_parts:02d}/{total_parts:02d}] "
                    f"{contig} strand={strand} positions={represented:,} "
                    f"joint_rows={len(counted):,} "
                    f"part_time={time.monotonic() - part_started:.1f}s "
                    f"elapsed={time.monotonic() - counting_started:.1f}s",
                    flush=True,
                )

    totals = pd.concat(records, ignore_index=True, copy=False)
    contig_codes = totals.pop("contig_code").to_numpy(dtype=np.int16)
    strand_codes = totals.pop("strand_code").to_numpy(dtype=np.int8)
    totals.insert(
        0,
        "strand",
        pd.Categorical.from_codes(strand_codes, categories=["+", "-"]),
    )
    totals.insert(
        0,
        "contig",
        pd.Categorical.from_codes(contig_codes, categories=contigs),
    )
    positives_with_features = _positive_feature_rows(sequences, positives, config)
    expected_population = int((intervals["end"] - intervals["start"]).sum())
    if int(totals["total_positions"].sum()) != expected_population:
        raise AssertionError("joint CpG counts do not cover the complete split")

    keys = [
        "contig",
        "strand",
        "kmer_6_code",
        "gc_bin_code",
        "cpg_bin_code",
    ]
    total_by_key = totals.groupby(keys, observed=True)["total_positions"].sum()
    positive_by_key = positives_with_features.groupby(keys, observed=True).size()
    aligned = total_by_key.reindex(positive_by_key.index, fill_value=0)
    if np.any(positive_by_key.to_numpy() > aligned.to_numpy()):
        raise AssertionError("positive tuple count exceeds its denominator")
    return KmerCpGCounts(
        split_name,
        config.window_start,
        config.window_end,
        totals,
        positives_with_features,
    )


def kmer_counts_from_cpg_counts(
    counts: KmerCpGCounts,
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
        local = local.groupby(
            ["k", "contig", "strand", "category_code"],
            as_index=False,
            observed=True,
        )["total_positions"].sum()
        total_frames.append(local)
        positive_rows[f"kmer_{k}_code"] = np.where(
            max_codes_positive == max_n_edge,
            n_edge,
            max_codes_positive % (4**k),
        ).astype(np.int32)
    return KmerCounts(
        counts.split_name,
        normalized,
        pd.concat(total_frames, ignore_index=True),
        positive_rows,
    )


def _joint_summary(
    counts: KmerCpGCounts,
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
    keys = [
        "kmer_6_code",
        "gc_bin_code",
        "cpg_bin_code",
    ]
    total = totals.groupby(keys, as_index=False)["total_positions"].sum()
    positive = positives.groupby(keys).size().rename("positive_positions").reset_index()
    result = total.merge(positive, on=keys, how="left", validate="one_to_one")
    result["positive_positions"] = result["positive_positions"].fillna(0).astype("int64")
    result["negative_positions"] = result["total_positions"] - result["positive_positions"]
    if (result["negative_positions"] < 0).any():
        raise AssertionError("positive tuple count exceeds total")
    return result


def _matrix_for_summary(
    summary: pd.DataFrame,
    design: HierarchicalKmerDesign,
    config: CpGLogisticConfig,
) -> sparse.csr_matrix:
    kmer = design.matrix[summary["kmer_6_code"].to_numpy(dtype=np.int32)]
    gc = _gc_bin_matrix(
        summary["gc_bin_code"].to_numpy(dtype=np.int8), config
    )
    cpg = _cpg_bin_matrix(
        summary["cpg_bin_code"].to_numpy(dtype=np.int8), config
    )
    return sparse.hstack(
        [kmer, gc, cpg], format="csr"
    )


def aggregate_cpg_likelihood(
    counts: KmerCpGCounts,
    design: HierarchicalKmerDesign,
    *,
    config: CpGLogisticConfig = CpGLogisticConfig(),
    contigs: Iterable[str] | None = None,
) -> AggregateCpGLikelihood:
    """Build exact class rows for fixed GC and CpG O/E bins."""

    if counts.split_name != "train":
        raise ValueError("training aggregate requires train counts")
    summary = _joint_summary(counts, contigs=contigs)
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
    if config.balance_classes:
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
    rows = summary.iloc[tuple_indices].reset_index(drop=True).assign(
        target=targets,
        population_count=population_counts,
        class_multiplier=multipliers[targets],
        sample_weight=weights,
    )
    tuple_matrix = _matrix_for_summary(summary, design, config)
    matrix = tuple_matrix[tuple_indices]
    if int(rows["population_count"].sum()) != population:
        raise AssertionError("aggregate rows do not preserve the population")
    if config.balance_classes:
        sums = rows.groupby("target")["sample_weight"].sum()
        if not np.isclose(sums.loc[0], sums.loc[1]):
            raise AssertionError("balanced class weights must have equal sums")
    return AggregateCpGLikelihood(matrix, targets, weights, rows)


def _coefficient_table(
    estimator: LogisticRegression,
    design: HierarchicalKmerDesign,
    config: CpGLogisticConfig,
) -> pd.DataFrame:
    kmer = design.feature_table.copy()
    kmer["feature_group"] = kmer["level_k"].map(lambda value: f"{int(value)}-mer")
    bins = gc_bin_feature_table(config).copy()
    bins["feature_index"] = len(kmer) + bins["feature_index_within_gc"]
    bins["level_k"] = pd.NA
    bins["category_code"] = bins["bin_code"]
    bins["category"] = bins["bin_label"]
    bins = bins[
        [
            "feature_index",
            "level_k",
            "category_code",
            "category",
            "feature_name",
            "feature_group",
        ]
    ]
    cpg = cpg_bin_feature_table(config).copy()
    cpg["feature_index"] = (
        len(kmer) + len(bins) + cpg["feature_index_within_cpg"]
    )
    cpg["level_k"] = pd.NA
    cpg["category_code"] = cpg["bin_code"]
    cpg["category"] = cpg["bin_label"]
    cpg = cpg[
        [
            "feature_index",
            "level_k",
            "category_code",
            "category",
            "feature_name",
            "feature_group",
        ]
    ]
    table = pd.concat([kmer, bins, cpg], ignore_index=True)
    table["coefficient"] = estimator.coef_[0]
    table["absolute_coefficient"] = np.abs(estimator.coef_[0])
    return table


def _fit_from_aggregate(
    aggregate: AggregateCpGLikelihood,
    design: HierarchicalKmerDesign,
    *,
    C: float,
    config: CpGLogisticConfig,
) -> CpGLogisticFit:
    estimator = LogisticRegression(
        penalty=config.penalty,
        C=float(C),
        solver=config.solver,
        max_iter=config.max_iter,
        tol=config.tol,
        random_state=config.seed,
    )
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
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
    coefficients = _coefficient_table(estimator, design, config)
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
                "cpg_bin_features": len(CPG_DIAGNOSTIC_LABELS) - 1,
                "reference_gc_bin": GC_BIN_LABELS[config.reference_bin],
                "reference_cpg_bin": CPG_DIAGNOSTIC_LABELS[
                    config.cpg_reference_bin
                ],
                "intercept": float(estimator.intercept_[0]),
                "warnings": " | ".join(messages),
            }
        ]
    )
    return CpGLogisticFit(
        estimator, design, aggregate, coefficients, fit_report
    )


def fit_cpg_logistic(
    counts: KmerCpGCounts,
    *,
    C: float,
    config: CpGLogisticConfig = CpGLogisticConfig(),
    contigs: Iterable[str] | None = None,
    design: HierarchicalKmerDesign | None = None,
) -> CpGLogisticFit:
    if counts.split_name != "train":
        raise ValueError("fit_cpg_logistic requires train counts")
    if C not in config.C_grid:
        raise ValueError("C must belong to the preregistered five-value grid")
    design = design or hierarchical_kmer_design(config)
    aggregate = aggregate_cpg_likelihood(
        counts, design, config=config, contigs=contigs
    )
    return _fit_from_aggregate(aggregate, design, C=C, config=config)


def _evaluate_segment(
    counts: KmerCpGCounts,
    fit: CpGLogisticFit,
    config: CpGLogisticConfig,
    *,
    contigs: Iterable[str] | None,
    variant: str,
    include_curve: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = _normalise_contigs(contigs)
    summary = _joint_summary(counts, contigs=selected)
    matrix = _matrix_for_summary(summary, fit.design, config)
    scores = fit.estimator.decision_function(matrix).astype(np.float64)
    total = summary["total_positions"].to_numpy(dtype=np.int64)
    positive = summary["positive_positions"].to_numpy(dtype=np.int64)
    ap, curve = _average_precision_from_rows(
        total, positive, scores, include_curve=include_curve
    )
    keys = [
        "kmer_6_code",
        "gc_bin_code",
        "cpg_bin_code",
    ]
    score_table = summary[keys].copy()
    score_table["score"] = scores
    positive_rows = counts.positives
    if selected is not None:
        positive_rows = positive_rows.loc[
            positive_rows["contig"].astype(str).isin(selected)
        ]
    predictions = positive_rows.merge(
        score_table, on=keys, how="left", validate="many_to_one"
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
            }
        ]
    )
    if not curve.empty:
        curve["segment"] = segment
        curve["variant"] = variant
    predictions = predictions.assign(segment=segment, variant=variant)
    return metric, curve, predictions


def evaluate_cpg_logistic(
    counts: KmerCpGCounts,
    fit: CpGLogisticFit,
    *,
    config: CpGLogisticConfig = CpGLogisticConfig(),
    contigs: Iterable[str] | None = None,
    variant: str = "EXP-009 EXP-007 + CpG O/E201",
    include_curve: bool = True,
    include_per_contig: bool = True,
) -> GCWindowEvaluation:
    overall, curve, predictions = _evaluate_segment(
        counts,
        fit,
        config,
        contigs=contigs,
        variant=variant,
        include_curve=include_curve,
    )
    if not include_per_contig:
        per_contig = pd.DataFrame()
    else:
        selected = _normalise_contigs(contigs)
        if selected is None:
            selected = tuple(sorted(counts.totals["contig"].astype(str).unique()))
        per_contig = pd.concat(
            [
                _evaluate_segment(
                    counts,
                    fit,
                    config,
                    contigs=[contig],
                    variant=variant,
                    include_curve=False,
                )[0]
                for contig in selected
            ],
            ignore_index=True,
        ).rename(columns={"segment": "contig"})
    return GCWindowEvaluation(overall, per_contig, curve, predictions)


def cross_validate_cpg_logistic(
    train_counts: KmerCpGCounts,
    fold_assignment: pd.DataFrame,
    *,
    config: CpGLogisticConfig = CpGLogisticConfig(),
    verbose: bool = False,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> pd.DataFrame:
    """Run exactly five C values across three fixed train-contig folds."""

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
    total_runs = len(folds) * len(config.C_grid)
    if total_runs != 15:
        raise AssertionError("EXP-009 protocol requires exactly 15 CV fits")
    run_index = 0
    for fold in folds:
        validation_contigs = tuple(
            fold_assignment.loc[
                fold_assignment["fold"].astype(int) == fold, "contig"
            ].astype(str)
        )
        fitting_contigs = tuple(sorted(expected - set(validation_contigs)))
        aggregate = aggregate_cpg_likelihood(
            train_counts, design, config=config, contigs=fitting_contigs
        )
        for C in config.C_grid:
            run_index += 1
            fitted = _fit_from_aggregate(
                aggregate, design, C=float(C), config=config
            )
            evaluated = evaluate_cpg_logistic(
                train_counts,
                fitted,
                config=config,
                contigs=validation_contigs,
                include_curve=False,
                include_per_contig=False,
            )
            report = fitted.fit_report.iloc[0]
            cpg_coefficients = {
                int(row.category_code): float(row.coefficient)
                for row in fitted.coefficient_table.loc[
                    fitted.coefficient_table["feature_group"] == "CpG O/E201 bin"
                ].itertuples(index=False)
            }
            record: dict[str, object] = {
                "C": float(C),
                "fold": int(fold),
                "validation_contigs": ",".join(validation_contigs),
                **evaluated.metric_summary.iloc[0].to_dict(),
                **report[
                    [
                        "iterations",
                        "converged",
                        "fit_seconds",
                        "aggregate_rows",
                        "warnings",
                    ]
                ].to_dict(),
                **{
                    f"cpg_bin_coefficient_{code}": cpg_coefficients.get(code, 0.0)
                    for code in range(len(CPG_DIAGNOSTIC_LABELS))
                },
            }
            records.append(record)
            event = {
                "completed": run_index,
                "total": total_runs,
                "fold": int(fold),
                "C": float(C),
                "fit_seconds": float(report["fit_seconds"]),
                "iterations": int(report["iterations"]),
                "converged": bool(report["converged"]),
                "average_precision": float(
                    evaluated.metric_summary.iloc[0]["average_precision"]
                ),
            }
            if verbose:
                print(
                    f"[{run_index:02d}/{total_runs}] fold={fold} C={float(C):g} "
                    f"AP={event['average_precision']:.9f} "
                    f"iter={event['iterations']} "
                    f"time={event['fit_seconds']:.2f}s "
                    f"converged={event['converged']}",
                    flush=True,
                )
            if progress is not None:
                progress(event)
    return pd.DataFrame(records).sort_values(["C", "fold"]).reset_index(drop=True)


def selected_cv_cpg_coefficients(
    cv_results: pd.DataFrame,
    *,
    C: float,
) -> pd.DataFrame:
    selected = cv_results.loc[np.isclose(cv_results["C"], float(C))]
    records: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        for code, label in enumerate(CPG_DIAGNOSTIC_LABELS):
            coefficient = float(getattr(row, f"cpg_bin_coefficient_{code}"))
            records.append(
                {
                    "fold": int(row.fold),
                    "cpg_bin_code": code,
                    "cpg_oe_bin": label,
                    "is_reference": code == CPG_REFERENCE_BIN,
                    "coefficient": coefficient,
                    "odds_multiplier": float(np.exp(coefficient)),
                }
            )
    return pd.DataFrame(records)


def cpg_enrichment_table(
    counts: KmerCpGCounts,
    *,
    contigs: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Train-only support and enrichment for the fitted fixed bins."""

    summary = _joint_summary(counts, contigs=contigs)
    grouped = summary.groupby("cpg_bin_code", as_index=False).agg(
        positions=("total_positions", "sum"),
        positives=("positive_positions", "sum"),
    )
    full = pd.DataFrame(
        {"cpg_bin_code": np.arange(len(CPG_DIAGNOSTIC_LABELS))}
    )
    grouped = full.merge(grouped, on="cpg_bin_code", how="left").fillna(0)
    grouped[["positions", "positives"]] = grouped[
        ["positions", "positives"]
    ].astype(np.int64)
    grouped["cpg_oe_bin"] = np.asarray(CPG_DIAGNOSTIC_LABELS, dtype=object)[
        grouped["cpg_bin_code"].to_numpy(dtype=np.int8)
    ]
    grouped["positive_rate"] = np.divide(
        grouped["positives"],
        grouped["positions"],
        out=np.full(len(grouped), np.nan),
        where=grouped["positions"].to_numpy() > 0,
    )
    prevalence = summary["positive_positions"].sum() / summary["total_positions"].sum()
    grouped["enrichment"] = grouped["positive_rate"] / prevalence
    return grouped[
        [
            "cpg_bin_code",
            "cpg_oe_bin",
            "positions",
            "positives",
            "positive_rate",
            "enrichment",
        ]
    ]


__all__ = [
    "CPG_C_GRID",
    "CPG_DIAGNOSTIC_EDGES",
    "CPG_DIAGNOSTIC_LABELS",
    "AggregateCpGLikelihood",
    "CpGLogisticConfig",
    "CpGLogisticFit",
    "KmerCpGCounts",
    "aggregate_cpg_likelihood",
    "cpg_enrichment_table",
    "cpg_bin_codes",
    "cpg_bin_feature_table",
    "cpg_observed_expected",
    "cross_validate_cpg_logistic",
    "evaluate_cpg_logistic",
    "fit_cpg_logistic",
    "kmer_counts_from_cpg_counts",
    "selected_cv_cpg_coefficients",
    "split_kmer_cpg_counts",
]
