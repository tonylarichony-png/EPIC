"""EXP-011: exact bounded-memory GG + TA sufficient statistics.

The train scan is collapsed directly into the three frozen CV folds.  It never
retains one large table per contig/strand, while preserving exactly the same
weighted logistic likelihood as a position-level table.  Importing this module
performs no counting or fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize
from scipy.special import expit

from ml_project.epic_cpg_logistic import (
    _add_count_part,
    _collapse_count_parts,
    _merge_sorted_counts,
    _oriented_letters,
)
from ml_project.epic_baseline import _dense_rank_correlation
from ml_project.epic_gc_binned_logistic import (
    GC_BIN_LABELS,
    GCWindowEvaluation,
    gc_bin_feature_table,
)
from ml_project.epic_gc_logistic import _average_precision_from_rows
from ml_project.epic_kmer_logistic import HierarchicalKmerDesign, hierarchical_kmer_design
from ml_project.epic_kmer import kmer_category_count
from ml_project.epic_data import PreparedSplit
from ml_project.epic_experiments.exp_010_regionalnye_dinukleotidy_v_okne_201_bp import (
    DINUCLEOTIDE_LABELS,
    DINUCLEOTIDE_REFERENCE_BIN,
    RegionalDinucleotideConfig,
    _feature_keys_for_batch,
    dinucleotide_feature_table,
)
from ml_project.epic_cpg_logistic import cpg_bin_feature_table


EXPERIMENT_ID = "EXP-011"
CHAMPION_EXPERIMENT = "EXP-009"
CHAMPION_RUN = "run_001"
PAIRS = ("GG", "TA")
VARIANTS = ("EXP-009", "EXP-009+GG", "EXP-009+TA", "EXP-009+GG+TA")
CANDIDATE_VARIANTS = VARIANTS[1:]
FIXED_C = 1e-4
FEATURE_READY = True
CV_READY = True


@dataclass(frozen=True)
class GGTAConfig(RegionalDinucleotideConfig):
    """Frozen settings inherited from EXP-009 and EXP-010."""

    def to_dict(self) -> dict[str, object]:
        result = super().to_dict()
        result.update({
            "pairs": list(PAIRS),
            "variants": list(VARIANTS),
            "joint_interaction_columns": 0,
            "combined_extra_columns": 14,
            "combined_total_columns": 4400,
            "new_cv_fits": 9,
        })
        return result


@dataclass(frozen=True)
class GGTAJointCounts:
    """Exact joint counts partitioned only by the groups needed downstream."""

    split_name: str
    group_kind: str
    totals_by_group: Mapping[int | str, pd.DataFrame]
    group_assignment: pd.DataFrame
    positives: pd.DataFrame

    @property
    def represented_positions(self) -> int:
        return sum(
            int(frame["total_positions"].sum())
            for frame in self.totals_by_group.values()
        )


@dataclass(frozen=True)
class VariantSummary:
    """One exact total/positive counter for a fixed model vocabulary."""

    variant: str
    packed_keys: np.ndarray
    total_positions: np.ndarray
    positive_positions: np.ndarray


@dataclass(frozen=True)
class StreamingLogisticFit:
    """Compact result of exact L-BFGS without a position-level sparse matrix."""

    variant: str
    intercept: float
    coefficients: np.ndarray
    design: HierarchicalKmerDesign
    optimization_result: OptimizeResult
    coefficient_table: pd.DataFrame
    fit_report: pd.DataFrame


@dataclass(frozen=True)
class CVGateResult:
    summary: pd.DataFrame
    paired: pd.DataFrame
    checks: pd.Series
    passed: bool
    best_single_variant: str


def feature_contract() -> dict[str, Any]:
    return {
        "champion": f"{CHAMPION_EXPERIMENT}/{CHAMPION_RUN}",
        "variants": VARIANTS,
        "formula": "pair_count * valid_count / (left_count * right_count)",
        "window": "strand-oriented inclusive [-100,+100]",
        "pairs": PAIRS,
        "bins": DINUCLEOTIDE_LABELS,
        "aggregation_key": (
            "contig x strand x 6-mer x GC-bin x CpG-bin x GG-bin x TA-bin"
        ),
        "combined_columns": 4400,
        "interaction_columns": 0,
        "candidate_C": FIXED_C,
        "new_cv_fits": 9,
        "validation": "closed until paired CV gate passes",
        "train_aggregation": "exact streaming aggregation directly by frozen CV fold",
        "validation_aggregation": "exact streaming aggregation by contig",
        "current_stage": "full exact count enabled; CV remains gated",
    }


def _joint_keys_for_batch(
    letters: np.ndarray,
    start: int,
    end: int,
    config: GGTAConfig,
) -> np.ndarray:
    """Combine two EXP-010 keys without changing either feature definition."""

    gg = _feature_keys_for_batch(letters, start, end, "GG", config)
    ta = _feature_keys_for_batch(letters, start, end, "TA", config)
    champion = gg // len(DINUCLEOTIDE_LABELS)
    if not np.array_equal(champion, ta // len(DINUCLEOTIDE_LABELS)):
        raise AssertionError("GG and TA disagree on their EXP-009 base key")
    gg_bin = gg % len(DINUCLEOTIDE_LABELS)
    ta_bin = ta % len(DINUCLEOTIDE_LABELS)
    return (
        (champion * len(DINUCLEOTIDE_LABELS) + gg_bin)
        * len(DINUCLEOTIDE_LABELS)
        + ta_bin
    )


def _count_oriented_ranges(
    letters: np.ndarray,
    ranges: Sequence[tuple[int, int]],
    config: GGTAConfig,
) -> pd.DataFrame:
    """Count exact joint tuples with bounded chunk memory."""

    levels: list[tuple[np.ndarray, np.ndarray] | None] = []
    for interval_start, interval_end in ranges:
        cursor, stop = int(interval_start), int(interval_end)
        if not 0 <= cursor < stop <= len(letters):
            raise ValueError("whitelist interval is outside FASTA")
        while cursor < stop:
            end = min(stop, cursor + config.count_batch_size)
            keys, counts = np.unique(
                _joint_keys_for_batch(letters, cursor, end, config),
                return_counts=True,
            )
            _add_count_part(
                levels,
                keys.astype(np.int64, copy=False),
                counts.astype(np.int64, copy=False),
            )
            cursor = end

    packed, population = _collapse_count_parts(levels)
    size = len(DINUCLEOTIDE_LABELS)
    remainder = packed.copy()
    ta_bin = (remainder % size).astype(np.int8)
    remainder //= size
    gg_bin = (remainder % size).astype(np.int8)
    remainder //= size
    cpg_bin = (remainder % size).astype(np.int8)
    remainder //= size
    gc_bin = (remainder % 9).astype(np.int8)
    remainder //= 9
    return pd.DataFrame({
        "kmer_6_code": remainder.astype(np.int32),
        "gc_bin_code": gc_bin,
        "cpg_bin_code": cpg_bin,
        "gg_bin_code": gg_bin,
        "ta_bin_code": ta_bin,
        "total_positions": population.astype(np.int64),
    })


def _pack_count_frame(frame: pd.DataFrame) -> np.ndarray:
    """Recreate sorted joint keys from their compact decoded columns."""

    packed = frame["kmer_6_code"].to_numpy(dtype=np.int64, copy=True)
    packed = packed * 9 + frame["gc_bin_code"].to_numpy(dtype=np.int64)
    size = len(DINUCLEOTIDE_LABELS)
    packed = packed * size + frame["cpg_bin_code"].to_numpy(dtype=np.int64)
    packed = packed * size + frame["gg_bin_code"].to_numpy(dtype=np.int64)
    packed = packed * size + frame["ta_bin_code"].to_numpy(dtype=np.int64)
    return packed


def _positive_rows_joint(
    sequences: Mapping[str, np.ndarray],
    positives: pd.DataFrame,
    config: GGTAConfig,
) -> pd.DataFrame:
    """Calculate the two candidate bins only for observed positive positions."""

    from ml_project.epic_experiments.exp_010_regionalnye_dinukleotidy_v_okne_201_bp import (
        _positive_rows,
    )

    gg = _positive_rows(sequences, positives, "GG", config)
    ta = _positive_rows(sequences, positives, "TA", config)
    identity = ["template_index", "contig", "coordinate_0based", "strand", "count"]
    base = ["kmer_6_code", "gc_bin_code", "cpg_bin_code"]
    for column in (*identity, *base):
        if not gg[column].equals(ta[column]):
            raise AssertionError(f"GG and TA positive rows disagree on {column}")
    result = gg[identity + base].copy()
    result["gg_bin_code"] = gg["dinucleotide_bin_code"].to_numpy(dtype=np.int8)
    result["ta_bin_code"] = ta["dinucleotide_bin_code"].to_numpy(dtype=np.int8)
    return result


def _validate_positive_support(counts: GGTAJointCounts) -> None:
    keys = ["kmer_6_code", "gc_bin_code", "cpg_bin_code", "gg_bin_code", "ta_bin_code"]
    for group, totals in counts.totals_by_group.items():
        local = counts.positives.loc[counts.positives["group"] == group]
        positive = local.groupby(keys, observed=True).size().rename("positive_positions")
        if positive.empty:
            continue
        positive_frame = positive.reset_index()
        positive_keys = _pack_count_frame(positive_frame)
        total_keys = _pack_count_frame(totals)
        locations = np.searchsorted(total_keys, positive_keys)
        present = locations < len(total_keys)
        present_indices = np.flatnonzero(present)
        present[present_indices] = (
            total_keys[locations[present_indices]] == positive_keys[present_indices]
        )
        if not present.all():
            raise AssertionError("positive tuple is absent from total counts")
        available = totals["total_positions"].to_numpy(dtype=np.int64)[locations]
        if np.any(positive_frame["positive_positions"].to_numpy(dtype=np.int64) > available):
            raise AssertionError("positive tuple count exceeds total positions")


def count_candidate(
    split: PreparedSplit,
    sequences: Mapping[str, np.ndarray],
    split_name: str,
    *,
    fold_assignment: pd.DataFrame | None = None,
    config: GGTAConfig = GGTAConfig(),
    verbose: bool = False,
    **_: Any,
) -> GGTAJointCounts:
    """Count exact GG+TA tuples, merging each strand immediately into its group."""

    if split_name not in {"train", "validation", "test"}:
        raise ValueError("EXP-011 labelled counts require train, validation, or test")
    intervals = split.intervals(split_name).copy()
    positives = split.positive_targets(split_name)
    if positives is None or positives.empty:
        raise ValueError(f"{split_name} positives are unavailable")
    contigs = sorted(intervals["contig"].astype(str).unique())
    missing_sequences = sorted(set(contigs) - set(sequences))
    if missing_sequences:
        raise ValueError("split contigs missing from FASTA: " + ", ".join(missing_sequences))

    if split_name == "train":
        if fold_assignment is None:
            raise ValueError("train count requires EXP-009 cv_fold_assignment.csv")
        required = {"contig", "fold"}
        if not required.issubset(fold_assignment.columns):
            raise ValueError("fold_assignment requires contig and fold columns")
        mapping = dict(zip(
            fold_assignment["contig"].astype(str),
            fold_assignment["fold"].astype(int),
        ))
        if set(mapping) != set(contigs):
            raise ValueError("fold_assignment must contain every train contig exactly once")
        if sorted(set(mapping.values())) != [0, 1, 2]:
            raise ValueError("EXP-011 requires the three frozen folds 0, 1, 2")
        group_kind = "fold"
        assignment = fold_assignment[["contig", "fold"]].copy()
        assignment["contig"] = assignment["contig"].astype(str)
    else:
        mapping = {contig: contig for contig in contigs}
        group_kind = "contig"
        assignment = pd.DataFrame({"contig": contigs, "group": contigs})

    levels_by_group: dict[int | str, list[tuple[np.ndarray, np.ndarray] | None]] = {
        group: [] for group in sorted(set(mapping.values()), key=str)
    }
    parts = []
    for contig in contigs:
        local = intervals.loc[intervals["contig"].astype(str) == contig]
        for strand in ("+", "-"):
            if not local.loc[local["strand"] == strand].empty:
                parts.append((contig, strand))
    started = time.monotonic()
    for completed, (contig, strand) in enumerate(parts, 1):
        frame = intervals.loc[
            (intervals["contig"].astype(str) == contig)
            & (intervals["strand"] == strand),
            ["start", "end"],
        ]
        ranges = [tuple(map(int, row)) for row in frame.to_numpy()]
        letters = _oriented_letters(sequences[contig], strand)
        if strand == "-":
            ranges = [(len(letters) - end, len(letters) - start) for start, end in ranges]
        part = _count_oriented_ranges(letters, ranges, config)
        packed = _pack_count_frame(part)
        # pandas 3 may expose a read-only view under Copy-on-Write.  The binary
        # counter intentionally increments this array when a later contig or
        # strand contributes the same packed key, so it must be owned/writable.
        population = part["total_positions"].to_numpy(dtype=np.int64, copy=True)
        group = mapping[contig]
        _add_count_part(levels_by_group[group], packed, population)
        if verbose:
            retained = sum(
                len(item[0]) for item in levels_by_group[group] if item is not None
            )
            print(
                f"[{completed:02d}/{len(parts):02d}] group={group} {contig} "
                f"strand={strand} positions={int(population.sum()):,} "
                f"part_rows={len(part):,} pending_group_rows={retained:,} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

    totals_by_group: dict[int | str, pd.DataFrame] = {}
    for group, levels in levels_by_group.items():
        packed, population = _collapse_count_parts(levels)
        size = len(DINUCLEOTIDE_LABELS)
        remainder = packed.copy()
        ta_bin = (remainder % size).astype(np.int8); remainder //= size
        gg_bin = (remainder % size).astype(np.int8); remainder //= size
        cpg_bin = (remainder % size).astype(np.int8); remainder //= size
        gc_bin = (remainder % 9).astype(np.int8); remainder //= 9
        totals_by_group[group] = pd.DataFrame({
            "kmer_6_code": remainder.astype(np.int32),
            "gc_bin_code": gc_bin,
            "cpg_bin_code": cpg_bin,
            "gg_bin_code": gg_bin,
            "ta_bin_code": ta_bin,
            "total_positions": population.astype(np.int64, copy=False),
        })

    positive_rows = _positive_rows_joint(sequences, positives, config)
    positive_rows["group"] = positive_rows["contig"].astype(str).map(mapping)
    if positive_rows["group"].isna().any():
        raise AssertionError("positive contig has no aggregation group")
    if group_kind == "fold":
        positive_rows["group"] = positive_rows["group"].astype(np.int8)
    result = GGTAJointCounts(
        split_name, group_kind, totals_by_group, assignment, positive_rows
    )
    expected = int((intervals["end"] - intervals["start"]).sum())
    if result.represented_positions != expected:
        raise AssertionError("joint counts do not preserve the split population")
    _validate_positive_support(result)
    return result


def count_diagnostics(counts: GGTAJointCounts) -> pd.DataFrame:
    """Small table suitable for inspecting memory before CV is enabled."""

    records = []
    for group, totals in counts.totals_by_group.items():
        positions = int(totals["total_positions"].sum())
        records.append({
            "split": counts.split_name,
            "group_kind": counts.group_kind,
            "group": group,
            "positions": positions,
            "joint_rows": len(totals),
            "unique_rows_per_position": len(totals) / positions,
            "totals_memory_mib": totals.memory_usage(deep=True).sum() / 2**20,
            "positive_rows": int((counts.positives["group"] == group).sum()),
        })
    result = pd.DataFrame(records)
    result.loc[len(result)] = {
        "split": counts.split_name,
        "group_kind": "all",
        "group": "TOTAL",
        "positions": int(result["positions"].sum()),
        "joint_rows": int(result["joint_rows"].sum()),
        "unique_rows_per_position": result["joint_rows"].sum() / result["positions"].sum(),
        "totals_memory_mib": float(result["totals_memory_mib"].sum()),
        "positive_rows": int(result["positive_rows"].sum()),
    }
    return result


def _keys_for_variant(full_keys: np.ndarray, variant: str) -> np.ndarray:
    """Marginalize the unused pair while retaining the champion key."""

    full = np.asarray(full_keys, dtype=np.int64)
    if variant == "EXP-009+GG+TA":
        return full
    if variant == "EXP-009+GG":
        return full // len(DINUCLEOTIDE_LABELS)
    if variant == "EXP-009+TA":
        champion = full // (len(DINUCLEOTIDE_LABELS) ** 2)
        ta_bin = full % len(DINUCLEOTIDE_LABELS)
        return champion * len(DINUCLEOTIDE_LABELS) + ta_bin
    raise ValueError(f"unknown EXP-011 variant: {variant}")


def _collapse_weighted_keys(
    keys: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.asarray(keys, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.int64)
    if len(keys) != len(weights):
        raise ValueError("keys and weights must have equal length")
    if not len(keys):
        return keys.copy(), weights.copy()
    if np.any(keys[1:] < keys[:-1]):
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        weights = weights[order]
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    return keys[starts].copy(), np.add.reduceat(weights, starts)


def _group_variant_summary(
    counts: GGTAJointCounts, group: int | str, variant: str
) -> VariantSummary:
    totals = counts.totals_by_group[group]
    total_keys, total_values = _collapse_weighted_keys(
        _keys_for_variant(_pack_count_frame(totals), variant),
        totals["total_positions"].to_numpy(dtype=np.int64),
    )
    positive_rows = counts.positives.loc[counts.positives["group"] == group]
    positive_full = _pack_count_frame(positive_rows)
    positive_raw = _keys_for_variant(positive_full, variant)
    positive_keys, positive_values = np.unique(positive_raw, return_counts=True)
    locations = np.searchsorted(total_keys, positive_keys)
    if np.any(locations >= len(total_keys)) or not np.array_equal(
        total_keys[locations], positive_keys
    ):
        raise AssertionError("positive variant tuple is absent from totals")
    aligned_positive = np.zeros(len(total_keys), dtype=np.int64)
    aligned_positive[locations] = positive_values.astype(np.int64, copy=False)
    if np.any(aligned_positive > total_values):
        raise AssertionError("positive variant count exceeds total")
    return VariantSummary(
        variant, total_keys, total_values.astype(np.int64, copy=False), aligned_positive
    )


def _merge_summaries(
    summaries: Sequence[VariantSummary], variant: str
) -> VariantSummary:
    if not summaries:
        raise ValueError("at least one summary is required")
    total_keys = np.array([], dtype=np.int64)
    total_values = np.array([], dtype=np.int64)
    positive_keys = np.array([], dtype=np.int64)
    positive_values = np.array([], dtype=np.int64)
    for summary in summaries:
        if summary.variant != variant:
            raise ValueError("cannot merge different variants")
        total_keys, total_values = _merge_sorted_counts(
            total_keys, total_values,
            summary.packed_keys, summary.total_positions.copy(),
        )
        local_positive = summary.positive_positions > 0
        positive_keys, positive_values = _merge_sorted_counts(
            positive_keys, positive_values,
            summary.packed_keys[local_positive],
            summary.positive_positions[local_positive].copy(),
        )
    locations = np.searchsorted(total_keys, positive_keys)
    aligned = np.zeros(len(total_keys), dtype=np.int64)
    aligned[locations] = positive_values
    return VariantSummary(variant, total_keys, total_values, aligned)


def _decode_variant_keys(
    keys: np.ndarray, variant: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    remainder = np.asarray(keys, dtype=np.int64).copy()
    gg: np.ndarray | None = None
    ta: np.ndarray | None = None
    size = len(DINUCLEOTIDE_LABELS)
    if variant == "EXP-009+GG+TA":
        ta = (remainder % size).astype(np.int8); remainder //= size
        gg = (remainder % size).astype(np.int8); remainder //= size
    elif variant == "EXP-009+GG":
        gg = (remainder % size).astype(np.int8); remainder //= size
    elif variant == "EXP-009+TA":
        ta = (remainder % size).astype(np.int8); remainder //= size
    else:
        raise ValueError(f"unknown EXP-011 variant: {variant}")
    cpg = (remainder % 8).astype(np.int8); remainder //= 8
    gc = (remainder % 9).astype(np.int8); remainder //= 9
    return remainder.astype(np.int32), gc, cpg, gg, ta


def _nonreference_codes(size: int, reference: int) -> np.ndarray:
    return np.asarray([code for code in range(size) if code != reference], dtype=np.int8)


def _parameter_layout(
    design: HierarchicalKmerDesign, variant: str, config: GGTAConfig
) -> dict[str, slice]:
    cursor = design.matrix.shape[1]
    result = {"kmer": slice(0, cursor)}
    result["gc"] = slice(cursor, cursor + len(GC_BIN_LABELS) - 1); cursor += 8
    result["cpg"] = slice(cursor, cursor + 7); cursor += 7
    if "GG" in variant:
        result["gg"] = slice(cursor, cursor + 7); cursor += 7
    if "TA" in variant:
        result["ta"] = slice(cursor, cursor + 7); cursor += 7
    result["all"] = slice(0, cursor)
    return result


def _code_coefficient_lookup(
    coefficients: np.ndarray, local_slice: slice, size: int, reference: int
) -> np.ndarray:
    result = np.zeros(size, dtype=np.float64)
    result[_nonreference_codes(size, reference)] = coefficients[local_slice]
    return result


def _kmer_level_structure(
    design: HierarchicalKmerDesign,
) -> list[tuple[slice, np.ndarray]]:
    """Return coefficient slices and max-k -> local-code lookup without sparse BLAS."""

    result: list[tuple[slice, np.ndarray]] = []
    cursor = 0
    for k in design.ks:
        width = kmer_category_count(k)
        codes = design.category_table[f"kmer_{k}_code"].to_numpy(dtype=np.int32)
        result.append((slice(cursor, cursor + width), codes))
        cursor += width
    if cursor != design.matrix.shape[1]:
        raise AssertionError("hierarchical k-mer layout width mismatch")
    return result


def _kmer_category_scores(
    coefficients: np.ndarray,
    structure: Sequence[tuple[slice, np.ndarray]],
) -> np.ndarray:
    scores = np.zeros(len(structure[0][1]), dtype=np.float64)
    for local_slice, local_codes in structure:
        scores += coefficients[local_slice][local_codes]
    return scores


def _streaming_coefficient_table(
    coefficients: np.ndarray,
    design: HierarchicalKmerDesign,
    variant: str,
    config: GGTAConfig,
) -> pd.DataFrame:
    """Describe every fitted coefficient using the established report schema."""

    layout = _parameter_layout(design, variant, config)
    tables: list[pd.DataFrame] = []
    kmer = design.feature_table.copy()
    kmer["feature_group"] = kmer["level_k"].map(lambda value: f"{int(value)}-mer")
    tables.append(kmer[[
        "feature_index", "level_k", "category_code", "category",
        "feature_name", "feature_group",
    ]])

    def add_bins(table: pd.DataFrame, within: str, local_slice: slice) -> None:
        local = table.copy()
        local["feature_index"] = local_slice.start + local[within].astype(int)
        local["level_k"] = pd.NA
        local["category_code"] = local["bin_code"]
        local["category"] = local["bin_label"]
        tables.append(local[[
            "feature_index", "level_k", "category_code", "category",
            "feature_name", "feature_group",
        ]])

    add_bins(gc_bin_feature_table(config), "feature_index_within_gc", layout["gc"])
    add_bins(cpg_bin_feature_table(config), "feature_index_within_cpg", layout["cpg"])
    if "gg" in layout:
        add_bins(dinucleotide_feature_table("GG"), "feature_index_within_pair", layout["gg"])
    if "ta" in layout:
        add_bins(dinucleotide_feature_table("TA"), "feature_index_within_pair", layout["ta"])
    result = pd.concat(tables, ignore_index=True).sort_values("feature_index")
    result["coefficient"] = coefficients[result["feature_index"].to_numpy(dtype=int)]
    result["absolute_coefficient"] = np.abs(result["coefficient"])
    return result.reset_index(drop=True)


def _score_codes(
    coefficients: np.ndarray,
    intercept: float,
    design: HierarchicalKmerDesign,
    layout: Mapping[str, slice],
    codes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None],
    config: GGTAConfig,
) -> np.ndarray:
    kmer, gc, cpg, gg, ta = codes
    kmer_scores = _kmer_category_scores(
        coefficients[layout["kmer"]], _kmer_level_structure(design)
    )
    score = intercept + kmer_scores[kmer]
    score = score + _code_coefficient_lookup(
        coefficients, layout["gc"], len(GC_BIN_LABELS), config.reference_bin
    )[gc]
    score = score + _code_coefficient_lookup(
        coefficients, layout["cpg"], 8, config.cpg_reference_bin
    )[cpg]
    if gg is not None:
        score = score + _code_coefficient_lookup(
            coefficients, layout["gg"], 8, DINUCLEOTIDE_REFERENCE_BIN
        )[gg]
    if ta is not None:
        score = score + _code_coefficient_lookup(
            coefficients, layout["ta"], 8, DINUCLEOTIDE_REFERENCE_BIN
        )[ta]
    return score


def _fit_streaming_summary(
    summary: VariantSummary,
    *,
    config: GGTAConfig,
    design: HierarchicalKmerDesign,
    batch_size: int = 500_000,
    verbose: bool = False,
    progress_label: str = "fit",
) -> StreamingLogisticFit:
    """Minimize the exact balanced L2 logistic objective in bounded chunks."""

    codes = _decode_variant_keys(summary.packed_keys, summary.variant)
    layout = _parameter_layout(design, summary.variant, config)
    feature_count = layout["all"].stop
    total = summary.total_positions.astype(np.float64, copy=False)
    positive = summary.positive_positions.astype(np.float64, copy=False)
    population = float(total.sum())
    positive_population = float(positive.sum())
    negative_population = population - positive_population
    if positive_population <= 0 or negative_population <= 0:
        raise ValueError("training summary must contain both classes")
    negative_multiplier = population / (2.0 * negative_population)
    positive_multiplier = population / (2.0 * positive_population)
    regularization = 1.0 / (config.candidate_C * population)
    gc_active = _nonreference_codes(len(GC_BIN_LABELS), config.reference_bin)
    cpg_active = _nonreference_codes(8, config.cpg_reference_bin)
    pair_active = _nonreference_codes(8, DINUCLEOTIDE_REFERENCE_BIN)
    kmer_structure = _kmer_level_structure(design)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = float(parameters[0])
        coefficients = parameters[1:]
        kmer_coefficients = coefficients[layout["kmer"]]
        kmer_scores = _kmer_category_scores(kmer_coefficients, kmer_structure)
        gc_scores = _code_coefficient_lookup(
            coefficients, layout["gc"], len(GC_BIN_LABELS), config.reference_bin
        )
        cpg_scores = _code_coefficient_lookup(
            coefficients, layout["cpg"], 8, config.cpg_reference_bin
        )
        gg_scores = (_code_coefficient_lookup(
            coefficients, layout["gg"], 8, DINUCLEOTIDE_REFERENCE_BIN
        ) if codes[3] is not None else None)
        ta_scores = (_code_coefficient_lookup(
            coefficients, layout["ta"], 8, DINUCLEOTIDE_REFERENCE_BIN
        ) if codes[4] is not None else None)
        gradient = np.zeros(feature_count + 1, dtype=np.float64)
        loss = 0.0
        for start in range(0, len(total), batch_size):
            end = min(len(total), start + batch_size)
            kmer, gc, cpg, gg, ta = (value[start:end] if value is not None else None
                                      for value in codes)
            score = intercept + kmer_scores[kmer] + gc_scores[gc] + cpg_scores[cpg]
            if gg is not None:
                score = score + gg_scores[gg]
            if ta is not None:
                score = score + ta_scores[ta]
            weighted_negative = (total[start:end] - positive[start:end]) * negative_multiplier
            weighted_positive = positive[start:end] * positive_multiplier
            loss += float(np.sum(
                weighted_negative * np.logaddexp(0.0, score)
                + weighted_positive * np.logaddexp(0.0, -score)
            ))
            residual = (
                (weighted_negative + weighted_positive) * expit(score)
                - weighted_positive
            )
            gradient[0] += residual.sum()
            category_gradient = np.bincount(
                kmer, weights=residual, minlength=len(kmer_structure[0][1])
            )
            kmer_gradient = gradient[
                1 + layout["kmer"].start:1 + layout["kmer"].stop
            ]
            for local_slice, local_codes in kmer_structure:
                kmer_gradient[local_slice] += np.bincount(
                    local_codes, weights=category_gradient,
                    minlength=local_slice.stop - local_slice.start,
                )
            gc_gradient = np.bincount(gc, weights=residual, minlength=len(GC_BIN_LABELS))
            gradient[1 + layout["gc"].start:1 + layout["gc"].stop] += gc_gradient[gc_active]
            cpg_gradient = np.bincount(cpg, weights=residual, minlength=8)
            gradient[1 + layout["cpg"].start:1 + layout["cpg"].stop] += cpg_gradient[cpg_active]
            if gg is not None:
                gg_gradient = np.bincount(gg, weights=residual, minlength=8)
                gradient[1 + layout["gg"].start:1 + layout["gg"].stop] += gg_gradient[pair_active]
            if ta is not None:
                ta_gradient = np.bincount(ta, weights=residual, minlength=8)
                gradient[1 + layout["ta"].start:1 + layout["ta"].stop] += ta_gradient[pair_active]
        loss /= population
        gradient /= population
        loss += 0.5 * regularization * float(coefficients @ coefficients)
        gradient[1:] += regularization * coefficients
        return loss, gradient

    initial = np.zeros(feature_count + 1, dtype=np.float64)
    started = time.monotonic()
    callback_iterations = 0

    def callback(_: np.ndarray) -> None:
        nonlocal callback_iterations
        callback_iterations += 1
        if verbose and (callback_iterations == 1 or callback_iterations % 10 == 0):
            print(
                f"    {progress_label} optimizer_iter={callback_iterations} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        callback=callback,
        options={"maxiter": config.max_iter, "ftol": config.tol, "gtol": config.tol},
    )
    elapsed = time.monotonic() - started
    report = pd.DataFrame([{
        "variant": summary.variant,
        "C": config.candidate_C,
        "optimizer": "scipy L-BFGS-B exact aggregated objective",
        "iterations": int(result.nit),
        "max_iter": config.max_iter,
        "converged": bool(result.success),
        "fit_seconds": elapsed,
        "joint_rows": len(summary.packed_keys),
        "features": feature_count,
        "represented_positions": int(population),
        "represented_positives": int(positive_population),
        "final_objective": float(result.fun),
        "optimizer_message": str(result.message),
    }])
    coefficients = result.x[1:].copy()
    coefficient_table = _streaming_coefficient_table(
        coefficients, design, summary.variant, config
    )
    return StreamingLogisticFit(
        summary.variant, float(result.x[0]), coefficients, design, result,
        coefficient_table, report,
    )


def _evaluate_summary(
    summary: VariantSummary,
    fit: StreamingLogisticFit,
    positive_rows: pd.DataFrame,
    *,
    config: GGTAConfig,
) -> dict[str, Any]:
    layout = _parameter_layout(fit.design, fit.variant, config)
    scores = _score_codes(
        fit.coefficients, fit.intercept, fit.design, layout,
        _decode_variant_keys(summary.packed_keys, fit.variant), config,
    )
    average_precision, _ = _average_precision_from_rows(
        summary.total_positions, summary.positive_positions, scores, include_curve=False
    )
    positive_keys = _keys_for_variant(_pack_count_frame(positive_rows), fit.variant)
    positive_scores = _score_codes(
        fit.coefficients, fit.intercept, fit.design, layout,
        _decode_variant_keys(positive_keys, fit.variant), config,
    )
    spearman = _dense_rank_correlation(
        positive_rows["count"].to_numpy(), positive_scores
    )
    return {
        "average_precision": float(average_precision),
        "epic_spearman": float(spearman),
        "positions": int(summary.total_positions.sum()),
        "positives": int(summary.positive_positions.sum()),
        "prevalence": float(summary.positive_positions.sum() / summary.total_positions.sum()),
    }


def cross_validate_candidate(
    train_counts: GGTAJointCounts,
    *,
    champion_dir: str | Any,
    config: GGTAConfig = GGTAConfig(),
    verbose: bool = False,
    **_: Any,
) -> pd.DataFrame:
    """Fit the three preregistered candidates on the frozen three-fold split."""

    if train_counts.split_name != "train" or train_counts.group_kind != "fold":
        raise ValueError("EXP-011 CV requires fold-aggregated train counts")
    folds = sorted(int(value) for value in train_counts.totals_by_group)
    if folds != [0, 1, 2]:
        raise ValueError("EXP-011 CV requires folds 0, 1, 2")
    design = hierarchical_kmer_design(config)
    records: list[dict[str, Any]] = []
    completed = 0
    for variant in CANDIDATE_VARIANTS:
        group_summaries = {
            fold: _group_variant_summary(train_counts, fold, variant) for fold in folds
        }
        for held_out in folds:
            training = _merge_summaries(
                [group_summaries[fold] for fold in folds if fold != held_out], variant
            )
            fitted = _fit_streaming_summary(
                training, config=config, design=design, verbose=verbose,
                progress_label=f"variant={variant} fold={held_out}",
            )
            held_rows = train_counts.positives.loc[
                train_counts.positives["group"] == held_out
            ]
            metrics = _evaluate_summary(
                group_summaries[held_out], fitted, held_rows, config=config
            )
            report = fitted.fit_report.iloc[0].to_dict()
            contigs = train_counts.group_assignment.loc[
                train_counts.group_assignment["fold"].astype(int) == held_out, "contig"
            ].astype(str)
            completed += 1
            record = {
                "variant": variant,
                "fold": held_out,
                "validation_contigs": ",".join(contigs),
                **metrics,
                **report,
            }
            records.append(record)
            if verbose:
                print(
                    f"[{completed:02d}/09] variant={variant} fold={held_out} "
                    f"C={config.candidate_C:g} AP={metrics['average_precision']:.9f} "
                    f"Spearman={metrics['epic_spearman']:.6f} "
                    f"iter={int(report['iterations'])} time={report['fit_seconds']:.1f}s "
                    f"converged={bool(report['converged'])}",
                    flush=True,
                )
            del training, fitted
    candidate = pd.DataFrame(records)
    reference = pd.read_csv(Path(champion_dir) / "cv_results.csv")
    reference = reference.loc[np.isclose(reference["C"], FIXED_C)].copy()
    if set(reference["fold"].astype(int)) != set(folds):
        raise ValueError("EXP-009 reference does not contain the frozen three folds")
    reference_rows = reference[[
        "fold", "validation_contigs", "average_precision", "epic_spearman",
        "positions", "positives", "prevalence", "iterations", "converged",
        "fit_seconds", "aggregate_rows",
    ]].copy()
    reference_rows.insert(0, "variant", "EXP-009")
    reference_rows["C"] = FIXED_C
    reference_rows["optimizer"] = "saved EXP-009 sklearn lbfgs"
    reference_rows["max_iter"] = config.max_iter
    reference_rows["joint_rows"] = reference_rows.pop("aggregate_rows")
    reference_rows["features"] = 4386
    reference_rows["represented_positions"] = reference_rows["positions"]
    reference_rows["represented_positives"] = reference_rows["positives"]
    reference_rows["final_objective"] = np.nan
    reference_rows["optimizer_message"] = "saved reference"
    columns = candidate.columns
    return pd.concat([reference_rows.reindex(columns=columns), candidate], ignore_index=True)


def summarize_cv_gate(cv_results: pd.DataFrame) -> CVGateResult:
    """Apply the preregistered EXP-011 complementarity gate."""

    required = {"variant", "fold", "average_precision", "epic_spearman", "converged"}
    if not required.issubset(cv_results.columns):
        raise ValueError("cv_results is missing columns required by the CV gate")
    expected = set(VARIANTS)
    if set(cv_results["variant"]) != expected:
        raise ValueError(f"CV requires exactly these variants: {sorted(expected)}")
    summary = (
        cv_results.groupby("variant", as_index=False)
        .agg(
            mean_average_precision=("average_precision", "mean"),
            std_average_precision=("average_precision", "std"),
            mean_epic_spearman=("epic_spearman", "mean"),
            std_epic_spearman=("epic_spearman", "std"),
            all_folds_converged=("converged", "all"),
            total_fit_seconds=("fit_seconds", "sum"),
        )
        .sort_values("mean_average_precision", ascending=False)
        .reset_index(drop=True)
    )
    singles = summary.loc[summary["variant"].isin(("EXP-009+GG", "EXP-009+TA"))]
    best_single = str(singles.iloc[0]["variant"])
    comparison = cv_results.loc[
        cv_results["variant"].isin(("EXP-009", best_single, "EXP-009+GG+TA")),
        ["variant", "fold", "average_precision", "epic_spearman"],
    ].pivot(index="fold", columns="variant")
    paired = pd.DataFrame({"fold": comparison.index.astype(int)})
    for variant in ("EXP-009", best_single, "EXP-009+GG+TA"):
        label = variant.lower().replace("-", "_").replace("+", "_plus_")
        paired[f"{label}_ap"] = comparison["average_precision", variant].to_numpy()
        paired[f"{label}_spearman"] = comparison["epic_spearman", variant].to_numpy()
    paired["combined_minus_best_single_ap"] = (
        comparison["average_precision", "EXP-009+GG+TA"].to_numpy()
        - comparison["average_precision", best_single].to_numpy()
    )
    combined = summary.set_index("variant").loc["EXP-009+GG+TA"]
    single = summary.set_index("variant").loc[best_single]
    reference = summary.set_index("variant").loc["EXP-009"]
    relative_gain = (
        combined["mean_average_precision"] / single["mean_average_precision"] - 1.0
    )
    checks = pd.Series({
        "combined_mean_ap_gain_vs_best_single_at_least_1pct": bool(relative_gain >= 0.01),
        "combined_ap_wins_at_least_2_of_3_folds": bool(
            (paired["combined_minus_best_single_ap"] > 0).sum() >= 2
        ),
        "combined_mean_spearman_not_below_best_single": bool(
            combined["mean_epic_spearman"] >= single["mean_epic_spearman"]
        ),
        "combined_mean_spearman_not_below_exp009": bool(
            combined["mean_epic_spearman"] >= reference["mean_epic_spearman"]
        ),
        "combined_all_folds_converged": bool(combined["all_folds_converged"]),
    }, name="passed")
    paired["best_single_variant"] = best_single
    paired["combined_relative_mean_ap_gain"] = float(relative_gain)
    return CVGateResult(summary, paired, checks, bool(checks.all()), best_single)


def fit_candidate(
    train_counts: GGTAJointCounts,
    *,
    selected_parameters: Mapping[str, Any] | None = None,
    config: GGTAConfig = GGTAConfig(),
    verbose: bool = False,
    **_: Any,
) -> StreamingLogisticFit:
    """Fit the train-only selected candidate on all three exact fold tables."""

    if train_counts.split_name != "train" or train_counts.group_kind != "fold":
        raise ValueError("final fit requires fold-aggregated train counts")
    selected = dict(selected_parameters or {})
    variant = str(selected.get("variant", "EXP-009+GG+TA"))
    if variant not in CANDIDATE_VARIANTS:
        raise ValueError(f"selected variant must be one of {CANDIDATE_VARIANTS}")
    selected_c = float(selected.get("C", config.candidate_C))
    if not np.isclose(selected_c, config.candidate_C):
        raise ValueError("EXP-011 final fit must use the preregistered C=0.0001")
    groups = sorted(train_counts.totals_by_group, key=str)
    if verbose:
        print(f"Preparing exact full-train summary for {variant} ...", flush=True)
    summaries = [
        _group_variant_summary(train_counts, group, variant) for group in groups
    ]
    combined = _merge_summaries(summaries, variant)
    if verbose:
        print(
            f"Full-train summary: positions={combined.total_positions.sum():,} "
            f"joint_rows={len(combined.packed_keys):,}",
            flush=True,
        )
    return _fit_streaming_summary(
        combined, config=config, design=hierarchical_kmer_design(config),
        verbose=verbose, progress_label=f"full-train {variant}",
    )


def _evaluate_group_selection(
    counts: GGTAJointCounts,
    fit: StreamingLogisticFit,
    groups: Sequence[int | str],
    *,
    config: GGTAConfig,
    include_curve: bool,
    segment: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries = [_group_variant_summary(counts, group, fit.variant) for group in groups]
    summary = _merge_summaries(summaries, fit.variant)
    layout = _parameter_layout(fit.design, fit.variant, config)
    scores = _score_codes(
        fit.coefficients, fit.intercept, fit.design, layout,
        _decode_variant_keys(summary.packed_keys, fit.variant), config,
    )
    average_precision, curve = _average_precision_from_rows(
        summary.total_positions, summary.positive_positions, scores,
        include_curve=include_curve,
    )
    positive_rows = counts.positives.loc[counts.positives["group"].isin(groups)].copy()
    positive_keys = _keys_for_variant(_pack_count_frame(positive_rows), fit.variant)
    positive_rows["score"] = _score_codes(
        fit.coefficients, fit.intercept, fit.design, layout,
        _decode_variant_keys(positive_keys, fit.variant), config,
    )
    positive_rows["segment"] = segment
    positive_rows["variant"] = fit.variant
    spearman = _dense_rank_correlation(
        positive_rows["count"].to_numpy(), positive_rows["score"].to_numpy()
    )
    positions = int(summary.total_positions.sum())
    positives = int(summary.positive_positions.sum())
    metric = pd.DataFrame([{
        "segment": segment,
        "variant": fit.variant,
        "average_precision": float(average_precision),
        "epic_spearman": float(spearman),
        "positions": positions,
        "positives": positives,
        "prevalence": positives / positions,
    }])
    if not curve.empty:
        curve["segment"] = segment
        curve["variant"] = fit.variant
    return metric, curve, positive_rows


def evaluate_candidate(
    counts: GGTAJointCounts,
    fit: StreamingLogisticFit,
    *,
    config: GGTAConfig = GGTAConfig(),
    include_curve: bool = True,
    include_per_contig: bool = True,
    **_: Any,
) -> GCWindowEvaluation:
    """Evaluate the frozen final model exactly on labelled validation counts."""

    if counts.split_name not in {"validation", "test"} or counts.group_kind != "contig":
        raise ValueError("EXP-011 candidate evaluation requires validation or test counts")
    groups = sorted(counts.totals_by_group, key=str)
    overall, curve, predictions = _evaluate_group_selection(
        counts, fit, groups, config=config, include_curve=include_curve,
        segment="overall",
    )
    if include_per_contig:
        per_contig = pd.concat([
            _evaluate_group_selection(
                counts, fit, [group], config=config, include_curve=False,
                segment=str(group),
            )[0]
            for group in groups
        ], ignore_index=True).rename(columns={"segment": "contig"})
    else:
        per_contig = pd.DataFrame()
    return GCWindowEvaluation(overall, per_contig, curve, predictions)


def diagnose_feature(
    split: PreparedSplit,
    sequences: Mapping[str, np.ndarray],
    *,
    config: GGTAConfig = GGTAConfig(),
    max_positions: int = 2_000_000,
    **_: Any,
) -> pd.DataFrame:
    """Measure joint cardinality on the largest train contig-strand."""

    intervals = split.intervals("train").copy()
    intervals["length"] = intervals["end"] - intervals["start"]
    contig, strand = (
        intervals.groupby(["contig", "strand"], observed=True)["length"]
        .sum()
        .idxmax()
    )
    contig, strand = str(contig), str(strand)
    local = intervals.loc[
        (intervals["contig"].astype(str) == contig)
        & (intervals["strand"] == strand),
        ["start", "end"],
    ]
    remaining = int(max_positions)
    ranges: list[tuple[int, int]] = []
    for start, end in local.to_numpy():
        take = min(int(end - start), remaining)
        if take <= 0:
            break
        ranges.append((int(start), int(start) + take))
        remaining -= take
    letters = _oriented_letters(sequences[contig], strand)
    if strand == "-":
        ranges = [(len(letters) - end, len(letters) - start)
                  for start, end in ranges]
    started = time.monotonic()
    totals = _count_oriented_ranges(letters, ranges, config)
    positions = int(totals["total_positions"].sum())
    return pd.DataFrame([{
        "pairs": "GG+TA",
        "pilot_contig": contig,
        "pilot_strand": strand,
        "positions": positions,
        "joint_rows": len(totals),
        "unique_rows_per_position": len(totals) / positions,
        "totals_memory_mib": totals.memory_usage(deep=True).sum() / 2**20,
        "seconds": time.monotonic() - started,
    }])


__all__ = [
    "CV_READY", "CVGateResult", "FEATURE_READY", "FIXED_C", "GGTAConfig",
    "GGTAJointCounts", "StreamingLogisticFit",
    "PAIRS", "VARIANTS", "count_candidate", "count_diagnostics",
    "cross_validate_candidate", "diagnose_feature", "summarize_cv_gate",
    "evaluate_candidate", "feature_contract", "fit_candidate",
]
