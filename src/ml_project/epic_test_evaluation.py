"""Explicit, read-only access to the labelled Nematostella demo test.

The ordinary :mod:`epic_data` contract deliberately keeps test targets absent.
This module is the only bridge to ``ground_truth/csRNA-test.txt.gz`` and is
intended for a final frozen-model evaluation, never for training or selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from ml_project.epic_data import PreparedSplit, sha256_file


EXPECTED_TEST_POSITIONS = 72_613_390


@dataclass(frozen=True)
class LabeledTestData:
    """Minimal split-like object accepted by feature counters at test time."""

    intervals_table: pd.DataFrame
    positive_targets_table: pd.DataFrame
    diagnostics: pd.DataFrame
    ground_truth_path: Path
    ground_truth_sha256: str

    def intervals(self, split_name: str) -> pd.DataFrame:
        if split_name != "test":
            raise ValueError("LabeledTestData exposes only the frozen test split")
        return self.intervals_table

    def positive_targets(self, split_name: str) -> pd.DataFrame:
        if split_name != "test":
            raise ValueError("LabeledTestData exposes only the frozen test targets")
        return self.positive_targets_table


@dataclass(frozen=True)
class FrozenRankScorer:
    """Frozen EXP-011 coefficients sufficient for rank-based metrics."""

    variant: str
    intercept: float
    coefficients: np.ndarray
    design: Any
    coefficient_table: pd.DataFrame
    fit_report: pd.DataFrame
    coefficient_sha256: str
    rank_metrics_only: bool = True


def audit_test_template(intervals: pd.DataFrame) -> dict[str, Any]:
    """Prove that processed intervals retain official template order."""

    required = {
        "contig", "start", "end", "strand", "length_bp",
        "template_start", "template_stop", "source_split", "split",
    }
    missing = sorted(required - set(intervals.columns))
    if missing:
        raise ValueError("test intervals are missing columns: " + ", ".join(missing))
    if intervals.empty:
        raise ValueError("test template is empty")
    if set(intervals["split"].astype(str)) != {"test"}:
        raise ValueError("test intervals contain another local split")
    if set(intervals["source_split"].astype(str)) != {"test"}:
        raise ValueError("test intervals are not sourced from template.test")
    starts = intervals["template_start"].to_numpy(dtype=np.int64)
    stops = intervals["template_stop"].to_numpy(dtype=np.int64)
    lengths = intervals["length_bp"].to_numpy(dtype=np.int64)
    if starts[0] != 0 or not np.array_equal(stops - starts, lengths):
        raise ValueError("template offsets do not match interval lengths")
    if not np.array_equal(starts[1:], stops[:-1]):
        raise ValueError("template offsets contain a gap or overlap")
    strands = intervals["strand"].astype(str).to_numpy()
    minus_start = int(np.flatnonzero(strands == "-")[0])
    if np.any(strands[:minus_start] != "+") or np.any(strands[minus_start:] != "-"):
        raise ValueError("template must contain all plus intervals before minus")
    plus = intervals.iloc[:minus_start][["contig", "start", "end"]].reset_index(drop=True)
    minus = intervals.iloc[minus_start:][["contig", "start", "end"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(plus, minus)
    return {
        "template_interval_rows": int(len(intervals)),
        "whitelist_interval_rows": int(len(plus)),
        "test_contigs": int(intervals["contig"].nunique()),
        "plus_positions": int(lengths[:minus_start].sum()),
        "minus_positions": int(lengths[minus_start:].sum()),
        "test_positions": int(stops[-1]),
    }


def read_dense_test_ground_truth(
    path: Path | str,
    intervals: pd.DataFrame,
    *,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """Read dense counts in chunks and retain only nonzero test positions."""

    if not isinstance(chunksize, int) or chunksize <= 0:
        raise ValueError("chunksize must be a positive integer")
    contract = audit_test_template(intervals)
    expected = int(contract["test_positions"])
    starts = intervals["template_start"].to_numpy(dtype=np.int64)
    stops = intervals["template_stop"].to_numpy(dtype=np.int64)
    genomic_starts = intervals["start"].to_numpy(dtype=np.int64)
    contigs = intervals["contig"].astype(str).to_numpy()
    strands = intervals["strand"].astype(str).to_numpy()
    records: list[pd.DataFrame] = []
    cursor = 0
    reader = pd.read_csv(
        path,
        header=None,
        names=["count"],
        dtype={"count": "int64"},
        chunksize=chunksize,
        compression="infer",
    )
    for chunk in reader:
        values = chunk["count"].to_numpy(dtype=np.int64, copy=False)
        if np.any(values < 0):
            raise ValueError("test ground truth contains a negative count")
        local = np.flatnonzero(values > 0)
        if len(local):
            template_index = cursor + local.astype(np.int64)
            row = np.searchsorted(stops, template_index, side="right")
            if np.any(row >= len(intervals)):
                raise ValueError("test ground truth is longer than template.test")
            coordinate = genomic_starts[row] + template_index - starts[row]
            records.append(pd.DataFrame({
                "template_index": template_index,
                "contig": contigs[row],
                "coordinate_0based": coordinate,
                "strand": strands[row],
                "count": values[local],
                "target": np.ones(len(local), dtype=np.int8),
                "source_split": "test",
                "split": "test",
            }))
        cursor += len(values)
        if cursor > expected:
            raise ValueError("test ground truth is longer than template.test")
    if cursor != expected:
        raise ValueError(
            f"test ground truth length {cursor:,} != template positions {expected:,}"
        )
    if not records:
        raise ValueError("test ground truth contains no positive positions")
    positives = pd.concat(records, ignore_index=True)
    if not positives["template_index"].is_unique:
        raise AssertionError("test template indices are not unique")
    return positives.sort_values("template_index").reset_index(drop=True)


def load_labeled_test(
    project_root: Path | str,
    split: PreparedSplit,
    *,
    chunksize: int = 1_000_000,
) -> LabeledTestData:
    """Open the frozen demo ground truth and return a test-only data contract."""

    root = Path(project_root).resolve()
    if split.config.assembly != "jaNemVect1.1":
        raise ValueError("open labelled test exists only for jaNemVect1.1")
    intervals = split.intervals("test").copy()
    contract = audit_test_template(intervals)
    if int(contract["test_positions"]) != EXPECTED_TEST_POSITIONS:
        raise AssertionError("unexpected jaNemVect1.1 test population")
    path = (
        root / "data/raw/unpacked/jaNemVect1.1/ground_truth/csRNA-test.txt.gz"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    positives = read_dense_test_ground_truth(path, intervals, chunksize=chunksize)
    diagnostics = pd.DataFrame([{
        **contract,
        "positive_positions": int(len(positives)),
        "negative_positions": int(contract["test_positions"] - len(positives)),
        "prevalence": len(positives) / int(contract["test_positions"]),
        "total_csrna_count": int(positives["count"].sum()),
        "ground_truth_file": path.relative_to(root).as_posix(),
        "ground_truth_sha256": sha256_file(path),
    }])
    return LabeledTestData(
        intervals, positives, diagnostics, path, str(diagnostics.iloc[0]["ground_truth_sha256"])
    )


def load_frozen_exp011_rank_scorer(
    artifact_dir: Path | str,
) -> FrozenRankScorer:
    """Rebuild the exact EXP-011 ranking function from immutable CSV artifacts.

    The historical run did not persist the intercept.  AP and Spearman are
    invariant to a single additive constant, so zero is used deliberately.
    This object must not be used for probabilities or log-loss.
    """

    from ml_project.epic_experiments.exp_011_gg_ta_complementarity import (
        GGTAConfig,
        _streaming_coefficient_table,
    )
    from ml_project.epic_kmer_logistic import hierarchical_kmer_design

    directory = Path(artifact_dir).resolve()
    metadata_path = directory / "metadata.json"
    coefficient_path = directory / "coefficient_table.csv"
    report_path = directory / "fit_report.csv"
    for path in (metadata_path, coefficient_path, report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["experiment"]["experiment_id"] != "EXP-011":
        raise ValueError("artifact is not EXP-011")
    if metadata["run"]["decision"] != "adopt":
        raise ValueError("EXP-011 artifact was not adopted")
    variant = str(metadata["parameters"]["selected_variant"])
    if variant != "EXP-009+GG+TA":
        raise ValueError("unexpected frozen EXP-011 variant")
    if float(metadata["parameters"]["selected_C"]) != 1e-4:
        raise ValueError("unexpected frozen EXP-011 C")

    coefficient_table = pd.read_csv(coefficient_path).sort_values(
        "feature_index"
    ).reset_index(drop=True)
    indices = coefficient_table["feature_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(indices, np.arange(4400, dtype=np.int64)):
        raise ValueError("EXP-011 coefficient indices must be exactly 0..4399")
    coefficients = coefficient_table["coefficient"].to_numpy(dtype=np.float64)
    if not np.isfinite(coefficients).all():
        raise ValueError("EXP-011 contains a non-finite coefficient")
    config = GGTAConfig()
    design = hierarchical_kmer_design(config)
    expected = _streaming_coefficient_table(
        np.zeros(4400, dtype=np.float64), design, variant, config
    )
    identity = [
        "feature_index", "category_code", "category", "feature_name", "feature_group"
    ]
    pd.testing.assert_frame_equal(
        coefficient_table[identity], expected[identity],
        check_dtype=False, check_names=True,
    )
    fit_report = pd.read_csv(report_path)
    if len(fit_report) != 1 or not bool(fit_report.iloc[0]["converged"]):
        raise ValueError("frozen EXP-011 fit did not converge")
    if int(fit_report.iloc[0]["features"]) != 4400:
        raise ValueError("frozen EXP-011 fit has an unexpected width")
    return FrozenRankScorer(
        variant=variant,
        intercept=0.0,
        coefficients=coefficients,
        design=design,
        coefficient_table=coefficient_table,
        fit_report=fit_report,
        coefficient_sha256=sha256_file(coefficient_path),
    )


def _merge_touching_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[tuple[str, int, int]] = []
    ordered = frame.sort_values(["contig", "start", "end"])
    for contig, group in ordered.groupby("contig", sort=False):
        current_start: int | None = None
        current_end: int | None = None
        for start, end in group[["start", "end"]].to_numpy(dtype=np.int64):
            start, end = int(start), int(end)
            if current_start is None:
                current_start, current_end = start, end
            elif start <= int(current_end):
                current_end = max(int(current_end), end)
            else:
                records.append((str(contig), int(current_start), int(current_end)))
                current_start, current_end = start, end
        if current_start is not None:
            records.append((str(contig), int(current_start), int(current_end)))
    return pd.DataFrame(records, columns=["contig", "start", "end"])


def _positive_rows_in_intervals(
    positives: pd.DataFrame, intervals: pd.DataFrame
) -> pd.DataFrame:
    keep = np.zeros(len(positives), dtype=bool)
    for contig, group in positives.groupby("contig", sort=False):
        allowed = intervals.loc[intervals["contig"].astype(str) == str(contig)]
        if allowed.empty:
            continue
        starts = allowed["start"].to_numpy(dtype=np.int64)
        ends = allowed["end"].to_numpy(dtype=np.int64)
        coordinates = group["coordinate_0based"].to_numpy(dtype=np.int64)
        locations = np.searchsorted(starts, coordinates, side="right") - 1
        inside = locations >= 0
        inside_indices = np.flatnonzero(inside)
        inside[inside_indices] = coordinates[inside_indices] < ends[locations[inside_indices]]
        keep[group.index.to_numpy(dtype=np.int64)] = inside
    return positives.loc[keep].copy().reset_index(drop=True)


def load_official_test_halves(
    project_root: Path | str,
    labeled_test: LabeledTestData,
) -> dict[str, LabeledTestData]:
    """Build the exact half1/half2 populations used by EPIC scoring."""

    root = Path(project_root).resolve()
    scoring = root / "data/raw/unpacked/jaNemVect1.1/scoring"
    bed_frames: dict[str, pd.DataFrame] = {}
    for name in ("half1", "half2"):
        path = scoring / f"{name}-test.bed.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(
            path, sep="\t", header=None, names=["contig", "start", "end"],
            dtype={"contig": str, "start": "int64", "end": "int64"},
        )
        if frame.empty or (frame["end"] <= frame["start"]).any():
            raise ValueError(f"invalid {name} BED")
        if frame.duplicated().any():
            raise ValueError(f"duplicate interval in {name} BED")
        bed_frames[name] = frame

    combined = pd.concat(bed_frames.values(), ignore_index=True)
    combined_length = int((combined["end"] - combined["start"]).sum())
    full_plus = labeled_test.intervals_table.loc[
        labeled_test.intervals_table["strand"] == "+", ["contig", "start", "end"]
    ].reset_index(drop=True)
    if combined_length != int((full_plus["end"] - full_plus["start"]).sum()):
        raise ValueError("scoring halves do not preserve whitelist length")
    pd.testing.assert_frame_equal(
        _merge_touching_intervals(combined).reset_index(drop=True),
        _merge_touching_intervals(full_plus).reset_index(drop=True),
    )

    result: dict[str, LabeledTestData] = {}
    assigned_positive_indices: list[np.ndarray] = []
    for name, bed in bed_frames.items():
        intervals = pd.concat(
            [bed.assign(strand="+"), bed.assign(strand="-")], ignore_index=True
        )
        intervals["length_bp"] = intervals["end"] - intervals["start"]
        intervals["source_split"] = "test"
        intervals["split"] = "test"
        positives = _positive_rows_in_intervals(
            labeled_test.positive_targets_table, bed
        )
        assigned_positive_indices.append(
            positives["template_index"].to_numpy(dtype=np.int64)
        )
        positions = int(intervals["length_bp"].sum())
        diagnostics = pd.DataFrame([{
            "scoring_half": name,
            "interval_rows_both_strands": len(intervals),
            "test_contigs": intervals["contig"].nunique(),
            "positions": positions,
            "positive_positions": len(positives),
            "negative_positions": positions - len(positives),
            "prevalence": len(positives) / positions,
            "total_csrna_count": int(positives["count"].sum()),
            "ground_truth_sha256": labeled_test.ground_truth_sha256,
        }])
        result[name] = LabeledTestData(
            intervals, positives, diagnostics,
            labeled_test.ground_truth_path, labeled_test.ground_truth_sha256,
        )

    assigned = np.sort(np.concatenate(assigned_positive_indices))
    expected = labeled_test.positive_targets_table["template_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(assigned, expected):
        raise AssertionError("half1/half2 do not partition all positive test positions")
    if sum(int(item.diagnostics.iloc[0]["positions"]) for item in result.values()) != EXPECTED_TEST_POSITIONS:
        raise AssertionError("half1/half2 do not partition all test positions")
    return result


__all__ = [
    "EXPECTED_TEST_POSITIONS", "FrozenRankScorer", "LabeledTestData",
    "audit_test_template", "load_frozen_exp011_rank_scorer",
    "load_labeled_test", "load_official_test_halves",
    "read_dense_test_ground_truth",
]
