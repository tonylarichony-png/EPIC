"""CPU-only resolution audit of the existing EXP046/EXP045 scalar cache.

This is a validation-label oracle, never a deployable predictor. Smaller bins
reveal more target-location information; larger oracle AP is not evidence that
the corresponding regional prediction problem can be learned from sequence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Keep this inexpensive diagnostic single-threaded, even if BLAS is imported.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from ml_project.epic_cnn.profile import ProfileWindowConfig, build_profile_tile_index
from ml_project.epic_data import load_split, sha256_file


REGIONS = (1, 2, 4, 8, 16, 32, 64, 128)
LEVELS = 100_001
EXPECTED_POSITIONS = 83_807_486
EXPECTED_POSITIVES = 56_205
EXPECTED_TILES = 47_743
AP_TOL = 1e-12
EXPERIMENT = ROOT / "artifacts/experiments/CNN_EXP046_REGIONAL_ACTIVITY_GATE/run_001"


def ap_from_histograms(positive: np.ndarray, total: np.ndarray) -> float:
    pos = positive[::-1]
    tp = np.cumsum(pos, dtype=np.int64)
    predicted = np.cumsum(total[::-1], dtype=np.int64)
    return float(np.sum((pos / int(pos.sum())) * tp / np.maximum(predicted, 1)))


def interval_membership(coordinates: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    preceding = np.searchsorted(starts, coordinates, side="right") - 1
    return (preceding >= 0) & (coordinates < ends[np.maximum(preceding, 0)])


def positive_membership(coordinates: np.ndarray, positives: np.ndarray) -> np.ndarray:
    if not len(positives):
        return np.zeros(coordinates.shape, dtype=bool)
    location = np.searchsorted(positives, coordinates)
    return (location < len(positives)) & (positives[np.minimum(location, len(positives) - 1)] == coordinates)


def make_accumulator() -> dict:
    return {
        "positive_hist": np.zeros(LEVELS, dtype=np.int64),
        "baseline_hist": np.zeros(LEVELS, dtype=np.int64),
        "oracle_hist": {r: np.zeros(LEVELS, dtype=np.int64) for r in REGIONS},
        "valid_regions": {r: 0 for r in REGIONS},
        "active_regions": {r: 0 for r in REGIONS},
        "retained_negatives": {r: 0 for r in REGIONS},
    }


def summarize(accumulator: dict) -> dict:
    positive_hist = accumulator["positive_hist"]
    positives = int(positive_hist.sum())
    positions = int(accumulator["baseline_hist"].sum())
    negatives = positions - positives
    baseline = ap_from_histograms(positive_hist, accumulator["baseline_hist"])
    return {
        "positions": positions,
        "positives": positives,
        "baseline_ap5": baseline,
        "positive_positions_with_zero_score_bin": int(positive_hist[0]),
        "oracle_by_region_size": {
            str(r): {
                "oracle_ap5": ap_from_histograms(positive_hist, accumulator["oracle_hist"][r]),
                "delta_ap5": ap_from_histograms(positive_hist, accumulator["oracle_hist"][r]) - baseline,
                "valid_regions": accumulator["valid_regions"][r],
                "active_regions": accumulator["active_regions"][r],
                "active_region_prevalence": accumulator["active_regions"][r] / accumulator["valid_regions"][r],
                "retained_negative_positions": accumulator["retained_negatives"][r],
                "retained_negative_fraction": accumulator["retained_negatives"][r] / negatives,
                "suppressed_negative_fraction": 1 - accumulator["retained_negatives"][r] / negatives,
                "positive_position_retention": 1.0,
            }
            for r in REGIONS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-tiles", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT / "oracle_resolution_audit")
    args = parser.parse_args()
    assert args.chunk_tiles > 0
    started = time.perf_counter()
    split = load_split(ROOT)
    tiles = build_profile_tile_index(
        split, "validation", ProfileWindowConfig(target_length=1024, context_flank=130, positive_branch_fraction=0.5)
    )
    assert len(tiles) == EXPECTED_TILES
    cache = EXPERIMENT / "feasibility_frontier_r32/_scratch_region_cache"
    scores_path = cache / "score_bins_u32.dat"
    masks_path = cache / "valid_mask_u32.dat"
    expected_rows = len(tiles) * 2 * 32
    assert scores_path.stat().st_size == expected_rows * 32 * 4
    assert masks_path.stat().st_size == expected_rows * 4
    scores = np.memmap(scores_path, mode="r", dtype=np.uint32, shape=(len(tiles), 2, 1024))
    masks = np.memmap(masks_path, mode="r", dtype=np.uint32, shape=(len(tiles), 2, 32))
    intervals = split.intervals("validation")
    target = split.positive_targets("validation")
    assert target is not None and len(target) == EXPECTED_POSITIVES
    assert (target["count"] > 0).all()
    assert not target.duplicated(["contig", "strand", "coordinate_0based"]).any()
    intervals_lookup = {}
    positives_lookup = {}
    for contig in tiles["contig"].unique():
        for strand in ("+", "-"):
            frame = intervals.loc[(intervals["contig"] == contig) & (intervals["strand"] == strand)].sort_values("start")
            starts, ends = frame["start"].to_numpy(dtype=np.int64), frame["end"].to_numpy(dtype=np.int64)
            assert len(starts) and np.all(ends > starts) and np.all(starts[1:] >= ends[:-1])
            intervals_lookup[(contig, strand)] = (starts, ends)
            positives_lookup[(contig, strand)] = np.sort(target.loc[
                (target["contig"] == contig) & (target["strand"] == strand), "coordinate_0based"
            ].to_numpy(dtype=np.int64))

    all_accumulators = {}
    bit_weights = np.left_shift(np.uint32(1), np.arange(32, dtype=np.uint32))
    checked_rows = 0
    for contig, contig_tiles in tiles.groupby("contig", sort=False):
        acc = make_accumulator()
        for chunk_start in range(0, len(contig_tiles), args.chunk_tiles):
            selected = contig_tiles.iloc[chunk_start:chunk_start + args.chunk_tiles]
            ids = selected["tile_id"].to_numpy(dtype=np.int64)
            assert np.array_equal(ids, np.arange(ids[0], ids[-1] + 1))
            coordinates = selected["target_start"].to_numpy(dtype=np.int64)[:, None] + np.arange(1024, dtype=np.int64)
            valid = np.empty((len(selected), 2, 1024), dtype=bool)
            positive = np.empty_like(valid)
            for strand_id, strand in enumerate(("+", "-")):
                valid[:, strand_id, :] = interval_membership(coordinates, *intervals_lookup[(contig, strand)])
                positive[:, strand_id, :] = positive_membership(coordinates, positives_lookup[(contig, strand)])
            assert np.all(~positive | valid), "Positive outside independently reconstructed whitelist"
            reconstructed_masks = (valid.reshape(-1, 32).astype(np.uint32) * bit_weights).sum(axis=1, dtype=np.uint32)
            cached_masks = np.asarray(masks[ids[0]:ids[-1] + 1]).reshape(-1)
            assert np.array_equal(reconstructed_masks, cached_masks), f"Whitelist/cache row mismatch: {contig}, tile {ids[0]}"
            checked_rows += len(reconstructed_masks)
            score = np.asarray(scores[ids[0]:ids[-1] + 1])
            assert int(score.max()) < LEVELS
            acc["positive_hist"] += np.bincount(score[positive], minlength=LEVELS)
            acc["baseline_hist"] += np.bincount(score[valid], minlength=LEVELS)
            positives_in_chunk = int(positive.sum())
            positions_in_chunk = int(valid.sum())
            for r in REGIONS:
                positive_regions = positive.reshape(-1, r).any(axis=1)
                valid_regions = valid.reshape(-1, r).any(axis=1)
                keep = (valid.reshape(-1, r) & positive_regions[:, None]).reshape(valid.shape)
                retained = int(keep.sum())
                hist = np.bincount(score[keep], minlength=LEVELS)
                hist[0] += positions_in_chunk - retained
                acc["oracle_hist"][r] += hist
                acc["valid_regions"][r] += int(valid_regions.sum())
                acc["active_regions"][r] += int(positive_regions.sum())
                acc["retained_negatives"][r] += retained - positives_in_chunk
        all_accumulators[str(contig)] = acc
        print(f"Verified {contig}: {int(acc['baseline_hist'].sum()):,} positions; elapsed {time.perf_counter() - started:.1f}s", flush=True)

    assert checked_rows == expected_rows
    overall = make_accumulator()
    for acc in all_accumulators.values():
        for name in ("positive_hist", "baseline_hist"):
            overall[name] += acc[name]
        for name in ("oracle_hist", "valid_regions", "active_regions", "retained_negatives"):
            for r in REGIONS:
                overall[name][r] += acc[name][r]
    summaries = {"overall": summarize(overall), **{key: summarize(value) for key, value in all_accumulators.items()}}
    assert summaries["overall"]["positions"] == EXPECTED_POSITIONS
    assert summaries["overall"]["positives"] == EXPECTED_POSITIVES

    # Require every existing regional/contig metric to reproduce, not only one AP.
    reference_path = EXPERIMENT / "oracle_regional_gate_results.csv"
    references = pd.read_csv(reference_path)
    for row in references.itertuples(index=False):
        segment = summaries[row.segment]
        oracle = segment["oracle_by_region_size"][str(row.region_size)]
        assert abs(segment["baseline_ap5"] - row.baseline_ap1) <= AP_TOL
        assert abs(oracle["oracle_ap5"] - row.oracle_ap1) <= AP_TOL, (row.segment, row.region_size, oracle["oracle_ap5"], row.oracle_ap1)
        assert abs(oracle["retained_negative_fraction"] - row.negative_active_fraction) <= AP_TOL

    # Independently reuse the *fixed*, previously published 20k hard-FP cohort.
    hard_fp_path = EXPERIMENT / "fixed_hard_fp_coordinates.csv"
    hard_fp = pd.read_csv(hard_fp_path)
    assert len(hard_fp) == 20_000
    for r in REGIONS:
        active = np.zeros(len(hard_fp), dtype=bool)
        for (contig, strand), frame in hard_fp.groupby(["contig", "strand"], sort=False):
            active_ids = np.unique(positives_lookup[(contig, strand)] // r)
            active[frame.index] = positive_membership(frame["position"].to_numpy(dtype=np.int64) // r, active_ids)
        if f"active_r{r}" in hard_fp:
            assert np.array_equal(active, hard_fp[f"active_r{r}"].to_numpy(dtype=bool))
        summaries["overall"]["oracle_by_region_size"][str(r)]["fixed_hard_fp_empty_fraction"] = float((~active).mean())

    # R1 transmits the binary label itself. Only score-bin-zero positive ties can
    # prevent the suppression oracle from giving AP=1.
    zero_positive = int(overall["positive_hist"][0])
    expected_r1 = 1 - zero_positive / EXPECTED_POSITIVES + zero_positive / EXPECTED_POSITIONS
    assert abs(summaries["overall"]["oracle_by_region_size"]["1"]["oracle_ap5"] - expected_r1) <= AP_TOL
    ordered_ap = [summaries["overall"]["oracle_by_region_size"][str(r)]["oracle_ap5"] for r in REGIONS]
    assert np.all(np.diff(ordered_ap) <= AP_TOL), "Nested perfect suppression must be monotone with resolution"

    provenance_paths = [
        Path(__file__), scores_path, masks_path, reference_path, hard_fp_path,
        split.paths.output / "manifest.json", split.paths.output / "validation_intervals.csv.gz",
        split.paths.output / "validation_positive_targets.csv.gz",
        ROOT / "notebooks/experiments/CNN_experiments/CNN_EXP046_R32_FEASIBILITY_FRONTIER.ipynb",
    ]
    output = {
        "experiment_id": "CNN-EXP-046-ORACLE-RESOLUTION-AUDIT",
        "status": "all_cache_and_reference_checks_passed",
        "kind": "validation-label diagnostic, not a model result or architecture selection criterion",
        "score_source": "existing EXP045 width512_50k_final predictions cached by EXP046",
        "score_quantization": "uint32 round(probability * 100000), 100001 levels; historical AP1 fields reproduce AP on these 5dp bins",
        "oracle_rule": "for each strand-specific floor(genomic_coordinate/R) bin with no positive validation target, set every valid score to zero; preserve all other scores",
        "grid_origin": 0,
        "row_layout": "sorted validation tile; plus strand 1024 genomic coordinates followed by minus strand 1024 genomic coordinates",
        "new_inference_or_training": False,
        "validation_labels_used_for_training": False,
        "interpretation": [
            "R is bin width in nucleotides, not plus/minus radius.",
            "Nested R1/R2/R4/R8/R16/... reveals increasingly precise true target location; increased oracle AP at smaller R is mechanically expected.",
            "R1 directly reveals presence labels. AP may be below one only due to positives rounded to score bin zero tying with suppressed negatives.",
            "This diagnostic measures removal of negatives under perfect target knowledge, not trainability, learned gate recall/precision, or architecture capacity.",
            "It is not an upper bound on a learned model that can change ranking within/between active regions.",
            "Validation-label oracle outputs must not be used as model inputs, distillation targets, calibration targets, or selection evidence favoring a smaller R.",
            "These are EXP045 cache results, not diagnostics of the EXP049 joint checkpoint.",
        ],
        "checks": {"all_valid_masks_match_split": True, "verified_r32_cache_rows": checked_rows, "old_reference_rows_reproduced": len(references), "reference_ap_tolerance": AP_TOL, "raw_and_derived_split_hashes_verified_by_load_split": True},
        "provenance_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in provenance_paths},
        "segments": summaries,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "oracle_resolution_audit.json"
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# EXP046 oracle resolution audit", "",
        "CPU-only reuse of EXP045 cached scores. All original whitelist masks, AP values and R16/R32/R64/R128 results reproduced before accepting new results.", "",
        f"Validation: {EXPECTED_POSITIONS:,} position-strands, {EXPECTED_POSITIVES:,} positives. Baseline AP5: {summaries['overall']['baseline_ap5']:.12f}.", "",
        "| Bin width R | Oracle AP5 | Active bins | Retained negatives | Hard-FP cohort suppressed |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in REGIONS:
        item = summaries["overall"]["oracle_by_region_size"][str(r)]
        lines.append(f"| {r} | {item['oracle_ap5']:.9f} | {item['active_regions']:,} | {item['retained_negative_positions']:,} ({100 * item['retained_negative_fraction']:.4f}%) | {100 * item['fixed_hard_fp_empty_fraction']:.3f}% |")
    lines.extend(["", *[f"- {item}" for item in output["interpretation"]], "", f"Positive positions rounded to zero: {zero_positive}; R1 AP expected from tie handling: {expected_r1:.12f}.", "", "Full per-contig results, exact assertions and provenance SHA-256 hashes are in `oracle_resolution_audit.json`.", ""])
    (args.output_dir / "oracle_resolution_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Saved {json_path}", flush=True)


if __name__ == "__main__":
    main()
