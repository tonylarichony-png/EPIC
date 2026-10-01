from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd

KEY_SHIFT = np.uint64(56)
REGION_MASK = (np.uint64(1) << KEY_SHIFT) - np.uint64(1)
BIT_SHIFTS = np.arange(32, dtype=np.uint32)
SCORE_LEVELS = 100_001

NEGATIVE_BAND_EDGES = np.asarray([0.001, 0.002, 0.005, 0.01, 0.05, 0.15], dtype=np.float64)
NEGATIVE_BAND_NAMES = [
    "0-0.1%",
    "0.1-0.2%",
    "0.2-0.5%",
    "0.5-1%",
    "1-5%",
    "5-15%",
    "15-100%",
]
NEGATIVE_SAMPLE_PER_BAND = 1_000
POSITIVE_SAMPLE_PER_QUARTILE = 1_000
RNG_SEED = np.uint64(480048)
GPU_DUTY_CYCLE = float(os.environ.get("EPIC_GPU_DUTY_CYCLE", "0.90"))
if not (0.50 <= GPU_DUTY_CYCLE <= 1.0):
    raise RuntimeError(f"EPIC_GPU_DUTY_CYCLE must be in [0.50, 1.0], got {GPU_DUTY_CYCLE}")


def project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "README.md").is_file() and (candidate / "src/ml_project").is_dir():
            return candidate
    raise RuntimeError("Project root was not found")


ROOT = project_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml_project.epic_baseline import load_baseline_sequences
from ml_project.epic_cnn import (
    ProfileWindowConfig,
    ProfileWindowGenerator,
    create_cnn_presence_intensity,
    forward_both_strands_multitask,
)
from ml_project.epic_data import load_split

BASE_RUN = (
    ROOT / "artifacts" / "experiments"
    / "CNN_EXP045_WIDTH512_50K_TWO_PHASE_LR_RF1029" / "run_001"
)
BASE_CHECKPOINT = BASE_RUN / "model_step_50000.pt"

EXP046_RUN = (
    ROOT / "artifacts" / "experiments"
    / "CNN_EXP046B_FROZEN_REGIONAL_CORRECTION_R32" / "run_001"
)
CV_SHARDS = EXP046_RUN / "_cv_eval_v3"

EXP047_RUN = (
    ROOT / "artifacts" / "experiments"
    / "CNN_EXP047_WIDTH512_PAIRED_RISK_LOSS_CONTINUATION" / "run_001"
)
EXP047_CHECKPOINT = EXP047_RUN / "branch_B" / "branch_B_final.pt"
EXP047_COHORT = EXP047_RUN / "branch_B" / "fixed_cohort_scores.csv"

EXP048_RUN = (
    ROOT / "artifacts" / "experiments"
    / "CNN_EXP048_WIDTH512_ULTRA_TAIL_RISK_REFINEMENT" / "run_001"
)
EXP048_CHECKPOINT = EXP048_RUN / "branch_B" / "branch_B_final.pt"
EXP048_COHORT = EXP048_RUN / "branch_B" / "fixed_cohort_scores.csv"

RUN_DIR = (
    ROOT / "artifacts" / "experiments"
    / "CNN_EXP048_TAIL_DIAGNOSTIC" / "run_001"
)
SAMPLE_DIR = RUN_DIR / "sample"
SAMPLE_MANIFEST = SAMPLE_DIR / "train_tail_sample.csv"
SAMPLE_META = SAMPLE_DIR / "train_tail_sample_meta.json"
SCORES_047 = RUN_DIR / "sample_scores_EXP047B.csv"
SCORES_048 = RUN_DIR / "sample_scores_EXP048.csv"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    with open(tmp, "r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sigmoid_numpy(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    out = np.empty_like(logits)
    pos = logits >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-logits[pos]))
    ex = np.exp(logits[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def score_bins(logits: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(sigmoid_numpy(logits), 0.0, 1.0) * 100_000).astype(np.int32)


def unpack_mask(mask_u32: np.ndarray) -> np.ndarray:
    values = np.asarray(mask_u32, dtype=np.uint32)
    return ((values[:, None] >> BIT_SHIFTS[None, :]) & np.uint32(1)).astype(bool)


def iter_shards() -> list[Path]:
    files: list[Path] = []
    for fold in sorted(path for path in CV_SHARDS.glob("fold_*") if path.is_dir()):
        files.extend(sorted((fold / "shards").glob("tiles_*.npz")))
    if not files:
        raise FileNotFoundError(f"No corrected train shards under {CV_SHARDS}")
    return files


def splitmix64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint64)
    x = x + np.uint64(0x9E3779B97F4A7C15)
    z = x.copy()
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def coordinate_hash(region_key: np.ndarray, offset: np.ndarray, salt: int) -> np.ndarray:
    key = np.asarray(region_key, dtype=np.uint64)
    off = np.asarray(offset, dtype=np.uint64)
    mixed = key ^ (off * np.uint64(0xD6E8FEB86659FD93)) ^ np.uint64(salt)
    return splitmix64(mixed)


def _trim_smallest(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    if len(frame) <= k:
        return frame
    values = frame["_hash"].to_numpy(np.uint64)
    indices = np.argpartition(values, k - 1)[:k]
    return frame.iloc[indices].copy()


def _decode_coordinates(keys: np.ndarray, offsets: np.ndarray, all_contigs: np.ndarray):
    keys = np.asarray(keys, dtype=np.uint64)
    offsets = np.asarray(offsets, dtype=np.int64)
    prefix = keys >> KEY_SHIFT
    contig_codes = (prefix // np.uint64(2)).astype(np.int64)
    strand_codes = (prefix % np.uint64(2)).astype(np.int64)
    region_ids = (keys & REGION_MASK).astype(np.int64)
    contigs = all_contigs[contig_codes]
    strands = np.where(strand_codes == 0, "+", "-")
    positions = region_ids * 32 + offsets
    return contigs, strands, positions


def build_sample() -> None:
    required = [BASE_CHECKPOINT, EXP047_CHECKPOINT, EXP048_CHECKPOINT, CV_SHARDS]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)

    if SAMPLE_MANIFEST.exists() and SAMPLE_META.exists():
        meta = json.loads(SAMPLE_META.read_text(encoding="utf-8"))
        if meta.get("complete"):
            print(f"Sample already complete: {SAMPLE_MANIFEST}")
            print(json.dumps(meta, indent=2))
            return

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    split = load_split(ROOT)
    train_contigs = sorted(set(split.intervals("train")["contig"].astype(str)))
    validation_contigs = sorted(set(split.intervals("validation")["contig"].astype(str)))
    all_contigs = np.asarray(train_contigs + validation_contigs, dtype=object)

    files = iter_shards()
    histogram = np.zeros(SCORE_LEVELS, dtype=np.int64)
    positive_keys_parts = []
    positive_offsets_parts = []
    positive_logits_parts = []

    started = time.perf_counter()
    for i, path in enumerate(files, start=1):
        with np.load(path, allow_pickle=False) as data:
            valid = unpack_mask(data["valid_mask"])
            positive = unpack_mask(data["positive_mask"])
            negative = valid & ~positive
            logits = data["base_logit"]
            bins = score_bins(logits)
            histogram += np.bincount(bins[negative], minlength=SCORE_LEVELS)

            pos_rows, pos_offsets = np.nonzero(positive)
            if len(pos_rows):
                positive_keys_parts.append(
                    data["region_key"][pos_rows].astype(np.uint64, copy=False)
                )
                positive_offsets_parts.append(pos_offsets.astype(np.uint8, copy=False))
                positive_logits_parts.append(
                    logits[pos_rows, pos_offsets].astype(np.float32, copy=False)
                )
        if i == 1 or i % 500 == 0 or i == len(files):
            print(f"histogram + positives: {i:,}/{len(files):,}", flush=True)

    total_negative = int(histogram.sum())
    if total_negative <= 0:
        raise AssertionError("No train negatives found")

    cut_counts = np.asarray(
        [round(total_negative * x) for x in NEGATIVE_BAND_EDGES], dtype=np.int64
    )
    population_counts = np.diff(
        np.r_[np.int64(0), cut_counts, np.int64(total_negative)]
    )
    population_fractions = population_counts.astype(np.float64) / total_negative

    higher = np.cumsum(histogram[::-1], dtype=np.int64)[::-1] - histogram
    seen_by_score = np.zeros(SCORE_LEVELS, dtype=np.int64)

    reservoirs: dict[int, pd.DataFrame] = {
        i: pd.DataFrame() for i in range(len(NEGATIVE_BAND_NAMES))
    }
    realized = np.zeros(len(NEGATIVE_BAND_NAMES), dtype=np.int64)

    for file_index, path in enumerate(files, start=1):
        with np.load(path, allow_pickle=False) as data:
            valid = unpack_mask(data["valid_mask"])
            positive = unpack_mask(data["positive_mask"])
            negative = valid & ~positive
            logits = data["base_logit"]
            bins = score_bins(logits)

            neg_rows, neg_offsets = np.nonzero(negative)
            negative_scores = bins[neg_rows, neg_offsets]
            if len(negative_scores) == 0:
                continue

            order = np.argsort(negative_scores, kind="stable")
            ordered_scores = negative_scores[order]
            group_starts = np.r_[
                np.int64(0),
                np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]).astype(np.int64) + 1,
            ]
            group_ends = np.r_[group_starts[1:], np.int64(len(ordered_scores))]
            group_lengths = group_ends - group_starts
            unique_scores = ordered_scores[group_starts]
            offsets_in_score_group = np.arange(len(ordered_scores), dtype=np.int64) - np.repeat(
                group_starts, group_lengths
            )
            rank_bases = np.repeat(
                higher[unique_scores] + seen_by_score[unique_scores], group_lengths
            )
            ordered_ranks = rank_bases + offsets_in_score_group
            ordered_band = np.searchsorted(
                cut_counts, ordered_ranks, side="right"
            ).astype(np.uint8)

            band = np.empty(len(ordered_band), dtype=np.uint8)
            band[order] = ordered_band
            seen_by_score[unique_scores] += group_lengths
            realized += np.bincount(ordered_band, minlength=len(NEGATIVE_BAND_NAMES))

            keys = data["region_key"][neg_rows].astype(np.uint64, copy=False)
            base_logits = logits[neg_rows, neg_offsets].astype(np.float32, copy=False)

            for b in range(len(NEGATIVE_BAND_NAMES)):
                where = np.flatnonzero(band == b)
                if len(where) == 0:
                    continue
                # Keep at most K candidates from this shard before merging.
                hashes = coordinate_hash(
                    keys[where],
                    neg_offsets[where],
                    int(RNG_SEED + np.uint64(b) * np.uint64(10007)),
                )
                if len(where) > NEGATIVE_SAMPLE_PER_BAND:
                    local = np.argpartition(hashes, NEGATIVE_SAMPLE_PER_BAND - 1)[
                        :NEGATIVE_SAMPLE_PER_BAND
                    ]
                    where = where[local]
                    hashes = hashes[local]

                contigs, strands, positions = _decode_coordinates(
                    keys[where], neg_offsets[where], all_contigs
                )
                chunk = pd.DataFrame(
                    {
                        "kind": "negative",
                        "group": NEGATIVE_BAND_NAMES[b],
                        "contig": contigs,
                        "strand": strands,
                        "position": positions.astype(np.int64),
                        "base_logit": base_logits[where].astype(np.float64),
                        "_hash": hashes.astype(np.uint64),
                    }
                )
                merged = pd.concat([reservoirs[b], chunk], ignore_index=True)
                reservoirs[b] = _trim_smallest(merged, NEGATIVE_SAMPLE_PER_BAND)

        if file_index == 1 or file_index % 500 == 0 or file_index == len(files):
            print(f"negative rank sampling: {file_index:,}/{len(files):,}", flush=True)

    if not np.array_equal(realized, population_counts):
        raise AssertionError(
            f"Band population mismatch: realized={realized.tolist()} expected={population_counts.tolist()}"
        )

    negative_sample = pd.concat(
        [reservoirs[i] for i in range(len(NEGATIVE_BAND_NAMES))],
        ignore_index=True,
    )
    counts = negative_sample.groupby("group").size()
    for band_name, pop_count in zip(NEGATIVE_BAND_NAMES, population_counts):
        expected = min(NEGATIVE_SAMPLE_PER_BAND, int(pop_count))
        actual = int(counts.get(band_name, 0))
        if actual != expected:
            raise AssertionError((band_name, actual, expected))

    # Build a natural positive control, stratified by EXP045-score quartile.
    pos_keys = np.concatenate(positive_keys_parts).astype(np.uint64, copy=False)
    pos_offsets = np.concatenate(positive_offsets_parts).astype(np.uint8, copy=False)
    pos_logits = np.concatenate(positive_logits_parts).astype(np.float32, copy=False)
    pos_scores = sigmoid_numpy(pos_logits)
    quartile_edges = np.quantile(pos_scores, [0.25, 0.50, 0.75])
    pos_quartile = np.searchsorted(quartile_edges, pos_scores, side="right")
    positive_frames = []

    for q in range(4):
        where = np.flatnonzero(pos_quartile == q)
        hashes = coordinate_hash(
            pos_keys[where],
            pos_offsets[where],
            int(RNG_SEED + np.uint64(1_000_000 + q * 10007)),
        )
        k = min(POSITIVE_SAMPLE_PER_QUARTILE, len(where))
        if len(where) > k:
            selected_local = np.argpartition(hashes, k - 1)[:k]
            where = where[selected_local]
            hashes = hashes[selected_local]
        contigs, strands, positions = _decode_coordinates(
            pos_keys[where], pos_offsets[where], all_contigs
        )
        positive_frames.append(
            pd.DataFrame(
                {
                    "kind": "positive",
                    "group": f"Q{q+1}",
                    "contig": contigs,
                    "strand": strands,
                    "position": positions.astype(np.int64),
                    "base_logit": pos_logits[where].astype(np.float64),
                    "_hash": hashes.astype(np.uint64),
                }
            )
        )

    positive_sample = pd.concat(positive_frames, ignore_index=True)
    sample = pd.concat([negative_sample, positive_sample], ignore_index=True)
    sample["score_EXP045"] = sigmoid_numpy(sample["base_logit"].to_numpy(np.float64))
    sample["target_start"] = sample["position"].to_numpy(np.int64) // 1024 * 1024
    sample = sample.drop(columns=["_hash"]).sort_values(
        ["kind", "group", "contig", "strand", "position"], kind="stable"
    ).reset_index(drop=True)
    sample.insert(0, "sample_id", np.arange(len(sample), dtype=np.int64))

    if sample.duplicated(["kind", "contig", "strand", "position"]).any():
        dup = sample.loc[
            sample.duplicated(["kind", "contig", "strand", "position"], keep=False)
        ].head()
        raise AssertionError(f"Duplicate sampled coordinates:\n{dup}")

    meta = {
        "complete": True,
        "source_checkpoint": str(BASE_CHECKPOINT),
        "source_checkpoint_sha256": sha256_file(BASE_CHECKPOINT),
        "source_shards": str(CV_SHARDS),
        "source_shard_count": len(files),
        "score_definition_for_negative_rank": "sigmoid(EXP045 logit), rounded to 5 decimals",
        "negative_tie_break": "stable shard/path/row/position order, exactly as EXP047 risk cache",
        "negative_band_edges_fraction": NEGATIVE_BAND_EDGES.tolist(),
        "negative_band_names": NEGATIVE_BAND_NAMES,
        "negative_population_counts": population_counts.tolist(),
        "negative_population_fractions": population_fractions.tolist(),
        "negative_sample_per_band": NEGATIVE_SAMPLE_PER_BAND,
        "positive_population_n": int(len(pos_scores)),
        "positive_quartile_edges_EXP045_score": quartile_edges.tolist(),
        "positive_sample_per_quartile": POSITIVE_SAMPLE_PER_QUARTILE,
        "sample_rows": int(len(sample)),
        "unique_tiles": int(sample[["contig", "target_start"]].drop_duplicates().shape[0]),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_csv(SAMPLE_MANIFEST, sample)
    atomic_json(SAMPLE_META, meta)
    print(sample.groupby(["kind", "group"]).size())
    print(json.dumps(meta, indent=2))


def _soft_gpu_pause(active_seconds: float) -> None:
    if GPU_DUTY_CYCLE >= 0.999999:
        return
    time.sleep(max(float(active_seconds), 0.0) * (1.0 / GPU_DUTY_CYCLE - 1.0))


def score_model(label: str) -> None:
    import torch
    import torch_directml

    if label == "EXP047B":
        checkpoint_path = EXP047_CHECKPOINT
        output_path = SCORES_047
    elif label == "EXP048":
        checkpoint_path = EXP048_CHECKPOINT
        output_path = SCORES_048
    else:
        raise ValueError(label)

    if output_path.exists():
        existing = pd.read_csv(output_path)
        sample = pd.read_csv(SAMPLE_MANIFEST)
        if len(existing) == len(sample) and np.array_equal(
            existing["sample_id"].to_numpy(np.int64),
            sample["sample_id"].to_numpy(np.int64),
        ):
            print(f"{label} sample scores already complete: {output_path}")
            return

    sample = pd.read_csv(SAMPLE_MANIFEST)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    split = load_split(ROOT)
    sequences, _ = load_baseline_sequences(split)
    config = ProfileWindowConfig(**checkpoint["profile_config"])
    generator = ProfileWindowGenerator(
        prepared_split=split, sequences=sequences, part="train", config=config, seed=42
    )
    tile_map = {
        (str(row.contig), int(row.target_start)): index
        for index, row in generator.tile_index.iterrows()
    }

    grouped = sample.groupby(["contig", "target_start"], sort=True).groups
    tile_groups = list(grouped.items())
    missing = [key for key, _ in tile_groups if key not in tile_map]
    if missing:
        raise KeyError(f"Sample coordinates missing from train generator: {missing[:5]}")

    device = torch_directml.device()
    model = create_cnn_presence_intensity(channels=512)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    scores = np.full(len(sample), np.nan, dtype=np.float32)
    batch_size = 4
    started = time.perf_counter()

    with torch.no_grad():
        for start in range(0, len(tile_groups), batch_size):
            chunk = tile_groups[start : start + batch_size]
            indices = [tile_map[key] for key, _ in chunk]
            selected = generator.tile_index.iloc[indices]
            batch = generator.make_batch(selected)

            phase_started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                plus, minus_reverse, _, _ = forward_both_strands_multitask(
                    model,
                    batch.x_both_strands.to(device),
                    batch_size=len(chunk),
                    context_flank=config.context_flank,
                    target_length=config.target_length,
                )
                plus_np = sigmoid_numpy(plus[:, 0].detach().cpu().numpy())
                minus_np = sigmoid_numpy(
                    minus_reverse[:, 0].detach().cpu().numpy()[:, ::-1]
                )
            fallbacks = [
                str(x.message)
                for x in caught
                if "fall back" in str(x.message).lower()
            ]
            if fallbacks:
                raise RuntimeError("DirectML CPU fallback: " + " | ".join(fallbacks))

            _soft_gpu_pause(time.perf_counter() - phase_started)

            for local_batch, ((contig, target_start), row_ids) in enumerate(chunk):
                row_ids = np.asarray(list(row_ids), dtype=np.int64)
                locals_ = (
                    sample.loc[row_ids, "position"].to_numpy(np.int64)
                    - int(target_start)
                )
                strands = sample.loc[row_ids, "strand"].astype(str).to_numpy()
                values = np.where(
                    strands == "+",
                    plus_np[local_batch, locals_],
                    minus_np[local_batch, locals_],
                )
                scores[row_ids] = values.astype(np.float32)

            completed = min(start + len(chunk), len(tile_groups))
            if start == 0 or completed % 250 < len(chunk) or completed == len(tile_groups):
                print(
                    f"[{label}] tiles {completed:,}/{len(tile_groups):,} "
                    f"elapsed={(time.perf_counter()-started)/60:.1f} min",
                    flush=True,
                )
            del batch, plus, minus_reverse, plus_np, minus_np

    if not np.all(np.isfinite(scores)):
        bad = np.flatnonzero(~np.isfinite(scores))
        raise AssertionError(f"Non-finite/missing scores: {bad[:20].tolist()}")

    out = sample[["sample_id", "kind", "group", "contig", "strand", "position"]].copy()
    out[f"score_{label}"] = scores
    atomic_csv(output_path, out)
    print(f"Saved: {output_path}")


def _safe_bce_negative(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return -np.log1p(-p)


def _auc_pair(pos: np.ndarray, neg: np.ndarray) -> float:
    # Mann-Whitney AUC with average ranks for ties, implemented without sklearn.
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    values = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos), dtype=np.uint8), np.zeros(len(neg), dtype=np.uint8)])
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = avg_rank
        start = end
    rank_sum_pos = ranks[labels == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _bootstrap_mean_ci(values: np.ndarray, seed: int, n_boot: int = 1_000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def analyze() -> None:
    required = [SAMPLE_MANIFEST, SAMPLE_META, SCORES_047, SCORES_048]
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    sample = pd.read_csv(SAMPLE_MANIFEST)
    meta = json.loads(SAMPLE_META.read_text(encoding="utf-8"))
    s47 = pd.read_csv(SCORES_047)[["sample_id", "score_EXP047B"]]
    s48 = pd.read_csv(SCORES_048)[["sample_id", "score_EXP048"]]
    frame = sample.merge(s47, on="sample_id", validate="one_to_one").merge(
        s48, on="sample_id", validate="one_to_one"
    )
    atomic_csv(RUN_DIR / "train_tail_scored_sample.csv", frame)

    score_cols = ["score_EXP045", "score_EXP047B", "score_EXP048"]
    negative = frame.loc[frame["kind"] == "negative"].copy()
    positive = frame.loc[frame["kind"] == "positive"].copy()

    population = pd.DataFrame(
        {
            "group": meta["negative_band_names"],
            "population_n": meta["negative_population_counts"],
            "population_fraction": meta["negative_population_fractions"],
        }
    )

    neg_rows = []
    for group, part in negative.groupby("group", sort=False):
        row = {"group": group, "sample_n": len(part)}
        for col in score_cols:
            x = part[col].to_numpy(np.float64)
            suffix = col.removeprefix("score_")
            row[f"mean_{suffix}"] = float(x.mean())
            row[f"median_{suffix}"] = float(np.median(x))
            row[f"p90_{suffix}"] = float(np.quantile(x, 0.90))
            row[f"frac_gt_0p9_{suffix}"] = float((x > 0.9).mean())
            row[f"frac_gt_0p5_{suffix}"] = float((x > 0.5).mean())
            row[f"mean_neg_BCE_{suffix}"] = float(_safe_bce_negative(x).mean())
        d47 = part["score_EXP047B"].to_numpy(np.float64) - part["score_EXP045"].to_numpy(np.float64)
        d48 = part["score_EXP048"].to_numpy(np.float64) - part["score_EXP047B"].to_numpy(np.float64)
        row["mean_delta_047_vs_045"] = float(d47.mean())
        row["mean_delta_048_vs_047"] = float(d48.mean())
        row["frac_lower_047_vs_045"] = float((d47 < 0).mean())
        row["frac_lower_048_vs_047"] = float((d48 < 0).mean())
        lo, hi = _bootstrap_mean_ci(d48, seed=4800 + NEGATIVE_BAND_NAMES.index(group))
        row["delta_048_vs_047_ci95_low"] = lo
        row["delta_048_vs_047_ci95_high"] = hi
        neg_rows.append(row)

    neg_summary = pd.DataFrame(neg_rows).merge(population, on="group", validate="one_to_one")
    neg_summary["population_weighted_BCE_EXP045"] = (
        neg_summary["population_fraction"] * neg_summary["mean_neg_BCE_EXP045"]
    )
    neg_summary["population_weighted_BCE_EXP047B"] = (
        neg_summary["population_fraction"] * neg_summary["mean_neg_BCE_EXP047B"]
    )
    neg_summary["population_weighted_BCE_EXP048"] = (
        neg_summary["population_fraction"] * neg_summary["mean_neg_BCE_EXP048"]
    )
    band_order = {name: i for i, name in enumerate(NEGATIVE_BAND_NAMES)}
    neg_summary["_order"] = neg_summary["group"].map(band_order)
    neg_summary = neg_summary.sort_values("_order").drop(columns="_order")
    atomic_csv(RUN_DIR / "negative_band_summary.csv", neg_summary)

    pos_rows = []
    for group, part in positive.groupby("group", sort=False):
        row = {"group": group, "sample_n": len(part)}
        for col in score_cols:
            x = part[col].to_numpy(np.float64)
            suffix = col.removeprefix("score_")
            row[f"mean_{suffix}"] = float(x.mean())
            row[f"median_{suffix}"] = float(np.median(x))
            row[f"p10_{suffix}"] = float(np.quantile(x, 0.10))
        d47 = part["score_EXP047B"].to_numpy(np.float64) - part["score_EXP045"].to_numpy(np.float64)
        d48 = part["score_EXP048"].to_numpy(np.float64) - part["score_EXP047B"].to_numpy(np.float64)
        row["mean_delta_047_vs_045"] = float(d47.mean())
        row["mean_delta_048_vs_047"] = float(d48.mean())
        row["frac_higher_047_vs_045"] = float((d47 > 0).mean())
        row["frac_higher_048_vs_047"] = float((d48 > 0).mean())
        lo, hi = _bootstrap_mean_ci(d48, seed=4900 + int(group[1:]))
        row["delta_048_vs_047_ci95_low"] = lo
        row["delta_048_vs_047_ci95_high"] = hi
        pos_rows.append(row)
    pos_summary = pd.DataFrame(pos_rows).sort_values("group")
    atomic_csv(RUN_DIR / "positive_quartile_summary.csv", pos_summary)

    # Pairwise ranking separation: positives should outrank each negative band.
    separation_rows = []
    positive_sets = {
        "all_positive_quartiles": positive,
        "weak_positive_Q1": positive.loc[positive["group"] == "Q1"],
    }
    for pos_name, pos_part in positive_sets.items():
        for neg_group in NEGATIVE_BAND_NAMES:
            neg_part = negative.loc[negative["group"] == neg_group]
            pop_frac = float(
                population.loc[population["group"] == neg_group, "population_fraction"].iloc[0]
            )
            row = {
                "positive_set": pos_name,
                "negative_group": neg_group,
                "negative_population_fraction": pop_frac,
            }
            for col in score_cols:
                suffix = col.removeprefix("score_")
                auc = _auc_pair(
                    pos_part[col].to_numpy(np.float64),
                    neg_part[col].to_numpy(np.float64),
                )
                row[f"AUC_{suffix}"] = auc
                row[f"inversion_rate_{suffix}"] = 1.0 - auc
                row[f"weighted_inversion_{suffix}"] = pop_frac * (1.0 - auc)
            row["delta_AUC_047_vs_045"] = row["AUC_EXP047B"] - row["AUC_EXP045"]
            row["delta_AUC_048_vs_047"] = row["AUC_EXP048"] - row["AUC_EXP047B"]
            separation_rows.append(row)
    separation = pd.DataFrame(separation_rows)
    separation["_order"] = separation["negative_group"].map(band_order)
    separation = separation.sort_values(["positive_set", "_order"]).drop(columns="_order")
    atomic_csv(RUN_DIR / "negative_vs_positive_separation.csv", separation)

    # Aggregate the exact user hypothesis zone 0.2-1%.
    zone_parts = negative.loc[negative["group"].isin(["0.2-0.5%", "0.5-1%"])].copy()
    zone_weights = {
        "0.2-0.5%": 0.003,
        "0.5-1%": 0.005,
    }
    # Normalize sample weights so the sampled 0.2-1% zone represents its natural 0.8% composition.
    zone_parts["_natural_weight"] = zone_parts["group"].map(
        {
            g: w / max(1, int((zone_parts["group"] == g).sum()))
            for g, w in zone_weights.items()
        }
    )
    # Weighted score means are enough here; pairwise AUC remains reported separately by sub-band.
    zone_summary = {
        "zone": "0.2-1%",
        "natural_fraction": 0.008,
        "sample_n": int(len(zone_parts)),
    }
    w = zone_parts["_natural_weight"].to_numpy(np.float64)
    w = w / w.sum()
    for col in score_cols:
        zone_summary[f"weighted_mean_{col.removeprefix('score_')}"] = float(
            np.sum(w * zone_parts[col].to_numpy(np.float64))
        )
    zone_summary["weighted_mean_delta_048_vs_047"] = (
        zone_summary["weighted_mean_EXP048"] - zone_summary["weighted_mean_EXP047B"]
    )
    atomic_csv(RUN_DIR / "user_hypothesis_zone_0p2_1_summary.csv", pd.DataFrame([zone_summary]))

    # Existing validation fixed cohorts: no new validation inference.
    validation_summary = []
    if EXP047_COHORT.is_file() and EXP048_COHORT.is_file():
        v47 = pd.read_csv(EXP047_COHORT).rename(columns={"score_B": "score_EXP047B"})
        v48 = pd.read_csv(EXP048_COHORT).rename(columns={"score_B": "score_EXP048"})
        keys = ["contig", "strand", "position", "baseline_score", "cohort"]
        merged = v47[keys + ["score_EXP047B"]].merge(
            v48[keys + ["score_EXP048"]], on=keys, validate="one_to_one"
        )
        merged["delta_048_vs_047"] = merged["score_EXP048"] - merged["score_EXP047B"]
        atomic_csv(RUN_DIR / "validation_fixed_cohort_EXP047B_vs_EXP048.csv", merged)
        validation_summary = (
            merged.groupby("cohort", as_index=False)
            .agg(
                n=("delta_048_vs_047", "size"),
                mean_EXP047B=("score_EXP047B", "mean"),
                mean_EXP048=("score_EXP048", "mean"),
                mean_delta=("delta_048_vs_047", "mean"),
                frac_higher_048=("delta_048_vs_047", lambda x: float((x > 0).mean())),
                frac_lower_048=("delta_048_vs_047", lambda x: float((x < 0).mean())),
            )
        )
        atomic_csv(RUN_DIR / "validation_fixed_cohort_summary.csv", validation_summary)

    # Machine-readable diagnostic facts, not a decision to train another model.
    weak_sep = separation.loc[separation["positive_set"] == "weak_positive_Q1"].copy()
    strongest_047 = weak_sep.sort_values("delta_AUC_047_vs_045", ascending=False).iloc[0]
    worst_048 = weak_sep.sort_values("delta_AUC_048_vs_047", ascending=True).iloc[0]
    facts = {
        "complete": True,
        "purpose": (
            "Post-hoc causal diagnostic of where EXP047 helped and EXP048 hurt; "
            "not a new loss selection by itself."
        ),
        "train_negative_bands": NEGATIVE_BAND_NAMES,
        "sample_rows": int(len(frame)),
        "unique_tiles": int(frame[["contig", "target_start"]].drop_duplicates().shape[0]),
        "strongest_EXP047_AUC_gain_vs_EXP045_for_weak_positives": {
            "band": str(strongest_047["negative_group"]),
            "delta_AUC": float(strongest_047["delta_AUC_047_vs_045"]),
        },
        "largest_EXP048_AUC_damage_vs_EXP047_for_weak_positives": {
            "band": str(worst_048["negative_group"]),
            "delta_AUC": float(worst_048["delta_AUC_048_vs_047"]),
        },
        "user_hypothesis_zone_0p2_1": zone_summary,
        "validation_fixed_cohorts_reused_without_new_validation_inference": bool(
            EXP047_COHORT.is_file() and EXP048_COHORT.is_file()
        ),
        "files": {
            "negative_band_summary": str(RUN_DIR / "negative_band_summary.csv"),
            "positive_quartile_summary": str(RUN_DIR / "positive_quartile_summary.csv"),
            "separation": str(RUN_DIR / "negative_vs_positive_separation.csv"),
            "zone_0p2_1": str(RUN_DIR / "user_hypothesis_zone_0p2_1_summary.csv"),
            "validation_fixed_cohort_summary": str(RUN_DIR / "validation_fixed_cohort_summary.csv"),
        },
    }
    atomic_json(RUN_DIR / "diagnostic_summary.json", facts)

    print("\n=== NEGATIVE BANDS ===")
    print(
        neg_summary[
            [
                "group",
                "population_n",
                "population_fraction",
                "mean_EXP045",
                "mean_EXP047B",
                "mean_EXP048",
                "mean_delta_047_vs_045",
                "mean_delta_048_vs_047",
                "frac_gt_0p9_EXP047B",
                "frac_gt_0p9_EXP048",
            ]
        ].to_string(index=False)
    )
    print("\n=== POSITIVE QUARTILES ===")
    print(pos_summary.to_string(index=False))
    print("\n=== WEAK POSITIVE Q1 SEPARATION ===")
    print(
        weak_sep[
            [
                "negative_group",
                "AUC_EXP045",
                "AUC_EXP047B",
                "AUC_EXP048",
                "delta_AUC_047_vs_045",
                "delta_AUC_048_vs_047",
            ]
        ].to_string(index=False)
    )
    print("\n=== 0.2-1% USER HYPOTHESIS ZONE ===")
    print(pd.DataFrame([zone_summary]).to_string(index=False))
    if isinstance(validation_summary, pd.DataFrame) and not validation_summary.empty:
        print("\n=== EXISTING VALIDATION FIXED COHORTS ===")
        print(validation_summary.to_string(index=False))
    print(f"\nSaved diagnostics under: {RUN_DIR}")


def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("build-sample")
    score = sub.add_parser("score-model")
    score.add_argument("--model", choices=["EXP047B", "EXP048"], required=True)
    sub.add_parser("analyze")
    return p.parse_args()


def main():
    args = parse_args()
    if args.command == "build-sample":
        build_sample()
    elif args.command == "score-model":
        score_model(args.model)
    elif args.command == "analyze":
        analyze()
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
