"""CNN-EXP-048: WIDTH512 ultra-tail risk refinement from EXP045 @50k.

The command is intentionally process-oriented.  The notebook launches each long
DirectML stage as a fresh process so a backend crash cannot destroy already
committed checkpoints or the other branch.
"""

from __future__ import annotations

import argparse
import atexit
import copy
from datetime import date
from functools import lru_cache
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch


EXPERIMENT_ID = "CNN-EXP-048"
EXPERIMENT_SLUG = "CNN_EXP048_WIDTH512_ULTRA_TAIL_RISK_REFINEMENT"
BASE_EXPERIMENT = "CNN-EXP-045"
BASE_STEP = 50_000
CONTINUATION_STEPS = 20_000
BATCH_SIZE = 4
LOG_EVERY = 250
LATEST_EVERY = 500
SNAPSHOT_STEPS = {2_000, 5_000, 10_000, 15_000, 20_000}
WARMUP_STEPS = 500
START_LR = 3e-5
PEAK_LR = 1e-4
END_LR = 1e-5
INTENSITY_LOSS_WEIGHT = 0.1
NEGATIVE_RANK_CUT_FRACTIONS = np.asarray([0.001, 0.01, 0.05, 0.15], dtype=np.float64)
NEGATIVE_STRATA_NAMES = ["top_0_0p1pct", "top_0p1_1pct", "top_1_5pct", "top_5_15pct", "bottom_85pct"]
TARGET_NEGATIVE_MIXTURE = np.asarray([0.065, 0.065, 0.145, 0.175, 0.55], dtype=np.float64)
GPU_DUTY_CYCLE = float(os.environ.get("EPIC_GPU_DUTY_CYCLE", "0.90"))
if not (0.50 <= GPU_DUTY_CYCLE <= 1.0):
    raise RuntimeError(f"EPIC_GPU_DUTY_CYCLE must be in [0.50,1.0], got {GPU_DUTY_CYCLE}")
SCORE_LEVELS = 100_001
KEY_SHIFT = np.uint64(56)
REGION_MASK = (np.uint64(1) << KEY_SHIFT) - np.uint64(1)
BIT_SHIFTS = np.arange(32, dtype=np.uint32)
EXPECTED_CHANNELS = 512
EXPECTED_RF = 1029
EXPECTED_PARAMETERS = 14_708_738


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
    DirectMLAdam,
    ProfileWindowConfig,
    ProfileWindowGenerator,
    create_cnn_presence_intensity,
    dml_weighted_huber,
    dml_weighted_profile_bce_with_logits,
    evaluate_presence_intensity_model,
    forward_both_strands_multitask,
)
from ml_project.epic_data import load_split


RUN_DIR = ROOT / "artifacts" / "experiments" / EXPERIMENT_SLUG / "run_001"
BASE_RUN = (
    ROOT
    / "artifacts"
    / "experiments"
    / "CNN_EXP045_WIDTH512_50K_TWO_PHASE_LR_RF1029"
    / "run_001"
)
BASE_CHECKPOINT = BASE_RUN / "model_step_50000.pt"
BASE_METRICS = BASE_RUN / "validation_metrics_report.csv"
EXP046_RUN = (
    ROOT
    / "artifacts"
    / "experiments"
    / "CNN_EXP046B_FROZEN_REGIONAL_CORRECTION_R32"
    / "run_001"
)
CV_SHARDS = EXP046_RUN / "_cv_eval_v3"
ORACLE_RUN = (
    ROOT
    / "artifacts"
    / "experiments"
    / "CNN_EXP046_REGIONAL_ACTIVITY_GATE"
    / "run_001"
)
RISK_CACHE_DIR = RUN_DIR / "risk_strata_cache"
RISK_CACHE = RISK_CACHE_DIR / "train_negative_stratum_u8.dat"
RISK_MANIFEST = RISK_CACHE_DIR / "risk_strata_manifest.json"
PREREG_PATH = RUN_DIR / "EXP048_preregistration.json"
RESULT_PATH = RUN_DIR / "EXP048_RESULT.json"
EXP047_RUN = ROOT / "artifacts" / "experiments" / "CNN_EXP047_WIDTH512_PAIRED_RISK_LOSS_CONTINUATION" / "run_001"
EXP047_B_METRICS = EXP047_RUN / "branch_B" / "validation_metrics_report.csv"
EXP047_B_STRONG = EXP047_RUN / "branch_B" / "strong_validation_metrics.csv"
EXP047_B_COHORT = EXP047_RUN / "branch_B" / "fixed_cohort_scores.csv"
NOTEBOOK_REL = (
    "notebooks/experiments/CNN_experiments/"
    "CNN_EXP048_ULTRA_TAIL_RISK_REFINEMENT.ipynb"
)


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(json_ready(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
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


def atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


@lru_cache(maxsize=32)
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_state_on_cpu(model: torch.nn.Module) -> dict:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def tree_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(tree_to_cpu(item) for item in value)
    return value


def planned_lr(local_step: int) -> float:
    if not 1 <= local_step <= CONTINUATION_STEPS:
        raise ValueError(f"local_step outside 1..{CONTINUATION_STEPS}: {local_step}")
    if local_step <= WARMUP_STEPS:
        fraction = local_step / WARMUP_STEPS
        return START_LR + fraction * (PEAK_LR - START_LR)
    fraction = (local_step - WARMUP_STEPS) / (CONTINUATION_STEPS - WARMUP_STEPS)
    return END_LR + 0.5 * (PEAK_LR - END_LR) * (1.0 + math.cos(math.pi * fraction))


def soft_gpu_duty_pause(active_seconds: float) -> None:
    """Soft throughput throttle; not a hard hardware-utilization cap."""
    if GPU_DUTY_CYCLE >= 0.999999:
        return
    active_seconds = max(float(active_seconds), 0.0)
    time.sleep(active_seconds * (1.0 / GPU_DUTY_CYCLE - 1.0))


def sigmoid_numpy(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(logits)
    positive = logits >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_value = np.exp(logits[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    return output


def score_bins(logits: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(sigmoid_numpy(logits), 0.0, 1.0) * 100_000).astype(np.int32)


def unpack_mask(mask_u32: np.ndarray) -> np.ndarray:
    values = np.asarray(mask_u32, dtype=np.uint32)
    return ((values[:, None] >> BIT_SHIFTS[None, :]) & np.uint32(1)).astype(bool)


def branch_dir(branch: str) -> Path:
    if branch not in {"A", "B"}:
        raise ValueError("branch must be A or B")
    return RUN_DIR / f"branch_{branch}"


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_process_lock(path: Path, stage: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pid": os.getpid(), "stage": stage, "created": time.time()}
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                existing_pid = int(existing.get("pid", -1))
            except Exception:
                existing_pid = -1
                existing = {"unreadable": True}
            if _pid_is_running(existing_pid):
                raise RuntimeError(
                    f"Stage already has a live process lock: {path}; details={existing}"
                )
            path.unlink(missing_ok=True)
            continue
        else:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            break
    else:
        raise RuntimeError(f"Could not acquire process lock: {path}")

    def release() -> None:
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
            if int(current.get("pid", -1)) == os.getpid():
                path.unlink(missing_ok=True)
        except FileNotFoundError:
            pass

    atexit.register(release)


def preregister() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "created": date.today().isoformat(),
        "status": "pre_registered",
        "primary_question": (
            "Does redistributing the already-adopted EXP047 top-1% negative loss mass "
            "toward the top 0.1% improve natural genome-wide nucleotide AP?"
        ),
        "branch_point": str(BASE_CHECKPOINT),
        "reference_model": "EXP047-B historical matched control",
        "candidate": (
            "same EXP045@50k continuation as EXP047-B, but old top-1% stratum is split "
            "into top 0-0.1% and 0.1-1% while preserving total top-1% negative mass 0.13"
        ),
        "negative_strata": NEGATIVE_STRATA_NAMES,
        "stratum_source": (
            "EXP045 train-negative logits already saved in corrected EXP046B CV shards; "
            "validation is excluded"
        ),
        "natural_negative_mixture": [0.001, 0.009, 0.04, 0.10, 0.85],
        "candidate_negative_mixture": TARGET_NEGATIVE_MIXTURE.tolist(),
        "candidate_negative_weight_multipliers_ideal": (
            TARGET_NEGATIVE_MIXTURE
            / np.asarray([0.001, 0.009, 0.04, 0.10, 0.85], dtype=np.float64)
        ).tolist(),
        "top1_total_mass_preserved": float(TARGET_NEGATIVE_MIXTURE[:2].sum()),
        "continuation": {
            "steps": CONTINUATION_STEPS,
            "batch_size": BATCH_SIZE,
            "warmup_steps": WARMUP_STEPS,
            "start_lr": START_LR,
            "peak_lr": PEAK_LR,
            "end_lr": END_LR,
            "schedule": "linear warm-up then cosine decay",
        },
        "gpu_policy": {
            "type": "soft duty-cycle throttle after optimizer update",
            "target": GPU_DUTY_CYCLE,
            "hard_hardware_cap": False,
        },
        "primary_metric": "natural full-validation nucleotide AP after 5-decimal quantization",
        "selection_rule": (
            "ADOPT_ULTRA_TAIL only if candidate-EXP047B overall AP >= 0.010, "
            "candidate improves AP on every validation contig, and overall Spearman "
            "does not drop by more than 0.003; raw-AP and LOW_TP/top-budget audits remain mandatory"
        ),
        "validation_policy": "one full final candidate scan; no checkpoint selection on validation",
        "checkpoint_policy": (
            "atomic latest.pt every 500 steps; immutable snapshots at 2k/5k/10k/15k/20k"
        ),
        "scientific_caveat": (
            "The validation split has been inspected in prior experiments; EXP048 is exploratory, "
            "not a fresh unbiased estimate."
        ),
    }
    if PREREG_PATH.exists():
        existing = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
        expected = dict(payload)
        expected["created"] = existing.get("created")
        if existing != expected:
            raise RuntimeError(f"Refusing to overwrite changed pre-registration: {PREREG_PATH}")
    else:
        atomic_json(PREREG_PATH, payload)
    print(f"Pre-registration: {PREREG_PATH}")


def inspect_inputs() -> None:
    required = [BASE_CHECKPOINT, BASE_METRICS, CV_SHARDS, EXP047_B_METRICS]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")
    checkpoint = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=False)
    required_keys = {
        "model_state_dict", "optimizer_state_dict", "generator_rng_state", "torch_rng_state",
        "model_config", "profile_config", "step",
    }
    absent = required_keys - set(checkpoint)
    if absent:
        raise ValueError(f"Base checkpoint misses keys: {sorted(absent)}")
    assert int(checkpoint["step"]) == BASE_STEP
    assert int(checkpoint["model_config"]["channels"]) == EXPECTED_CHANNELS
    assert int(checkpoint["model_config"]["theoretical_rf"]) == EXPECTED_RF
    model = create_cnn_presence_intensity(channels=EXPECTED_CHANNELS)
    model.load_state_dict(checkpoint["model_state_dict"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count == EXPECTED_PARAMETERS
    fold_dirs = sorted(path for path in CV_SHARDS.glob("fold_*") if path.is_dir())
    shard_files = [file for fold in fold_dirs for file in sorted((fold / "shards").glob("*.npz"))]
    if len(fold_dirs) != 9 or not shard_files:
        raise RuntimeError(f"Incomplete corrected CV shards: folds={len(fold_dirs)}, files={len(shard_files)}")
    with np.load(shard_files[0], allow_pickle=False) as sample:
        expected = {"region_key", "base_logit", "valid_mask", "positive_mask"}
        if not expected <= set(sample.files):
            raise ValueError(f"Shard misses arrays: {sorted(expected - set(sample.files))}")
        assert sample["base_logit"].shape[1] == 32
    print(f"Python: {sys.executable}")
    print(f"Base checkpoint: step={checkpoint['step']}, sha256={sha256_file(BASE_CHECKPOINT)}")
    print(f"Model parameters: {parameter_count:,}")
    print(f"Corrected train shards: folds={len(fold_dirs)}, files={len(shard_files):,}")
    print(f"LR probes: step1={planned_lr(1):.8g}, warmup={planned_lr(WARMUP_STEPS):.8g}, final={planned_lr(CONTINUATION_STEPS):.8g}")


def iter_shards() -> list[Path]:
    files: list[Path] = []
    for fold in sorted(path for path in CV_SHARDS.glob("fold_*") if path.is_dir()):
        files.extend(sorted((fold / "shards").glob("tiles_*.npz")))
    return files


def prepare_risk_cache() -> None:
    if RISK_CACHE.exists() and RISK_MANIFEST.exists():
        manifest = json.loads(RISK_MANIFEST.read_text(encoding="utf-8"))
        expected_bytes = int(np.prod(manifest["shape"], dtype=np.int64))
        if RISK_CACHE.stat().st_size == expected_bytes and manifest["complete"]:
            actual_sha = sha256_file(RISK_CACHE)
            if "cache_sha256" not in manifest:
                manifest["cache_sha256"] = actual_sha
                atomic_json(RISK_MANIFEST, manifest)
            elif manifest["cache_sha256"] != actual_sha:
                raise RuntimeError("Completed risk cache checksum mismatch")
            print(f"Complete risk cache already exists: {RISK_CACHE}")
            return
        raise RuntimeError("Risk cache/manifest exists but is incomplete or inconsistent")

    RISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    split = load_split(ROOT)
    sequences, _ = load_baseline_sequences(split)
    base = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=False)
    profile_config = ProfileWindowConfig(**base["profile_config"])
    generator = ProfileWindowGenerator(
        prepared_split=split, sequences=sequences, part="train", config=profile_config, seed=42
    )
    tiles = generator.tile_index
    train_contigs = sorted(set(split.intervals("train")["contig"].astype(str)))
    validation_contigs = sorted(set(split.intervals("validation")["contig"].astype(str)))
    all_contigs = train_contigs + validation_contigs
    code_to_contig = {index: contig for index, contig in enumerate(all_contigs)}
    tile_lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for contig, frame in tiles.groupby(tiles["contig"].astype(str), sort=False):
        group_ids = (frame["target_start"].to_numpy(np.int64) // 1024).astype(np.int64)
        tile_ids = frame["tile_id"].to_numpy(np.int64)
        order = np.argsort(group_ids, kind="stable")
        tile_lookup[str(contig)] = (group_ids[order], tile_ids[order])

    files = iter_shards()
    histogram = np.zeros(SCORE_LEVELS, dtype=np.int64)
    started = time.perf_counter()
    for file_index, path in enumerate(files, start=1):
        with np.load(path, allow_pickle=False) as data:
            valid = unpack_mask(data["valid_mask"])
            positive = unpack_mask(data["positive_mask"])
            negative = valid & ~positive
            bins = score_bins(data["base_logit"])
            histogram += np.bincount(bins[negative], minlength=SCORE_LEVELS)
        if file_index == 1 or file_index % 500 == 0 or file_index == len(files):
            print(f"risk histogram: {file_index:,}/{len(files):,} shards", flush=True)
    total_negative = int(histogram.sum())
    if total_negative != generator.total_negative:
        raise AssertionError(f"Negative population mismatch: shards={total_negative}, generator={generator.total_negative}")

    cut_counts = np.asarray(
        [round(total_negative * fraction) for fraction in NEGATIVE_RANK_CUT_FRACTIONS],
        dtype=np.int64,
    )
    higher = np.cumsum(histogram[::-1], dtype=np.int64)[::-1] - histogram
    seen_by_score = np.zeros(SCORE_LEVELS, dtype=np.int64)
    shape = (len(tiles), 2, profile_config.target_length)
    temp = RISK_CACHE.with_suffix(RISK_CACHE.suffix + ".tmp")
    cache = np.memmap(temp, mode="w+", dtype=np.uint8, shape=shape)
    cache[:] = np.uint8(255)
    flat_cache = cache.reshape(-1)
    realized = np.zeros(len(NEGATIVE_STRATA_NAMES), dtype=np.int64)
    assigned = 0

    for file_index, path in enumerate(files, start=1):
        with np.load(path, allow_pickle=False) as data:
            keys = data["region_key"].astype(np.uint64, copy=False)
            valid = unpack_mask(data["valid_mask"])
            positive = unpack_mask(data["positive_mask"])
            negative = valid & ~positive
            bins = score_bins(data["base_logit"])
            strata = np.full(bins.shape, 255, dtype=np.uint8)
            negative_scores = bins[negative]
            order = np.argsort(negative_scores, kind="stable")
            ordered_scores = negative_scores[order]
            group_starts = np.r_[
                np.int64(0),
                np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]).astype(np.int64) + 1,
            ]
            group_ends = np.r_[group_starts[1:], np.int64(len(ordered_scores))]
            group_lengths = group_ends - group_starts
            unique_scores = ordered_scores[group_starts]
            offsets = np.arange(len(ordered_scores), dtype=np.int64) - np.repeat(
                group_starts, group_lengths
            )
            rank_bases = np.repeat(
                higher[unique_scores] + seen_by_score[unique_scores], group_lengths
            )
            ordered_ranks = rank_bases + offsets
            ordered_strata = np.searchsorted(
                cut_counts, ordered_ranks, side="right"
            ).astype(np.uint8)
            stratum_values = np.empty(len(ordered_strata), dtype=np.uint8)
            stratum_values[order] = ordered_strata
            strata[negative] = stratum_values
            seen_by_score[unique_scores] += group_lengths
            realized += np.bincount(ordered_strata, minlength=len(NEGATIVE_STRATA_NAMES))

            prefix = keys >> KEY_SHIFT
            contig_codes = (prefix // np.uint64(2)).astype(np.int64)
            strand_codes = (prefix % np.uint64(2)).astype(np.int64)
            if len(np.unique(contig_codes)) != 1:
                raise AssertionError(f"Mixed contigs in shard: {path}")
            contig = code_to_contig[int(contig_codes[0])]
            region_ids = (keys & REGION_MASK).astype(np.int64)
            tile_groups = region_ids // 32
            local_regions = region_ids % 32
            known_groups, known_tile_ids = tile_lookup[contig]
            positions = np.searchsorted(known_groups, tile_groups)
            if np.any(positions >= len(known_groups)) or not np.array_equal(known_groups[positions], tile_groups):
                raise AssertionError(f"Region-to-tile mapping failed: {path}")
            tile_ids = known_tile_ids[positions]
            destination = (
                tile_ids[:, None] * (2 * profile_config.target_length)
                + strand_codes[:, None] * profile_config.target_length
                + local_regions[:, None] * 32
                + BIT_SHIFTS[None, :].astype(np.int64)
            )
            selected_destination = destination[negative]
            selected_strata = strata[negative]
            if np.any(flat_cache[selected_destination] != 255):
                raise AssertionError(f"Duplicate risk-cache destination: {path}")
            flat_cache[selected_destination] = selected_strata
            assigned += len(selected_strata)
        if file_index == 1 or file_index % 500 == 0 or file_index == len(files):
            print(f"risk assignment: {file_index:,}/{len(files):,} shards", flush=True)

    cache.flush()
    del flat_cache, cache
    gc.collect()
    if assigned != total_negative or int(realized.sum()) != total_negative:
        raise AssertionError("Risk cache did not cover the complete train-negative population")
    os.replace(temp, RISK_CACHE)
    fractions = realized.astype(np.float64) / total_negative
    if len(fractions) != len(TARGET_NEGATIVE_MIXTURE):
        raise AssertionError((fractions, TARGET_NEGATIVE_MIXTURE))
    multipliers = TARGET_NEGATIVE_MIXTURE / fractions
    manifest = {
        "complete": True,
        "created": date.today().isoformat(),
        "source_checkpoint": str(BASE_CHECKPOINT),
        "source_checkpoint_sha256": sha256_file(BASE_CHECKPOINT),
        "source_shard_root": str(CV_SHARDS),
        "source_shard_count": len(files),
        "score_definition": "sigmoid(EXP045 logit), rounded to 5 decimals",
        "tie_break": "stable shard/path/row/position order within equal score",
        "shape": list(shape),
        "dtype": "uint8",
        "cache_sha256": sha256_file(RISK_CACHE),
        "invalid_or_positive_code": 255,
        "rank_cut_counts": cut_counts.tolist(),
        "stratum_counts": realized.tolist(),
        "stratum_names": NEGATIVE_STRATA_NAMES,
        "natural_fractions": fractions.tolist(),
        "target_negative_mixture": TARGET_NEGATIVE_MIXTURE.tolist(),
        "candidate_negative_weight_multipliers": multipliers.tolist(),
        "total_train_negatives": total_negative,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(RISK_MANIFEST, manifest)
    print(json.dumps(manifest, indent=2))


def checkpoint_payload(
    *, branch: str, local_step: int, model, optimizer, generator, history: list[dict],
    elapsed_seconds: float, seen_position_strand: int, risk_manifest_sha256: str | None,
) -> dict:
    return {
        "experiment": EXPERIMENT_ID,
        "branch": branch,
        "model_type": "presence_intensity",
        "architecture": "rf1029_width512",
        "base_experiment": BASE_EXPERIMENT,
        "base_checkpoint": str(BASE_CHECKPOINT),
        "base_checkpoint_sha256": sha256_file(BASE_CHECKPOINT),
        "base_step": BASE_STEP,
        "local_step": int(local_step),
        "step": BASE_STEP + int(local_step),
        "model_state_dict": model_state_on_cpu(model),
        "optimizer_state_dict": tree_to_cpu(optimizer.state_dict()),
        "generator_rng_state": copy.deepcopy(generator.rng.bit_generator.state),
        "torch_rng_state": torch.get_rng_state(),
        "model_config": {
            "input_channels": 5, "channels": EXPECTED_CHANNELS,
            "output_channels": 2, "theoretical_rf": EXPECTED_RF,
        },
        "profile_config": {
            "target_length": generator.config.target_length,
            "context_flank": generator.config.context_flank,
            "positive_branch_fraction": generator.config.positive_branch_fraction,
        },
        "training_config": {
            "continuation_steps": CONTINUATION_STEPS,
            "batch_size": BATCH_SIZE,
            "warmup_steps": WARMUP_STEPS,
            "start_lr": START_LR,
            "peak_lr": PEAK_LR,
            "end_lr": END_LR,
            "intensity_loss_weight": INTENSITY_LOSS_WEIGHT,
            "presence_objective": (
                "original_class_balanced_BCE" if branch == "A"
                else "ultra_tail_five_strata_negative_BCE"
            ),
        },
        "risk_manifest_sha256": risk_manifest_sha256,
        "seen_position_strand": int(seen_position_strand),
        "elapsed_seconds": float(elapsed_seconds),
        "history": history,
    }


def smoke(branch: str) -> None:
    import torch_directml

    if branch == "B" and not (RISK_CACHE.exists() and RISK_MANIFEST.exists()):
        raise FileNotFoundError("Prepare risk cache before branch B smoke-test")
    split = load_split(ROOT)
    sequences, _ = load_baseline_sequences(split)
    base = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=False)
    config = ProfileWindowConfig(**base["profile_config"])
    generator = ProfileWindowGenerator(
        prepared_split=split, sequences=sequences, part="train", config=config, seed=42
    )
    generator.rng.bit_generator.state = copy.deepcopy(base["generator_rng_state"])
    torch.set_rng_state(base["torch_rng_state"])
    device = torch_directml.device()
    model = create_cnn_presence_intensity(channels=EXPECTED_CHANNELS)
    model.load_state_dict(base["model_state_dict"])
    model = model.to(device)
    optimizer = DirectMLAdam(model.parameters(), lr=planned_lr(1))
    optimizer.load_state_dict(base["optimizer_state_dict"])
    for group in optimizer.param_groups:
        group["lr"] = planned_lr(1)
    model.train()
    batch = generator.sample_batch(BATCH_SIZE)
    plus_weight = batch.plus_loss_weight
    minus_weight = batch.minus_loss_weight_reverse
    if branch == "B":
        manifest = json.loads(RISK_MANIFEST.read_text(encoding="utf-8"))
        cache = np.memmap(RISK_CACHE, mode="r", dtype=np.uint8, shape=tuple(manifest["shape"]))
        strata = np.asarray(cache[batch.tile_id.numpy().astype(np.int64, copy=False)])
        valid_negative = batch.mask.numpy() & ~batch.target.numpy().astype(bool)
        assert not np.any(strata[valid_negative] == 255)
        multiplier = np.ones(strata.shape, dtype=np.float32)
        for stratum, value in enumerate(manifest["candidate_negative_weight_multipliers"]):
            multiplier[strata == stratum] = np.float32(value)
        plus_weight = plus_weight * torch.from_numpy(multiplier[:, 0:1, :])
        minus_weight = minus_weight * torch.from_numpy(multiplier[:, 1:2, ::-1].copy())
    plus_count = batch.count[:, 0:1, :].contiguous()
    minus_count = batch.count[:, 1:2, :].flip(-1).contiguous()
    optimizer.zero_grad(set_to_none=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plus_presence, minus_presence, plus_intensity, minus_intensity = (
            forward_both_strands_multitask(
                model, batch.x_both_strands.to(device), batch_size=BATCH_SIZE,
                context_flank=config.context_flank, target_length=config.target_length,
            )
        )
        presence_loss = dml_weighted_profile_bce_with_logits(
            plus_presence, minus_presence, batch.plus_target.to(device),
            batch.minus_target_reverse.to(device), plus_weight.to(device), minus_weight.to(device),
        )
        intensity_loss = (
            dml_weighted_huber(
                plus_intensity, torch.log1p(plus_count).to(device),
                (2.0 * batch.plus_loss_weight * (plus_count > 0).float()).to(device), delta=1.0,
            )
            + dml_weighted_huber(
                minus_intensity, torch.log1p(minus_count).to(device),
                (2.0 * batch.minus_loss_weight_reverse * (minus_count > 0).float()).to(device), delta=1.0,
            )
        )
        total_loss = presence_loss + INTENSITY_LOSS_WEIGHT * intensity_loss
        values = [float(value.detach().cpu().item()) for value in (presence_loss, intensity_loss, total_loss)]
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError(values)
        total_loss.backward()
        optimizer.step()
    fallbacks = [str(item.message) for item in caught if "fall back" in str(item.message).lower()]
    if fallbacks:
        raise RuntimeError("DirectML CPU fallback detected: " + " | ".join(fallbacks))
    risk_hash = sha256_file(RISK_MANIFEST) if branch == "B" else None
    preflight_checkpoint = RUN_DIR / f"_preflight_branch_{branch}.pt"
    payload = checkpoint_payload(
        branch=branch, local_step=1, model=model, optimizer=optimizer, generator=generator,
        history=[], elapsed_seconds=0.0, seen_position_strand=int(batch.mask.sum()),
        risk_manifest_sha256=risk_hash,
    )
    atomic_torch_save(preflight_checkpoint, payload)
    restored = torch.load(preflight_checkpoint, map_location="cpu", weights_only=False)
    assert restored["experiment"] == EXPERIMENT_ID
    assert restored["branch"] == branch
    assert restored["local_step"] == 1
    preflight_checkpoint.unlink()
    print(
        f"SMOKE OK branch={branch} device={torch_directml.device_name(0)} "
        f"presence={values[0]:.6f} intensity={values[1]:.6f} total={values[2]:.6f} "
        "atomic_checkpoint=OK"
    )


def train(branch: str) -> None:
    import torch_directml

    if branch == "B" and not (RISK_CACHE.exists() and RISK_MANIFEST.exists()):
        raise FileNotFoundError("Prepare risk cache before training branch B")
    current_base_sha = sha256_file(BASE_CHECKPOINT)
    current_risk_hash = sha256_file(RISK_MANIFEST) if branch == "B" else None
    output = branch_dir(branch)
    output.mkdir(parents=True, exist_ok=True)
    final_path = output / f"branch_{branch}_final.pt"
    if final_path.exists():
        checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
        if int(checkpoint["local_step"]) == CONTINUATION_STEPS:
            print(f"Branch {branch} already complete: {final_path}")
            return
    acquire_process_lock(output / "training.lock", f"train_branch_{branch}")

    split = load_split(ROOT)
    sequences, _ = load_baseline_sequences(split)
    base = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=False)
    profile_config = ProfileWindowConfig(**base["profile_config"])
    generator = ProfileWindowGenerator(
        prepared_split=split, sequences=sequences, part="train", config=profile_config, seed=42
    )
    device = torch_directml.device()
    model = create_cnn_presence_intensity(channels=EXPECTED_CHANNELS)
    latest = output / "latest.pt"
    if latest.exists():
        resume = torch.load(latest, map_location="cpu", weights_only=False)
        if resume["branch"] != branch or resume["experiment"] != EXPERIMENT_ID:
            raise RuntimeError(f"Wrong resume checkpoint: {latest}")
        if resume["base_checkpoint_sha256"] != current_base_sha:
            raise RuntimeError("Base checkpoint changed since branch training started")
        if resume.get("risk_manifest_sha256") != current_risk_hash:
            raise RuntimeError("Risk manifest changed since branch training started")
        model.load_state_dict(resume["model_state_dict"])
        generator.rng.bit_generator.state = copy.deepcopy(resume["generator_rng_state"])
        torch.set_rng_state(resume["torch_rng_state"])
        start_step = int(resume["local_step"])
        history = list(resume["history"])
        seen = int(resume["seen_position_strand"])
        previous_elapsed = float(resume["elapsed_seconds"])
        print(f"Resume branch {branch} at local step {start_step}")
    else:
        model.load_state_dict(base["model_state_dict"])
        generator.rng.bit_generator.state = copy.deepcopy(base["generator_rng_state"])
        torch.set_rng_state(base["torch_rng_state"])
        start_step = 0
        history = []
        seen = 0
        previous_elapsed = 0.0
    model = model.to(device)
    optimizer = DirectMLAdam(model.parameters(), lr=START_LR)
    if latest.exists():
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        del resume
    else:
        optimizer.load_state_dict(base["optimizer_state_dict"])
    del base
    model.train()

    risk_cache = None
    risk_multipliers = None
    risk_hash = None
    if branch == "B":
        manifest = json.loads(RISK_MANIFEST.read_text(encoding="utf-8"))
        if manifest["source_checkpoint_sha256"] != current_base_sha:
            raise RuntimeError("Risk cache was built from a different base checkpoint")
        if tuple(manifest["shape"]) != (
            len(generator.tile_index), 2, profile_config.target_length
        ):
            raise RuntimeError("Risk cache shape does not match the training generator")
        if RISK_CACHE.stat().st_size != int(np.prod(manifest["shape"], dtype=np.int64)):
            raise RuntimeError("Risk cache file size is inconsistent with its manifest")
        if manifest.get("cache_sha256") != sha256_file(RISK_CACHE):
            raise RuntimeError("Risk cache checksum does not match its manifest")
        risk_hash = current_risk_hash
        risk_cache = np.memmap(
            RISK_CACHE, mode="r", dtype=np.uint8, shape=tuple(manifest["shape"])
        )
        risk_multipliers = np.asarray(
            manifest["candidate_negative_weight_multipliers"], dtype=np.float32
        )

    started = time.perf_counter()
    ema_presence = history[-1]["ema_presence"] if history else None
    ema_intensity = history[-1]["ema_intensity"] if history else None
    ema_total = history[-1]["ema_total"] if history else None
    fallback_messages: list[str] = []

    for local_step in range(start_step + 1, CONTINUATION_STEPS + 1):
        lr = planned_lr(local_step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = generator.sample_batch(BATCH_SIZE)
        plus_weight = batch.plus_loss_weight
        minus_weight_reverse = batch.minus_loss_weight_reverse
        if branch == "B":
            tile_ids = batch.tile_id.numpy().astype(np.int64, copy=False)
            strata = np.asarray(risk_cache[tile_ids])
            valid_negative = batch.mask.numpy() & ~(batch.target.numpy().astype(bool))
            if np.any(strata[valid_negative] == 255):
                raise AssertionError("Risk cache misses a sampled valid negative")
            multiplier = np.ones(strata.shape, dtype=np.float32)
            for stratum, value in enumerate(risk_multipliers):
                multiplier[strata == stratum] = value
            plus_weight = plus_weight * torch.from_numpy(multiplier[:, 0:1, :])
            minus_weight_reverse = minus_weight_reverse * torch.from_numpy(
                multiplier[:, 1:2, ::-1].copy()
            )

        gpu_phase_started = time.perf_counter()
        x_gpu = batch.x_both_strands.to(device)
        plus_target_gpu = batch.plus_target.to(device)
        minus_target_gpu = batch.minus_target_reverse.to(device)
        plus_weight_gpu = plus_weight.to(device)
        minus_weight_gpu = minus_weight_reverse.to(device)
        plus_count = batch.count[:, 0:1, :].contiguous()
        minus_count_reverse = batch.count[:, 1:2, :].flip(-1).contiguous()
        plus_intensity_target = torch.log1p(plus_count).to(device)
        minus_intensity_target = torch.log1p(minus_count_reverse).to(device)
        plus_intensity_weight = (
            2.0 * batch.plus_loss_weight * (plus_count > 0).float()
        ).to(device)
        minus_intensity_weight = (
            2.0 * batch.minus_loss_weight_reverse * (minus_count_reverse > 0).float()
        ).to(device)

        optimizer.zero_grad(set_to_none=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plus_presence, minus_presence, plus_intensity, minus_intensity = (
                forward_both_strands_multitask(
                    model, x_gpu, batch_size=BATCH_SIZE,
                    context_flank=profile_config.context_flank,
                    target_length=profile_config.target_length,
                )
            )
            presence_loss = dml_weighted_profile_bce_with_logits(
                plus_presence, minus_presence, plus_target_gpu, minus_target_gpu,
                plus_weight_gpu, minus_weight_gpu,
            )
            intensity_loss = (
                dml_weighted_huber(
                    plus_intensity, plus_intensity_target, plus_intensity_weight, delta=1.0
                )
                + dml_weighted_huber(
                    minus_intensity, minus_intensity_target, minus_intensity_weight, delta=1.0
                )
            )
            total_loss = presence_loss + INTENSITY_LOSS_WEIGHT * intensity_loss
            values = [float(value.detach().cpu().item()) for value in (presence_loss, intensity_loss, total_loss)]
            if not all(math.isfinite(value) for value in values):
                raise FloatingPointError(f"Non-finite loss at branch {branch} step {local_step}: {values}")
            total_loss.backward()
            optimizer.step()
            soft_gpu_duty_pause(time.perf_counter() - gpu_phase_started)
        new_fallback = [
            str(item.message) for item in caught if "fall back" in str(item.message).lower()
        ]
        if new_fallback:
            fallback_messages.extend(new_fallback)
            raise RuntimeError("DirectML CPU fallback detected: " + " | ".join(new_fallback))

        presence_value, intensity_value, total_value = values
        if ema_presence is None:
            ema_presence, ema_intensity, ema_total = values
        else:
            ema_presence = 0.99 * ema_presence + 0.01 * presence_value
            ema_intensity = 0.99 * ema_intensity + 0.01 * intensity_value
            ema_total = 0.99 * ema_total + 0.01 * total_value
        seen += int(batch.mask.sum())
        elapsed = previous_elapsed + time.perf_counter() - started
        if local_step % LOG_EVERY == 0 or local_step == CONTINUATION_STEPS:
            run_elapsed = max(elapsed - previous_elapsed, 1e-9)
            completed_now = local_step - start_step
            sec_per_step = run_elapsed / completed_now
            record = {
                "branch": branch,
                "local_step": local_step,
                "step": BASE_STEP + local_step,
                "learning_rate": lr,
                "presence_loss": presence_value,
                "intensity_loss": intensity_value,
                "total_loss": total_value,
                "ema_presence": float(ema_presence),
                "ema_intensity": float(ema_intensity),
                "ema_total": float(ema_total),
                "elapsed_seconds": elapsed,
                "seconds_per_step_current_process": sec_per_step,
                "estimated_hours_remaining": (CONTINUATION_STEPS - local_step) * sec_per_step / 3600,
                "seen_position_strand": seen,
            }
            history.append(record)
            atomic_csv(output / "training_history.csv", pd.DataFrame(history))
            print(
                f"[{branch}] local={local_step:05d}/{CONTINUATION_STEPS} global={BASE_STEP + local_step} "
                f"lr={lr:.7g} loss=({presence_value:.5f},{intensity_value:.5f},{total_value:.5f}) "
                f"sec/step={sec_per_step:.3f} ETA={record['estimated_hours_remaining']:.2f}h",
                flush=True,
            )
        if local_step % LATEST_EVERY == 0 or local_step == CONTINUATION_STEPS:
            payload = checkpoint_payload(
                branch=branch, local_step=local_step, model=model, optimizer=optimizer,
                generator=generator, history=history, elapsed_seconds=elapsed,
                seen_position_strand=seen, risk_manifest_sha256=risk_hash,
            )
            atomic_torch_save(latest, payload)
            if local_step in SNAPSHOT_STEPS:
                atomic_torch_save(output / f"model_local_step_{local_step:05d}.pt", payload)
            if local_step == CONTINUATION_STEPS:
                atomic_torch_save(final_path, payload)
            del payload
            gc.collect()

        del (
            batch, x_gpu, plus_target_gpu, minus_target_gpu, plus_weight_gpu, minus_weight_gpu,
            plus_count, minus_count_reverse, plus_intensity_target, minus_intensity_target,
            plus_intensity_weight, minus_intensity_weight, plus_presence, minus_presence,
            plus_intensity, minus_intensity, presence_loss, intensity_loss, total_loss,
        )

    summary = {
        "branch": branch,
        "complete": True,
        "local_steps": CONTINUATION_STEPS,
        "global_step": BASE_STEP + CONTINUATION_STEPS,
        "training_hours": elapsed / 3600,
        "final_lr": planned_lr(CONTINUATION_STEPS),
        "fallback_warnings": fallback_messages,
        "final_checkpoint": str(final_path),
    }
    atomic_json(output / "training_complete.json", summary)
    print(json.dumps(summary, indent=2))


def score_fixed_cohorts(model, generator, device, branch: str) -> pd.DataFrame:
    paths = {
        "HARD_FP": ORACLE_RUN / "fixed_hard_fp_coordinates.csv",
        "LOW_TP": ORACLE_RUN / "fixed_low_tp_coordinates.csv",
    }
    frames = []
    for cohort, path in paths.items():
        if path.is_file():
            frame = pd.read_csv(path)
            frame = frame[["contig", "strand", "position", "baseline_score"]].copy()
            frame["cohort"] = cohort
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    cohorts = pd.concat(frames, ignore_index=True)
    cohorts["target_start"] = cohorts["position"] // 1024 * 1024
    tile_map = {
        (str(row.contig), int(row.target_start)): index
        for index, row in generator.tile_index.iterrows()
    }
    cohorts["tile_index"] = [
        tile_map[(str(contig), int(start))]
        for contig, start in zip(cohorts["contig"], cohorts["target_start"])
    ]
    scores = np.full(len(cohorts), np.nan, dtype=np.float32)
    groups = cohorts.groupby("tile_index", sort=True).groups
    tile_groups = list(groups.items())
    cohort_batch_size = 4
    model.eval()
    with torch.no_grad():
        for start in range(0, len(tile_groups), cohort_batch_size):
            chunk = tile_groups[start : start + cohort_batch_size]
            tile_indices = [int(tile_index) for tile_index, _ in chunk]
            selected = generator.tile_index.iloc[tile_indices]
            batch = generator.make_batch(selected)
            x = batch.x_both_strands.to(device)
            plus, minus_reverse, _, _ = forward_both_strands_multitask(
                model, x, batch_size=len(chunk), context_flank=generator.config.context_flank,
                target_length=generator.config.target_length,
            )
            plus_probability = sigmoid_numpy(plus[:, 0].detach().cpu().numpy())
            minus_probability = sigmoid_numpy(minus_reverse[:, 0].detach().cpu().numpy()[:, ::-1])
            for local_batch, (_, row_ids) in enumerate(chunk):
                for row_id in row_ids:
                    row = cohorts.loc[row_id]
                    local = int(row["position"] - row["target_start"])
                    scores[row_id] = (
                        plus_probability[local_batch, local]
                        if row["strand"] == "+"
                        else minus_probability[local_batch, local]
                    )
            completed = min(start + len(chunk), len(tile_groups))
            if start == 0 or completed % 1000 < len(chunk) or completed == len(tile_groups):
                print(f"[{branch}] cohort tiles {completed:,}/{len(tile_groups):,}", flush=True)
            del batch, x, plus, minus_reverse
    if not np.all(np.isfinite(scores)):
        raise AssertionError("Non-finite cohort scores")
    cohorts[f"score_{branch}"] = scores
    return cohorts.drop(columns=["tile_index", "target_start"])


def evaluate(branch: str) -> None:
    import torch_directml

    output = branch_dir(branch)
    final_path = output / f"branch_{branch}_final.pt"
    if not final_path.is_file():
        raise FileNotFoundError(f"Final checkpoint missing: {final_path}")
    metrics_path = output / "validation_metrics_report.csv"
    if metrics_path.exists() and (output / "evaluation_complete.json").exists():
        print(f"Branch {branch} evaluation already complete: {metrics_path}")
        return
    acquire_process_lock(output / "evaluation.lock", f"evaluate_branch_{branch}")
    checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
    device = torch_directml.device()
    model = create_cnn_presence_intensity(channels=EXPECTED_CHANNELS)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    split = load_split(ROOT)
    sequences, _ = load_baseline_sequences(split)
    config = ProfileWindowConfig(**checkpoint["profile_config"])
    generator = ProfileWindowGenerator(
        prepared_split=split, sequences=sequences, part="validation", config=config, seed=42
    )
    result = evaluate_presence_intensity_model(
        model, generator, device=device, batch_size=1, score_decimals=5,
        progress_every_batches=100, fail_on_cpu_fallback=True,
    )
    atomic_csv(output / "validation_all_scores.csv", result.metrics)
    presence = result.metrics.loc[result.metrics["score"] == "presence_probability"].copy()
    primary = presence[["segment", "average_precision", "epic_spearman"]].copy()
    atomic_csv(metrics_path, primary)
    intensity = result.metrics.loc[result.metrics["score"] == "intensity_prediction", ["segment", "epic_spearman"]]
    intensity = intensity.rename(columns={"epic_spearman": "intensity_EPIC_Spearman_raw"})
    strong = presence[["segment", "positives", "average_precision", "epic_spearman", "quantized_spearman"]].rename(
        columns={
            "positives": "positive_n", "average_precision": "presence_AP_5dp",
            "epic_spearman": "presence_EPIC_Spearman_raw",
            "quantized_spearman": "presence_EPIC_Spearman_5dp",
        }
    ).merge(intensity, on="segment", validate="one_to_one")
    atomic_csv(output / "strong_validation_metrics.csv", strong)
    cohorts = score_fixed_cohorts(model, generator, device, branch)
    if not cohorts.empty:
        atomic_csv(output / "fixed_cohort_scores.csv", cohorts)
        score_column = f"score_{branch}"
        rows = []
        for cohort, frame in cohorts.groupby("cohort", sort=False):
            values = frame[score_column].to_numpy(np.float64)
            rows.append({
                "branch": branch, "cohort": cohort, "n": len(values),
                "mean": values.mean(), "p10": np.quantile(values, 0.1),
                "p50": np.quantile(values, 0.5), "p90": np.quantile(values, 0.9),
                "p99": np.quantile(values, 0.99),
            })
        atomic_csv(output / "fixed_cohort_score_summary.csv", pd.DataFrame(rows))
    evaluation = {
        "branch": branch, "complete": True,
        "elapsed_seconds": result.elapsed_seconds,
        "positions_per_second": result.positions_per_second,
        "processed_tiles": result.processed_tiles,
        "processed_positions": result.processed_positions,
        "processed_positives": result.processed_positives,
        "score_decimals": result.score_decimals,
        "fallback_warnings": result.fallback_warnings,
    }
    atomic_json(output / "evaluation_complete.json", evaluation)
    print(primary.to_string(index=False))


def _font(size: int, *, bold: bool = False):
    candidates = (
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _draw_vertical_label(image: Image.Image, text: str, font) -> None:
    probe = ImageDraw.Draw(image)
    box = probe.textbbox((0, 0), text, font=font)
    label = Image.new("RGBA", (box[2] - box[0] + 8, box[3] - box[1] + 8), (255, 255, 255, 0))
    ImageDraw.Draw(label).text((4, 4), text, fill="#202124", font=font, anchor="la")
    rotated = label.rotate(90, expand=True)
    image.paste(rotated, (12, (image.height - rotated.height) // 2), rotated)


def _save_line_plot(path: Path, series: list[tuple[str, np.ndarray, np.ndarray]], title: str, y_label: str) -> None:
    width, height = 1200, 620
    left, top, right, bottom = 105, 75, 35, 85
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, small_font = _font(27, bold=True), _font(18), _font(15)
    colors = ["#1967D2", "#D93025", "#188038", "#9334E6"]
    x_values = np.concatenate([np.asarray(item[1], dtype=np.float64) for item in series])
    y_values = np.concatenate([np.asarray(item[2], dtype=np.float64) for item in series])
    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    if math.isclose(x_min, x_max):
        x_max = x_min + 1.0
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0
    padding = 0.08 * (y_max - y_min)
    y_min, y_max = y_min - padding, y_max + padding
    plot_width, plot_height = width - left - right, height - top - bottom

    def transform(x, y):
        px = left + (x - x_min) / (x_max - x_min) * plot_width
        py = top + (y_max - y) / (y_max - y_min) * plot_height
        return float(px), float(py)

    draw.text((width / 2, 25), title, fill="#202124", font=title_font, anchor="ma")
    for tick in range(6):
        fraction = tick / 5
        py = top + fraction * plot_height
        value = y_max - fraction * (y_max - y_min)
        draw.line((left, py, width - right, py), fill="#E8EAED", width=1)
        draw.text((left - 12, py), f"{value:.4f}", fill="#5F6368", font=small_font, anchor="rm")
    draw.line((left, top, left, height - bottom), fill="#3C4043", width=2)
    draw.line((left, height - bottom, width - right, height - bottom), fill="#3C4043", width=2)
    for tick in range(5):
        fraction = tick / 4
        px = left + fraction * plot_width
        value = x_min + fraction * (x_max - x_min)
        draw.text((px, height - bottom + 12), f"{value:,.0f}", fill="#5F6368", font=small_font, anchor="ma")
    draw.text((width / 2, height - 28), "Continuation step", fill="#202124", font=label_font, anchor="ma")
    _draw_vertical_label(image, y_label, label_font)
    for index, (label, xs, ys) in enumerate(series):
        points = [transform(float(x), float(y)) for x, y in zip(xs, ys)]
        if len(points) > 1:
            draw.line(points, fill=colors[index % len(colors)], width=4, joint="curve")
        for point in points[:: max(1, len(points) // 20)]:
            x, y = point
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=colors[index % len(colors)])
        legend_x = left + index * 210
        draw.line((legend_x, 58, legend_x + 35, 58), fill=colors[index % len(colors)], width=4)
        draw.text((legend_x + 43, 58), label, fill="#202124", font=small_font, anchor="lm")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _save_grouped_bars(
    path: Path, categories: list[str], series: list[tuple[str, list[float]]], title: str, y_label: str
) -> None:
    width, height = 1200, 650
    left, top, right, bottom = 105, 80, 35, 125
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, small_font = _font(27, bold=True), _font(18), _font(15)
    colors = ["#9AA0A6", "#1967D2", "#D93025", "#188038"]
    maximum = max(float(value) for _, values in series for value in values)
    y_max = max(maximum * 1.15, 1e-6)
    plot_width, plot_height = width - left - right, height - top - bottom
    draw.text((width / 2, 28), title, fill="#202124", font=title_font, anchor="ma")
    for tick in range(6):
        fraction = tick / 5
        py = top + fraction * plot_height
        value = y_max * (1.0 - fraction)
        draw.line((left, py, width - right, py), fill="#E8EAED", width=1)
        draw.text((left - 12, py), f"{value:.3f}", fill="#5F6368", font=small_font, anchor="rm")
    draw.line((left, top, left, height - bottom), fill="#3C4043", width=2)
    draw.line((left, height - bottom, width - right, height - bottom), fill="#3C4043", width=2)
    group_width = plot_width / max(len(categories), 1)
    bar_area = group_width * 0.72
    bar_width = bar_area / max(len(series), 1)
    for category_index, category in enumerate(categories):
        center = left + (category_index + 0.5) * group_width
        draw.text((center, height - bottom + 18), category, fill="#202124", font=small_font, anchor="ma")
        for series_index, (_, values) in enumerate(series):
            value = float(values[category_index])
            x0 = center - bar_area / 2 + series_index * bar_width + 2
            x1 = x0 + bar_width - 4
            y0 = top + (y_max - value) / y_max * plot_height
            draw.rectangle((x0, y0, x1, height - bottom), fill=colors[series_index % len(colors)])
            draw.text(((x0 + x1) / 2, y0 - 5), f"{value:.4f}", fill="#202124", font=_font(12), anchor="ms")
    legend_x = left
    for index, (label, _) in enumerate(series):
        x = legend_x + index * 215
        draw.rectangle((x, 55, x + 24, 70), fill=colors[index % len(colors)])
        draw.text((x + 32, 63), label, fill="#202124", font=small_font, anchor="lm")
    _draw_vertical_label(image, y_label, label_font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _write_report_and_registry(
    *, decision: str, comparison: pd.DataFrame, overall: pd.Series, metrics_a: pd.DataFrame,
    metrics_b: pd.DataFrame, plot_dir: Path,
) -> tuple[Path, Path]:
    adopt = decision == "ADOPT_RISK_FOCUSED_LOSS"
    card_path = ROOT / "cnn_exp" / f"{EXPERIMENT_ID}.md"
    rows = []
    for row in comparison.itertuples(index=False):
        rows.append(
            f"| {row.segment} | {row.ap_EXP045:.9f} | {row.ap_A:.9f} | "
            f"{row.ap_B:.9f} | {row.delta_B_vs_A:+.9f} |"
        )
    report = f"""## Итоговое сравнение

| Segment | EXP045 AP | A control AP | B risk-focused AP | B−A |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

- Pre-registered decision: **{decision}**.
- B−A overall AP: `{float(overall['delta_B_vs_A']):+.9f}`.
- B−EXP045 overall AP: `{float(overall['delta_B_vs_EXP045']):+.9f}`.
- AP 0.20 reached: `{float(overall['ap_B']) >= 0.20}`.
- Risk-loss selection rule satisfied: `{adopt}`.

### Графики

![[{plot_dir.relative_to(ROOT).as_posix()}/01_training_losses.png]]

![[{plot_dir.relative_to(ROOT).as_posix()}/02_overall_comparison.png]]

![[{plot_dir.relative_to(ROOT).as_posix()}/03_per_contig_comparison.png]]

![[{plot_dir.relative_to(ROOT).as_posix()}/04_paired_training_losses.png]]

![[{plot_dir.relative_to(ROOT).as_posix()}/05_exp045_vs_A_vs_B_ap.png]]

### Вывод

Ветка B сравнивается с matched continuation A, поэтому дополнительное обучение отделено от эффекта risk-focused loss. Validation остаётся exploratory: этот split просматривался в предыдущих экспериментах.
"""
    text = card_path.read_text(encoding="utf-8")
    start_marker = "<!-- auto:cnn-experiment-report:start -->"
    end_marker = "<!-- auto:cnn-experiment-report:end -->"
    if start_marker not in text or end_marker not in text:
        raise ValueError(f"Auto-report markers missing in {card_path}")
    before, remainder = text.split(start_marker, 1)
    _, after = remainder.split(end_marker, 1)
    text = before + start_marker + "\n\n" + report + "\n" + end_marker + after
    text = re.sub(r"(?m)^status: .+$", "status: completed", text, count=1)
    text = re.sub(r"(?m)^decision: .+$", f"decision: {'adopt' if adopt else 'reject'}", text, count=1)
    card_path.write_text(text, encoding="utf-8", newline="\n")

    summary_path = RUN_DIR / "experiment_summary.json"
    atomic_json(summary_path, {
        "experiment_id": EXPERIMENT_ID,
        "title": "paired WIDTH512 risk-focused loss continuation",
        "hypothesis": "Risk-focused end-to-end presence training improves natural nucleotide AP over a matched continuation",
        "reference_experiment": "CNN-EXP-047-A",
        "decision": "adopt" if adopt else "reject",
        "date": date.today().isoformat(),
        "notebook": NOTEBOOK_REL,
        "reference_ap": float(overall["ap_A"]),
        "candidate_ap": float(overall["ap_B"]),
        "delta_ap": float(overall["delta_B_vs_A"]),
        "relative_ap": float(overall["ap_B"] / overall["ap_A"] - 1.0),
        "reference_spearman": float(overall["spearman_A"]),
        "candidate_spearman": float(overall["spearman_B"]),
        "delta_spearman": float(overall["spearman_B"] - overall["spearman_A"]),
    })

    index_path = ROOT / "cnn_exp" / "_index.md"
    lines = index_path.read_text(encoding="utf-8").splitlines()
    row = (
        f"| [[cnn_exp/{EXPERIMENT_ID}.md\\|{EXPERIMENT_ID}]] | paired WIDTH512 risk-focused loss continuation | "
        f"{float(overall['ap_B']):.9f} | {float(overall['spearman_B']):.9f} | "
        f"`{'adopt' if adopt else 'reject'}` |"
    )
    prefix = f"| [[cnn_exp/{EXPERIMENT_ID}.md"
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = row
            break
    else:
        table_rows = [index for index, line in enumerate(lines) if line.startswith("| [[cnn_exp/CNN-EXP-")]
        if not table_rows:
            raise ValueError("CNN registry table was not found")
        lines.insert(table_rows[-1] + 1, row)
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return card_path, summary_path


def finalize() -> None:
    preregister()
    candidate_path = branch_dir("B") / "validation_metrics_report.csv"
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Evaluate candidate B first: {candidate_path}")
    if not EXP047_B_METRICS.is_file():
        raise FileNotFoundError(f"Historical EXP047-B metrics missing: {EXP047_B_METRICS}")

    candidate = pd.read_csv(candidate_path).rename(
        columns={"average_precision": "ap_EXP048", "epic_spearman": "spearman_EXP048"}
    )
    reference = pd.read_csv(EXP047_B_METRICS).rename(
        columns={"average_precision": "ap_EXP047B", "epic_spearman": "spearman_EXP047B"}
    )
    comparison = reference.merge(candidate, on="segment", validate="one_to_one")
    comparison["delta_ap"] = comparison["ap_EXP048"] - comparison["ap_EXP047B"]
    comparison["delta_spearman"] = comparison["spearman_EXP048"] - comparison["spearman_EXP047B"]
    atomic_csv(RUN_DIR / "EXP048_vs_EXP047B.csv", comparison)

    strong_src = branch_dir("B") / "strong_validation_metrics.csv"
    if strong_src.is_file():
        shutil.copy2(strong_src, RUN_DIR / "strong_validation_metrics.csv")
    hist_src = branch_dir("B") / "training_history.csv"
    if hist_src.is_file():
        shutil.copy2(hist_src, RUN_DIR / "training_history.csv")

    overall = comparison.loc[comparison["segment"] == "overall"].iloc[0]
    per_contig = comparison.loc[comparison["segment"] != "overall"]
    quantitative_gate = (
        float(overall["delta_ap"]) >= 0.010
        and bool((per_contig["delta_ap"] > 0).all())
        and float(overall["delta_spearman"]) >= -0.003
    )

    cohort_comparison_path = None
    candidate_cohort = branch_dir("B") / "fixed_cohort_scores.csv"
    if EXP047_B_COHORT.is_file() and candidate_cohort.is_file():
        left = pd.read_csv(EXP047_B_COHORT).rename(columns={"score_B": "score_EXP047B"})
        right = pd.read_csv(candidate_cohort).rename(columns={"score_B": "score_EXP048"})
        keys = ["contig", "strand", "position", "baseline_score", "cohort"]
        cohort = left[keys + ["score_EXP047B"]].merge(
            right[keys + ["score_EXP048"]], on=keys, validate="one_to_one"
        )
        cohort["delta_EXP048_vs_EXP047B"] = cohort["score_EXP048"] - cohort["score_EXP047B"]
        cohort_comparison_path = RUN_DIR / "fixed_cohort_EXP047B_vs_EXP048.csv"
        atomic_csv(cohort_comparison_path, cohort)
        summary = cohort.groupby("cohort", as_index=False).agg(
            n=("delta_EXP048_vs_EXP047B", "size"),
            mean_score_EXP047B=("score_EXP047B", "mean"),
            mean_score_EXP048=("score_EXP048", "mean"),
            mean_delta=("delta_EXP048_vs_EXP047B", "mean"),
        )
        atomic_csv(RUN_DIR / "fixed_cohort_EXP047B_vs_EXP048_summary.csv", summary)

    result = {
        "experiment_id": EXPERIMENT_ID,
        "operational_reference": "EXP047-B",
        "quantitative_adopt_gate_passed": bool(quantitative_gate),
        "final_adopt_pending_raw_ap_and_budget_recall_audit": True,
        "overall": {key: float(overall[key]) for key in comparison.columns if key != "segment"},
        "per_contig": comparison.loc[comparison["segment"] != "overall"].to_dict(orient="records"),
        "gpu_soft_duty_cycle": GPU_DUTY_CYCLE,
        "files": {
            "preregistration": str(PREREG_PATH),
            "risk_manifest": str(RISK_MANIFEST),
            "candidate_checkpoint": str(branch_dir("B") / "branch_B_final.pt"),
            "comparison": str(RUN_DIR / "EXP048_vs_EXP047B.csv"),
            "cohort_comparison": str(cohort_comparison_path) if cohort_comparison_path else None,
        },
        "caveat": (
            "Raw AP and equal-budget LOW_TP/HARD_FP recall are mandatory before final ADOPT_ULTRA_TAIL. "
            "The inherited EXP047 evaluator persists AP5 and cohort scores but not exact raw AP."
        ),
    }
    atomic_json(RESULT_PATH, result)
    print(comparison.to_string(index=False))
    print(json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preregister")
    subparsers.add_parser("inspect")
    subparsers.add_parser("prepare-risk-cache")
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--branch", required=True, choices=["A", "B"])
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--branch", required=True, choices=["A", "B"])
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--branch", required=True, choices=["A", "B"])
    subparsers.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preregister":
        preregister()
    elif args.command == "inspect":
        inspect_inputs()
    elif args.command == "prepare-risk-cache":
        prepare_risk_cache()
    elif args.command == "smoke":
        smoke(args.branch)
    elif args.command == "train":
        train(args.branch)
    elif args.command == "evaluate":
        evaluate(args.branch)
    elif args.command == "finalize":
        finalize()
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
