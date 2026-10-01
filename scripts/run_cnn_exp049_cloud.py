"""CUDA/CPU runner for the CNN-EXP-049 coarse-to-fine cascade.

The runner is intentionally split into restartable stages so that Colab or
Kaggle session limits do not invalidate completed work:

  inspect -> smoke -> train-a/cache-a (one fold at a time) -> evaluate-a
          -> train-b -> final train-a/cache-a -> evaluate-b

For leakage-safe Network-B training, run every Network-A fold and build one
shared OOF cache containing only predictions for that fold's held-out contigs.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml_project.epic_cnn import (
    ConditionalLocalizerB,
    ProfileWindowConfig,
    ProfileWindowGenerator,
    RegionalDetectorA,
    RegionalFeatureCache,
    RegionalWindowGenerator,
    encode_raw_dna_six_channel,
    predict_contig_to_regional_cache,
    reverse_complement_six_channel,
)
from ml_project.epic_data import ensure_prepared_split
from ml_project.sequence_context import read_fasta


EXPERIMENT_ID = "CNN-EXP-049"
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "experiments"
    / "CNN_EXP049_R16_COARSE_TO_FINE_CASCADE"
    / "run_001"
)


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def contract_sha256(contract: Mapping[str, object]) -> str:
    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_matching_contract(checkpoint: Mapping[str, object], expected: Mapping[str, object]) -> None:
    actual = checkpoint.get("training_contract")
    if actual != expected:
        raise ValueError(
            "checkpoint training contract differs from this command; start a new "
            "run-dir or restore the original arguments"
        )


def require_completed_checkpoint(checkpoint: Mapping[str, object], name: str) -> None:
    step = int(checkpoint["step"])
    total_steps = int(checkpoint["total_steps"])
    if step != total_steps:
        raise RuntimeError(
            f"{name} checkpoint is incomplete ({step}/{total_steps}); rerun its "
            "training stage to resume before caching or evaluation"
        )


def device_from_argument(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def amp_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def gradient_scaler(device: torch.device):
    """Use loss scaling only for CUDA fp16; bf16 keeps the fp32 exponent range."""

    enabled = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    return torch.amp.GradScaler("cuda", enabled=enabled)


def cuda_description() -> dict[str, object]:
    result: dict[str, object] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        result.update(
            {
                "gpu": properties.name,
                "gpu_memory_gib": properties.total_memory / 2**30,
                "bf16": torch.cuda.is_bf16_supported(),
            }
        )
    return result


def load_project_data():
    split = ensure_prepared_split(PROJECT_ROOT)
    sequences, _alphabet = read_fasta(split.paths.source / "genome/genome.fa.gz")
    required = set(split.contig_assignment["contig"].astype(str))
    missing = sorted(required - set(sequences))
    if missing:
        raise ValueError("split contigs missing from FASTA: " + ", ".join(missing))
    return split, sequences


def train_contigs(split) -> list[str]:
    return sorted(set(split.intervals("train")["contig"].astype(str)))


def validation_contigs(split) -> list[str]:
    return sorted(set(split.intervals("validation")["contig"].astype(str)))


def fold_assignment(contigs: Iterable[str], fold_count: int, seed: int) -> list[list[str]]:
    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    values = np.asarray(sorted(set(contigs)), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    folds = [sorted(map(str, values[index::fold_count])) for index in range(fold_count)]
    if any(not fold for fold in folds):
        raise ValueError("fold_count exceeds the number of train contigs")
    return folds


def a_checkpoint_path(run_dir: Path, fold: int) -> Path:
    name = "all_train" if fold < 0 else f"fold_{fold:02d}"
    return run_dir / "network_a" / name / "latest.pt"


def build_a() -> RegionalDetectorA:
    return RegionalDetectorA()


def load_a_checkpoint(path: Path, device: torch.device) -> tuple[RegionalDetectorA, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_a()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device), checkpoint


def build_b(film_blocks: tuple[int, ...]) -> ConditionalLocalizerB:
    return ConditionalLocalizerB(film_after_blocks=film_blocks)


def parse_film_blocks(value: str) -> tuple[int, ...]:
    value = value.strip().lower()
    if value in {"", "none"}:
        return ()
    result = tuple(int(item) for item in value.split(","))
    if len(set(result)) != len(result):
        raise ValueError("FiLM block list contains duplicates")
    return result


def validate_oof_cache(
    cache: RegionalFeatureCache,
    split,
    *,
    fold_count: int,
    fold_seed: int,
) -> None:
    """Reject missing, in-sample, or fold-mismatched Network-A train features."""

    expected_fold = {
        contig: fold
        for fold, contigs in enumerate(
            fold_assignment(train_contigs(split), fold_count, fold_seed)
        )
        for contig in contigs
    }
    errors: list[str] = []
    contract_hashes_by_fold: dict[int, set[str]] = {}
    records = cache.manifest.get("contigs", {})
    for contig, fold in expected_fold.items():
        record = records.get(contig)
        if record is None:
            errors.append(f"{contig}: missing")
            continue
        source = record.get("source", {})
        if source.get("oof") is not True:
            errors.append(f"{contig}: source is not marked oof=true")
        try:
            source_fold = int(source.get("fold"))
        except (TypeError, ValueError):
            source_fold = None
        if source_fold != fold:
            errors.append(
                f"{contig}: source fold={source_fold!r}, expected held-out fold={fold}"
            )
        source_contract = source.get("training_contract_sha256")
        if not isinstance(source_contract, str) or len(source_contract) != 64:
            errors.append(f"{contig}: missing Network-A training contract hash")
        else:
            contract_hashes_by_fold.setdefault(fold, set()).add(source_contract)
    for fold, hashes in contract_hashes_by_fold.items():
        if len(hashes) != 1:
            errors.append(f"fold {fold}: cache mixes {len(hashes)} A training contracts")
    if errors:
        preview = "; ".join(errors[:8])
        suffix = "" if len(errors) <= 8 else f"; ... ({len(errors)} errors total)"
        raise RuntimeError(f"invalid Network-A OOF cache: {preview}{suffix}")


def stage_inspect(args) -> None:
    split, sequences = load_project_data()
    payload = {
        "experiment": EXPERIMENT_ID,
        "project_root": str(PROJECT_ROOT),
        "run_dir": str(args.run_dir),
        "runtime": cuda_description(),
        "train_contigs": train_contigs(split),
        "validation_contigs": validation_contigs(split),
        "sequence_count": len(sequences),
        "folds": fold_assignment(train_contigs(split), args.folds, args.fold_seed),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    atomic_json(args.run_dir / "cloud_inspect.json", payload)


def stage_smoke(args) -> None:
    device = device_from_argument(args.device)
    print(json.dumps(cuda_description(), indent=2))
    torch.manual_seed(args.seed)

    a = build_a().to(device).train()
    raw_a = torch.randint(0, 9, (1, 8192), dtype=torch.uint8, device=device)
    with amp_context(device):
        activity, latent = a(encode_raw_dna_six_channel(raw_a))
        loss_a = activity.float().square().mean() + 0.001 * latent.float().square().mean()
    loss_a.backward()
    print("A", tuple(activity.shape), tuple(latent.shape), float(loss_a))
    del a, raw_a, activity, latent, loss_a
    if device.type == "cuda":
        torch.cuda.empty_cache()

    b = build_b(parse_film_blocks(args.film_blocks)).to(device).train()
    raw_b = torch.randint(0, 9, (1, 2052), dtype=torch.uint8, device=device)
    dna_b = encode_raw_dna_six_channel(raw_b)
    condition = torch.randn(1, 17, 2052, device=device)
    with amp_context(device):
        output = b(dna_b, condition)
        loss_b = output["final_logit"].float().square().mean()
    loss_b.backward()
    print(
        "B",
        {key: tuple(value.shape) for key, value in output.items()},
        "RF=", b.receptive_field,
        "loss=", float(loss_b),
    )


def masked_regional_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: float,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(
        logits.float(),
        target.float(),
        pos_weight=torch.tensor(pos_weight, device=logits.device),
        reduction="none",
    )
    weight = mask.float()
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def stage_train_a(args) -> None:
    device = device_from_argument(args.device)
    split, sequences = load_project_data()
    folds = fold_assignment(train_contigs(split), args.folds, args.fold_seed)
    if args.fold < -1 or args.fold >= args.folds:
        raise ValueError("--fold must be -1 (all train) or a valid fold index")
    held_out = set() if args.fold < 0 else set(folds[args.fold])
    fit_contigs = [name for name in train_contigs(split) if name not in held_out]
    generator = RegionalWindowGenerator(
        split,
        sequences,
        part="train",
        contigs=fit_contigs,
        window_bp=8192,
        positive_window_fraction=args.a_positive_fraction,
        seed=args.seed + max(args.fold, 0),
    )
    training_contract = {
        "experiment_id": EXPERIMENT_ID,
        "network": "A",
        "fold": args.fold,
        "fold_count": args.folds,
        "fold_seed": args.fold_seed,
        "seed": args.seed,
        "fit_contigs": fit_contigs,
        "held_out_contigs": sorted(held_out),
        "total_steps": args.a_steps,
        "batch_size": args.a_batch,
        "learning_rate": args.a_lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "positive_weight": args.a_pos_weight,
        "positive_window_fraction": args.a_positive_fraction,
        "input_bp": 8192,
        "region_size": 16,
        "condition_channels": 17,
    }
    print(
        f"A fold={args.fold} fit={fit_contigs} held_out={sorted(held_out)} "
        f"allowed_R16={generator.allowed_region_count:,} "
        f"active_R16={generator.active_region_count:,}"
    )

    checkpoint_path = a_checkpoint_path(args.run_dir, args.fold)
    model = build_a().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.a_lr, weight_decay=args.weight_decay
    )
    scaler = gradient_scaler(device)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.a_lr,
        total_steps=args.a_steps,
        pct_start=0.05,
    )
    start_step = 0
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        require_matching_contract(checkpoint, training_contract)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_step = int(checkpoint["step"])
        print("Resuming A from", checkpoint_path, "step", start_step)

    started = time.monotonic()
    model.train()
    for step in range(start_step + 1, args.a_steps + 1):
        batch = generator.sample_batch(args.a_batch)
        dna = generator.encode_oriented(batch, device)
        target = batch.target.to(device, non_blocking=True)
        mask = batch.mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device):
            activity, _ = model(dna)
            loss = masked_regional_bce(activity, target, mask, args.a_pos_weight)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step == 1 or step % args.log_every == 0:
            with torch.no_grad():
                prediction = torch.sigmoid(activity.float()) >= 0.5
                positive = target.bool() & mask
                recall = float((prediction & positive).sum() / positive.sum().clamp_min(1))
            elapsed = time.monotonic() - started
            print(
                f"A fold={args.fold} step={step}/{args.a_steps} "
                f"loss={float(loss):.6f} batch_recall@0.5={recall:.5f} "
                f"lr={scheduler.get_last_lr()[0]:.3e} elapsed={elapsed/60:.1f}m",
                flush=True,
            )

        should_save = step % args.save_every == 0 or step == args.a_steps
        time_limit = args.max_minutes > 0 and (time.monotonic() - started) >= 60 * args.max_minutes
        if should_save or time_limit:
            atomic_torch_save(
                checkpoint_path,
                {
                    "experiment_id": EXPERIMENT_ID,
                    "network": "A",
                    "fold": args.fold,
                    "fold_count": args.folds,
                    "fold_seed": args.fold_seed,
                    "held_out_contigs": sorted(held_out),
                    "fit_contigs": fit_contigs,
                    "step": step,
                    "total_steps": args.a_steps,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "model_config": {
                        "input_bp": 8192,
                        "region_size": 16,
                        "condition_channels": model.condition_channels,
                    },
                    "training_contract": training_contract,
                },
            )
            print("Saved", checkpoint_path, flush=True)
        if time_limit:
            print("Reached --max-minutes; checkpoint saved for resume.", flush=True)
            break


def stage_cache_a(args) -> None:
    device = device_from_argument(args.device)
    split, sequences = load_project_data()
    checkpoint_path = a_checkpoint_path(args.run_dir, args.fold)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    model, checkpoint = load_a_checkpoint(checkpoint_path, device)
    require_completed_checkpoint(checkpoint, "Network A")
    training_contract = checkpoint.get("training_contract")
    if not isinstance(training_contract, Mapping):
        raise RuntimeError("Network A checkpoint has no reproducible training contract")
    cache_command_contract = {
        "fold": args.fold,
        "fold_count": args.folds,
        "fold_seed": args.fold_seed,
    }
    for key, expected in cache_command_contract.items():
        if training_contract.get(key) != expected:
            raise ValueError(
                f"cache command {key}={expected!r} does not match Network A "
                f"checkpoint value {training_contract.get(key)!r}"
            )
    folds = fold_assignment(train_contigs(split), args.folds, args.fold_seed)

    if args.cache_part == "train":
        if args.fold < 0 and not args.allow_in_sample_cache:
            raise RuntimeError(
                "Refusing in-sample Network-A cache for B training. Run each OOF fold."
            )
        contigs = train_contigs(split) if args.fold < 0 else folds[args.fold]
    elif args.cache_part == "validation":
        if args.fold != -1:
            raise RuntimeError("validation cache requires Network A trained on all train contigs (--fold -1)")
        contigs = validation_contigs(split)
    else:
        raise ValueError(args.cache_part)

    cache = RegionalFeatureCache.create(
        args.cache_dir,
        channels=model.condition_channels,
        metadata={"experiment_id": EXPERIMENT_ID, "part": args.cache_part},
    )
    expected_source = {
        "fold": args.fold,
        "checkpoint_step": int(checkpoint["step"]),
        "oof": args.cache_part == "train" and args.fold >= 0,
        "training_contract_sha256": contract_sha256(training_contract),
    }
    for index, contig in enumerate(contigs, start=1):
        if cache.has_contig(contig) and not args.overwrite_cache_contig:
            cached_source = cache.manifest["contigs"][contig].get("source", {})
            mismatched = {
                key: (cached_source.get(key), expected)
                for key, expected in expected_source.items()
                if cached_source.get(key) != expected
            }
            if mismatched:
                raise RuntimeError(
                    f"cached {contig} has stale/incompatible source metadata "
                    f"{mismatched}; pass --overwrite-cache-contig after verifying "
                    "the intended checkpoint"
                )
            print(f"[{index}/{len(contigs)}] skip cached {contig}")
            continue
        print(f"[{index}/{len(contigs)}] predict {contig}", flush=True)
        predict_contig_to_regional_cache(
            model,
            sequences[contig],
            contig=contig,
            cache=cache,
            device=device,
            source={**expected_source, "checkpoint": str(checkpoint_path)},
        )
    cache.close()
    print("Cache ready:", args.cache_dir)


def stage_evaluate_a(args) -> None:
    """Evaluate the learned OOF regional detector without touching validation."""

    split, sequences = load_project_data()
    cache = RegionalFeatureCache(args.cache_dir)
    validate_oof_cache(
        cache,
        split,
        fold_count=args.folds,
        fold_seed=args.fold_seed,
    )
    generator = RegionalWindowGenerator(
        split,
        sequences,
        part="train",
        contigs=train_contigs(split),
        window_bp=8192,
        positive_window_fraction=0.5,
        seed=args.seed,
    )
    total_hist = np.zeros(100_001, dtype=np.int64)
    active_hist = np.zeros(100_001, dtype=np.int64)
    active_logits: list[np.ndarray] = []
    empty_logits: list[np.ndarray] = []
    for (contig, strand), allowed in generator.allowed.items():
        logit = np.asarray(cache._array(contig, strand)[:, 0], dtype=np.float32)
        active = generator.active[(contig, strand)] & allowed
        empty = (~generator.active[(contig, strand)]) & allowed
        probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
        score = np.rint(probability * 100_000).astype(np.int32)
        total_hist += np.bincount(score[allowed], minlength=100_001)
        active_hist += np.bincount(score[active], minlength=100_001)
        active_logits.append(logit[active].copy())
        empty_logits.append(logit[empty].copy())

    positives = int(active_hist.sum())
    pos_desc = active_hist[::-1]
    total_desc = total_hist[::-1]
    precision = np.cumsum(pos_desc) / np.maximum(np.cumsum(total_desc), 1)
    regional_ap = float(np.sum((pos_desc / max(positives, 1)) * precision))
    active_score = np.concatenate(active_logits)
    empty_score = np.concatenate(empty_logits)
    threshold_995 = float(np.quantile(active_score, 0.005, method="lower"))
    result = {
        "experiment_id": EXPERIMENT_ID,
        "evaluation": "train-contig OOF Network-A regional gate",
        "allowed_regions": int(total_hist.sum()),
        "active_regions": positives,
        "active_prevalence": positives / max(int(total_hist.sum()), 1),
        "regional_ap5": regional_ap,
        "threshold_for_0p995_active_recall": threshold_995,
        "realized_active_recall": float(np.mean(active_score >= threshold_995)),
        "empty_region_suppression_fraction": float(np.mean(empty_score < threshold_995)),
        "cache": str(args.cache_dir),
        "note": "This gate measures learnability, not downstream nucleotide AP.",
    }
    print(json.dumps(result, indent=2))
    atomic_json(args.run_dir / "network_a" / "oof_regional_metrics.json", result)
    cache.close()


def weighted_profile_bce(
    plus_logit: torch.Tensor,
    minus_logit: torch.Tensor,
    plus_target: torch.Tensor,
    minus_target: torch.Tensor,
    plus_weight: torch.Tensor,
    minus_weight: torch.Tensor,
) -> torch.Tensor:
    plus = F.binary_cross_entropy_with_logits(
        plus_logit.float(), plus_target.float(), reduction="none"
    )
    minus = F.binary_cross_entropy_with_logits(
        minus_logit.float(), minus_target.float(), reduction="none"
    )
    return (plus * plus_weight.float()).sum() + (minus * minus_weight.float()).sum()


def within_r16_pairwise_loss(logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logit.shape[-1] % 16:
        raise ValueError("within-R16 loss requires a target length divisible by 16")
    shape = (*logit.shape[:-1], logit.shape[-1] // 16, 16)
    z = logit.reshape(shape).float()
    y = target.reshape(shape).bool()
    valid = mask.reshape(shape).bool()
    positive = y & valid
    negative = (~y) & valid
    pair_mask = positive.unsqueeze(-1) & negative.unsqueeze(-2)
    if not bool(pair_mask.any()):
        return z.sum() * 0.0
    difference = z.unsqueeze(-2) - z.unsqueeze(-1)
    # Axes are [positive_index, negative_index]: softplus(z_neg - z_pos).
    return F.softplus(difference)[pair_mask].mean()


def stage_train_b(args) -> None:
    device = device_from_argument(args.device)
    split, sequences = load_project_data()
    cache = RegionalFeatureCache(args.cache_dir)
    validate_oof_cache(
        cache,
        split,
        fold_count=args.folds,
        fold_seed=args.fold_seed,
    )

    profile_config = ProfileWindowConfig(
        target_length=1024,
        context_flank=514,
        positive_branch_fraction=0.5,
    )
    generator = ProfileWindowGenerator(
        split,
        sequences,
        part="train",
        config=profile_config,
        seed=args.seed,
    )
    risk_cache = None
    risk_multipliers = None
    risk_manifest_payload = None
    if args.risk_cache is not None:
        risk_manifest_path = (
            args.risk_manifest
            if args.risk_manifest is not None
            else args.risk_cache.with_name("risk_strata_manifest.json")
        )
        if not args.risk_cache.is_file() or not risk_manifest_path.is_file():
            raise FileNotFoundError(
                f"risk cache/manifest not found: {args.risk_cache}, {risk_manifest_path}"
            )
        risk_manifest_payload = json.loads(risk_manifest_path.read_text(encoding="utf-8"))
        expected_shape = tuple(risk_manifest_payload["shape"])
        actual_shape = (len(generator.tile_index), 2, profile_config.target_length)
        if expected_shape != actual_shape:
            raise ValueError(
                f"EXP047 risk cache shape {expected_shape} does not match {actual_shape}"
            )
        risk_cache = np.memmap(
            args.risk_cache,
            mode="r",
            dtype=np.uint8,
            shape=expected_shape,
        )
        risk_multipliers = np.asarray(
            risk_manifest_payload["candidate_negative_weight_multipliers"],
            dtype=np.float32,
        )
        print("Using exact EXP047 risk-strata weights from", args.risk_cache)
    else:
        print("WARNING: EXP047 risk-strata cache is absent; using base natural-profile BCE")
    film_blocks = parse_film_blocks(args.film_blocks)
    cache_sources = {
        contig: {
            "fold": int(cache.manifest["contigs"][contig]["source"]["fold"]),
            "checkpoint_step": int(
                cache.manifest["contigs"][contig]["source"]["checkpoint_step"]
            ),
            "training_contract_sha256": cache.manifest["contigs"][contig]["source"][
                "training_contract_sha256"
            ],
        }
        for contig in train_contigs(split)
    }
    training_contract = {
        "experiment_id": EXPERIMENT_ID,
        "network": "B",
        "fold_count": args.folds,
        "fold_seed": args.fold_seed,
        "seed": args.seed,
        "total_steps": args.b_steps,
        "batch_size": args.b_batch,
        "learning_rate": args.b_lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "film_blocks": list(film_blocks),
        "lambda_within": args.lambda_within,
        "lambda_count": args.lambda_count,
        "lambda_residual": args.lambda_residual,
        "profile_target_length": profile_config.target_length,
        "profile_context_flank": profile_config.context_flank,
        "positive_branch_fraction": profile_config.positive_branch_fraction,
        "risk_cache_sha256": (
            risk_manifest_payload.get("cache_sha256")
            if risk_manifest_payload is not None
            else None
        ),
        "regional_cache_sources": cache_sources,
    }
    model = build_b(film_blocks).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.b_lr, weight_decay=args.weight_decay
    )
    scaler = gradient_scaler(device)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.b_lr,
        total_steps=args.b_steps,
        pct_start=0.05,
    )
    checkpoint_path = args.run_dir / "network_b" / f"film_{args.film_blocks.replace(',', '-')}.pt"
    start_step = 0
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        require_matching_contract(checkpoint, training_contract)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_step = int(checkpoint["step"])
        print("Resuming B from", checkpoint_path, "step", start_step)

    started = time.monotonic()
    model.train()
    flank = profile_config.context_flank
    crop = slice(flank, flank + profile_config.target_length)
    for step in range(start_step + 1, args.b_steps + 1):
        batch = generator.sample_batch(args.b_batch)
        raw = batch.raw_codes.to(device, non_blocking=True)
        plus_dna = encode_raw_dna_six_channel(raw)
        dna = torch.cat((plus_dna, reverse_complement_six_channel(plus_dna)), dim=0)
        condition = cache.condition_for_profile_tiles(
            batch.tiles,
            context_flank=flank,
            target_length=profile_config.target_length,
        ).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device):
            output = model(dna, condition)
            final = output["final_logit"][..., crop]
            intensity = output["intensity"][..., crop]
            delta = output["delta_logit"][..., crop]
            plus_logit = final[: args.b_batch]
            minus_logit = final[args.b_batch :]
            plus_intensity = intensity[: args.b_batch]
            minus_intensity = intensity[args.b_batch :]

            plus_target = batch.plus_target.to(device)
            minus_target = batch.minus_target_reverse.to(device)
            plus_weight_cpu = batch.plus_loss_weight
            minus_weight_cpu = batch.minus_loss_weight_reverse
            if risk_cache is not None:
                tile_ids = batch.tile_id.numpy().astype(np.int64, copy=False)
                strata = np.asarray(risk_cache[tile_ids])
                valid_negative = batch.mask.numpy() & ~batch.target.numpy().astype(bool)
                if np.any(strata[valid_negative] == 255):
                    raise AssertionError("valid negative is missing from EXP047 risk cache")
                multiplier = np.ones(strata.shape, dtype=np.float32)
                for stratum, value in enumerate(risk_multipliers):
                    multiplier[strata == stratum] = value
                plus_weight_cpu = plus_weight_cpu * torch.from_numpy(multiplier[:, 0:1, :])
                minus_weight_cpu = minus_weight_cpu * torch.from_numpy(
                    multiplier[:, 1:2, ::-1].copy()
                )
            plus_weight = plus_weight_cpu.to(device)
            minus_weight = minus_weight_cpu.to(device)
            presence_loss = weighted_profile_bce(
                plus_logit,
                minus_logit,
                plus_target,
                minus_target,
                plus_weight,
                minus_weight,
            )
            both_logit = torch.cat((plus_logit, minus_logit), dim=0)
            both_target = torch.cat((plus_target, minus_target), dim=0)
            both_mask = torch.cat(
                (batch.mask[:, 0:1, :], batch.mask[:, 1:2, :].flip(-1)), dim=0
            ).to(device)
            within_loss = within_r16_pairwise_loss(both_logit, both_target, both_mask)

            plus_count = batch.count[:, 0:1, :].to(device)
            minus_count = batch.count[:, 1:2, :].flip(-1).to(device)
            both_count = torch.cat((plus_count, minus_count), dim=0)
            both_intensity = torch.cat((plus_intensity, minus_intensity), dim=0).float()
            both_count_weight = torch.cat(
                (
                    2.0 * batch.plus_loss_weight * (batch.count[:, 0:1, :] > 0).float(),
                    2.0
                    * batch.minus_loss_weight_reverse
                    * (batch.count[:, 1:2, :].flip(-1) > 0).float(),
                ),
                dim=0,
            ).to(device)
            count_element = F.smooth_l1_loss(
                both_intensity,
                torch.log1p(both_count.float()),
                reduction="none",
            )
            count_loss = (count_element * both_count_weight.float()).sum()
            residual_loss = delta.float().square().mean()
            loss = (
                presence_loss
                + args.lambda_within * within_loss
                + args.lambda_count * count_loss
                + args.lambda_residual * residual_loss
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step == 1 or step % args.log_every == 0:
            elapsed = time.monotonic() - started
            print(
                f"B step={step}/{args.b_steps} total={float(loss):.6f} "
                f"presence={float(presence_loss):.6f} within={float(within_loss):.6f} "
                f"count={float(count_loss):.6f} residual={float(residual_loss):.6f} "
                f"lr={scheduler.get_last_lr()[0]:.3e} elapsed={elapsed/60:.1f}m",
                flush=True,
            )

        should_save = step % args.save_every == 0 or step == args.b_steps
        time_limit = args.max_minutes > 0 and (time.monotonic() - started) >= 60 * args.max_minutes
        if should_save or time_limit:
            atomic_torch_save(
                checkpoint_path,
                {
                    "experiment_id": EXPERIMENT_ID,
                    "network": "B",
                    "step": step,
                    "total_steps": args.b_steps,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "model_config": {
                        "channels": 256,
                        "film_blocks": film_blocks,
                        "input_channels": 6,
                        "condition_channels": 17,
                    },
                    "profile_config": {
                        "target_length": 1024,
                        "context_flank": 514,
                        "positive_branch_fraction": 0.5,
                    },
                    "loss_config": {
                        "lambda_within": args.lambda_within,
                        "lambda_count": args.lambda_count,
                        "lambda_residual": args.lambda_residual,
                        "exp047_risk_cache": (
                            str(args.risk_cache) if args.risk_cache is not None else None
                        ),
                        "exp047_risk_manifest": risk_manifest_payload,
                    },
                    "regional_cache": str(args.cache_dir),
                    "training_contract": training_contract,
                },
            )
            print("Saved", checkpoint_path, flush=True)
        if time_limit:
            print("Reached --max-minutes; checkpoint saved for resume.", flush=True)
            break
    cache.close()
    if risk_cache is not None:
        del risk_cache


class HistogramMetrics:
    def __init__(self):
        self.total = np.zeros(100_001, dtype=np.int64)
        self.positive = np.zeros(100_001, dtype=np.int64)
        self.positive_counts: list[np.ndarray] = []
        self.positive_scores: list[np.ndarray] = []

    def add(self, probability, target, count, mask) -> None:
        valid = np.asarray(mask, dtype=bool)
        target = np.asarray(target, dtype=bool)
        score = np.rint(np.clip(probability, 0, 1) * 100_000).astype(np.int32)
        self.total += np.bincount(score[valid], minlength=100_001)
        positive = valid & target
        self.positive += np.bincount(score[positive], minlength=100_001)
        if positive.any():
            self.positive_counts.append(np.asarray(count[positive], dtype=np.float32))
            self.positive_scores.append(np.asarray(probability[positive], dtype=np.float32))

    def result(self) -> dict[str, float | int]:
        positives = int(self.positive.sum())
        total = int(self.total.sum())
        pos_desc = self.positive[::-1]
        total_desc = self.total[::-1]
        precision = np.cumsum(pos_desc) / np.maximum(np.cumsum(total_desc), 1)
        ap = float(np.sum((pos_desc / max(positives, 1)) * precision))
        # Spearman is intentionally omitted here; AP is the primary cloud gate.
        return {"positions": total, "positives": positives, "ap5": ap}


@torch.no_grad()
def stage_evaluate_b(args) -> None:
    device = device_from_argument(args.device)
    split, sequences = load_project_data()
    cache = RegionalFeatureCache(args.cache_dir)
    missing = [contig for contig in validation_contigs(split) if not cache.has_contig(contig)]
    if missing:
        raise RuntimeError(f"validation cache misses contigs: {missing}")

    film_blocks = parse_film_blocks(args.film_blocks)
    checkpoint_path = args.run_dir / "network_b" / f"film_{args.film_blocks.replace(',', '-')}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require_completed_checkpoint(checkpoint, "Network B")
    model = build_b(film_blocks).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    config = ProfileWindowConfig(**checkpoint["profile_config"])
    generator = ProfileWindowGenerator(
        split,
        sequences,
        part="validation",
        config=config,
        seed=args.seed,
    )
    metrics = HistogramMetrics()
    flank = config.context_flank
    crop = slice(flank, flank + config.target_length)
    tiles = generator.tile_index
    started = time.monotonic()
    for start in range(0, len(tiles), args.eval_batch):
        selected = tiles.iloc[start : start + args.eval_batch]
        batch = generator.make_batch(selected)
        raw = batch.raw_codes.to(device)
        plus_dna = encode_raw_dna_six_channel(raw)
        dna = torch.cat((plus_dna, reverse_complement_six_channel(plus_dna)), dim=0)
        condition = cache.condition_for_profile_tiles(
            batch.tiles,
            context_flank=flank,
            target_length=config.target_length,
        ).to(device)
        with amp_context(device):
            logit = model(dna, condition)["final_logit"][..., crop]
        probability = torch.sigmoid(logit.float()).cpu().numpy()
        batch_size = len(selected)
        plus = probability[:batch_size, 0]
        minus = probability[batch_size:, 0, ::-1]
        counts = batch.count.numpy()
        masks = batch.mask.numpy()
        targets = batch.target.numpy()
        for index in range(batch_size):
            metrics.add(plus[index], targets[index, 0], counts[index, 0], masks[index, 0])
            metrics.add(minus[index], targets[index, 1], counts[index, 1], masks[index, 1])
        if start == 0 or (start // args.eval_batch + 1) % args.log_every == 0:
            print(
                f"eval tiles={min(start + batch_size, len(tiles))}/{len(tiles)} "
                f"elapsed={(time.monotonic()-started)/60:.1f}m",
                flush=True,
            )
    result = metrics.result()
    result.update({"checkpoint": str(checkpoint_path), "cache": str(args.cache_dir)})
    print(json.dumps(result, indent=2))
    atomic_json(args.run_dir / "network_b" / "validation_metrics.json", result)
    cache.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "inspect",
            "smoke",
            "train-a",
            "cache-a",
            "evaluate-a",
            "train-b",
            "evaluate-b",
        ),
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_RUN_DIR / "regional_cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=490049)
    parser.add_argument("--fold-seed", type=int, default=490049)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--film-blocks", default="2,4,6,8")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--max-minutes", type=float, default=0.0)

    parser.add_argument("--a-steps", type=int, default=20_000)
    parser.add_argument("--a-batch", type=int, default=2)
    parser.add_argument("--a-lr", type=float, default=1e-3)
    parser.add_argument("--a-pos-weight", type=float, default=32.0)
    parser.add_argument("--a-positive-fraction", type=float, default=0.5)

    parser.add_argument("--cache-part", choices=("train", "validation"), default="train")
    parser.add_argument("--allow-in-sample-cache", action="store_true")
    parser.add_argument("--overwrite-cache-contig", action="store_true")

    parser.add_argument("--b-steps", type=int, default=50_000)
    parser.add_argument("--b-batch", type=int, default=2)
    parser.add_argument("--b-lr", type=float, default=1e-3)
    parser.add_argument("--lambda-within", type=float, default=0.1)
    parser.add_argument("--lambda-count", type=float, default=0.1)
    parser.add_argument("--lambda-residual", type=float, default=1e-4)
    parser.add_argument("--risk-cache", type=Path, default=None)
    parser.add_argument("--risk-manifest", type=Path, default=None)
    parser.add_argument("--eval-batch", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.run_dir = args.run_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    if args.risk_cache is not None:
        args.risk_cache = args.risk_cache.resolve()
    if args.risk_manifest is not None:
        args.risk_manifest = args.risk_manifest.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    stages = {
        "inspect": stage_inspect,
        "smoke": stage_smoke,
        "train-a": stage_train_a,
        "cache-a": stage_cache_a,
        "evaluate-a": stage_evaluate_a,
        "train-b": stage_train_b,
        "evaluate-b": stage_evaluate_b,
    }
    stages[args.stage](args)


if __name__ == "__main__":
    main()
