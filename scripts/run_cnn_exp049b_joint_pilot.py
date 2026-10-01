"""Paired 20K end-to-end pilot for the CNN-EXP-049 cascade.

The control arm trains the localizer with an all-zero regional condition.  The
joint arm starts from the registered A-v1 20K checkpoint and propagates the
nucleotide loss through FiLM into Network A.  Both arms use the same Network-B
initialisation, profile sampler seed, loss, and validation scan.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_project.epic_cnn import (  # noqa: E402
    ConditionalLocalizerB,
    ProfileWindowConfig,
    ProfileWindowGenerator,
    RegionalRateDetectorA,
    encode_raw_dna_six_channel,
    reverse_complement_six_channel,
)
from scripts.run_cnn_exp049_cloud import (  # noqa: E402
    amp_context,
    atomic_json,
    atomic_torch_save,
    device_from_argument,
    gradient_scaler,
    load_project_data,
    parse_film_blocks,
    weighted_profile_bce,
    within_r16_pairwise_loss,
)
from scripts.run_cnn_exp049a_rate_pilot import (  # noqa: E402
    masked_poisson_nll,
    regional_pairwise_loss,
)


EXPERIMENT_ID = "CNN-EXP-049B-JOINT-PILOT"
REGION_SIZE = 16
A_INPUT_BP = 8192


def _padded_raw(sequence: np.ndarray, start: int, length: int) -> np.ndarray:
    result = np.full(length, 8, dtype=np.uint8)
    source_start = max(start, 0)
    source_end = min(start + length, len(sequence))
    if source_end > source_start:
        destination_start = source_start - start
        result[destination_start : destination_start + source_end - source_start] = (
            sequence[source_start:source_end]
        )
    return result


def aligned_long_raw(
    tiles,
    sequences: Mapping[str, np.ndarray],
    *,
    target_length: int,
    input_bp: int = A_INPUT_BP,
) -> tuple[torch.Tensor, np.ndarray]:
    """Materialise R16-aligned long windows centred on profile targets."""

    if input_bp % REGION_SIZE:
        raise ValueError("Network-A input must be divisible by R16")
    raw = np.full((len(tiles), input_bp), 8, dtype=np.uint8)
    starts = np.empty(len(tiles), dtype=np.int64)
    for index, tile in enumerate(tiles.itertuples(index=False)):
        target_start = int(tile.target_start)
        center = target_start + target_length // 2
        start = ((center - input_bp // 2) // REGION_SIZE) * REGION_SIZE
        starts[index] = start
        raw[index] = _padded_raw(np.asarray(sequences[str(tile.contig)]), start, input_bp)
    return torch.from_numpy(raw), starts


def _gather_regions(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if features.ndim != 3 or indices.ndim != 2:
        raise ValueError("features must be [B,C,R] and indices must be [B,L]")
    if int(indices.min()) < 0 or int(indices.max()) >= features.shape[-1]:
        raise ValueError("requested regional condition falls outside Network-A window")
    return torch.gather(features, 2, indices[:, None, :].expand(-1, features.shape[1], -1))


def online_regional_condition(
    model_a: RegionalRateDetectorA,
    tiles,
    sequences: Mapping[str, np.ndarray],
    *,
    target_length: int,
    context_flank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return differentiable base condition and target-region log-rates.

    The condition order is plus examples followed by reverse-oriented minus
    examples, matching Network B.  Target log-rates remain in genomic order for
    both strands so they align with ``ProfileBatch.count``.
    """

    raw_cpu, starts = aligned_long_raw(
        tiles, sequences, target_length=target_length, input_bp=A_INPUT_BP
    )
    plus_dna = encode_raw_dna_six_channel(raw_cpu.to(device, non_blocking=True))
    both_dna = torch.cat((plus_dna, reverse_complement_six_channel(plus_dna)), dim=0)
    condition_oriented = model_a.condition(both_dna)
    batch_size = len(tiles)
    plus_regions = condition_oriented[:batch_size]
    minus_regions_genomic = condition_oriented[batch_size:].flip(-1)

    input_length = target_length + 2 * context_flank
    base_indices = []
    target_indices = []
    for start, tile in zip(starts, tiles.itertuples(index=False), strict=True):
        target_start = int(tile.target_start)
        genomic_positions = target_start - context_flank + np.arange(input_length)
        base_indices.append(genomic_positions // REGION_SIZE - start // REGION_SIZE)
        target_regions = target_start // REGION_SIZE + np.arange(target_length // REGION_SIZE)
        target_indices.append(target_regions - start // REGION_SIZE)
    base_index = torch.as_tensor(np.stack(base_indices), device=device, dtype=torch.long)
    target_index = torch.as_tensor(np.stack(target_indices), device=device, dtype=torch.long)

    plus_condition = _gather_regions(plus_regions, base_index)
    minus_condition = _gather_regions(minus_regions_genomic, base_index).flip(-1)
    target_log_rate = torch.cat(
        (
            _gather_regions(plus_regions[:, :1], target_index),
            _gather_regions(minus_regions_genomic[:, :1], target_index),
        ),
        dim=0,
    )
    return torch.cat((plus_condition, minus_condition), dim=0), target_log_rate


def regional_targets_from_profile(batch, device: torch.device):
    target_length = batch.count.shape[-1]
    if target_length % REGION_SIZE:
        raise ValueError("profile target length must be divisible by R16")
    regions = target_length // REGION_SIZE
    count = batch.count.reshape(len(batch.tiles), 2, regions, REGION_SIZE).sum(-1)
    mask = batch.mask.reshape(len(batch.tiles), 2, regions, REGION_SIZE).any(-1)
    both_count = torch.cat((count[:, 0:1], count[:, 1:2]), dim=0).to(device)
    both_mask = torch.cat((mask[:, 0:1], mask[:, 1:2]), dim=0).to(device)
    return both_count, (both_count > 0).float(), both_mask


def build_b(seed: int, film_blocks: tuple[int, ...], channels: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return ConditionalLocalizerB(
        channels=channels,
        film_after_blocks=film_blocks,
    )


def load_a(path: Path, device: torch.device) -> tuple[RegionalRateDetectorA, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("experiment_id") != "CNN-EXP-049A-V1-RATE-PILOT":
        raise ValueError(f"unexpected Network-A checkpoint: {checkpoint.get('experiment_id')!r}")
    if int(checkpoint.get("step", -1)) != 20_000:
        raise ValueError("joint pilot requires the evaluated A-v1 step_020000.pt")
    model = RegionalRateDetectorA()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device), checkpoint


def risk_state(args, generator, config):
    if args.risk_cache is None:
        return None, None, None
    manifest_path = args.risk_manifest or args.risk_cache.with_name("risk_strata_manifest.json")
    if not args.risk_cache.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"risk cache/manifest not found: {args.risk_cache}, {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shape = tuple(manifest["shape"])
    expected = (len(generator.tile_index), 2, config.target_length)
    if shape != expected:
        raise ValueError(f"risk cache shape {shape} != {expected}")
    cache = np.memmap(args.risk_cache, mode="r", dtype=np.uint8, shape=shape)
    multipliers = np.asarray(manifest["candidate_negative_weight_multipliers"], dtype=np.float32)
    return cache, multipliers, manifest


def presence_weights(batch, risk_cache, multipliers):
    plus = batch.plus_loss_weight
    minus = batch.minus_loss_weight_reverse
    if risk_cache is None:
        return plus, minus
    tile_ids = batch.tile_id.numpy().astype(np.int64, copy=False)
    strata = np.asarray(risk_cache[tile_ids])
    valid_negative = batch.mask.numpy() & ~batch.target.numpy().astype(bool)
    if np.any(strata[valid_negative] == 255):
        raise AssertionError("valid negative is missing from EXP047 risk cache")
    multiplier = np.ones(strata.shape, dtype=np.float32)
    for stratum, value in enumerate(multipliers):
        multiplier[strata == stratum] = value
    return (
        plus * torch.from_numpy(multiplier[:, 0:1, :]),
        minus * torch.from_numpy(multiplier[:, 1:2, ::-1].copy()),
    )


def training_contract(args, arm: str, film_blocks, risk_manifest, a_checkpoint):
    return {
        "experiment_id": EXPERIMENT_ID,
        "arm": arm,
        "seed": args.seed,
        "steps": args.steps,
        "batch": args.batch,
        "b_channels": args.b_channels,
        "b_lr": args.b_lr,
        "a_lr": args.a_lr if arm == "joint" else None,
        "a_aux_weight": args.a_aux_weight if arm == "joint" else None,
        "a_rank_weight": args.a_rank_weight if arm == "joint" else None,
        "a_checkpoint": str(a_checkpoint) if arm == "joint" else None,
        "film_blocks": list(film_blocks),
        "lambda_within": args.lambda_within,
        "lambda_count": args.lambda_count,
        "lambda_residual": args.lambda_residual,
        "risk_cache_sha256": risk_manifest.get("cache_sha256") if risk_manifest else None,
        "target_length": 1024,
        "context_flank": 514,
        "positive_branch_fraction": 0.5,
    }


def stage_smoke(args):
    device = device_from_argument(args.device)
    split, sequences = load_project_data()
    config = ProfileWindowConfig(target_length=1024, context_flank=514, positive_branch_fraction=0.5)
    generator = ProfileWindowGenerator(split, sequences, part="train", config=config, seed=args.seed)
    batch = generator.sample_batch(2)
    model_a, _ = load_a(args.a_checkpoint, device)
    model_b = build_b(args.seed, parse_film_blocks(args.film_blocks), args.b_channels).to(device)
    raw = batch.raw_codes.to(device)
    plus = encode_raw_dna_six_channel(raw)
    dna = torch.cat((plus, reverse_complement_six_channel(plus)), dim=0)
    with amp_context(device):
        condition, regional_log_rate = online_regional_condition(
            model_a,
            batch.tiles,
            sequences,
            target_length=config.target_length,
            context_flank=config.context_flank,
            device=device,
        )
    count, target, mask = regional_targets_from_profile(batch, device)
    with amp_context(device):
        output = model_b(dna, condition)
        loss = output["final_logit"].float().square().mean()
        loss = loss + 0.01 * masked_poisson_nll(regional_log_rate, count, mask)
    loss.backward()
    if model_a.latent_head.weight.grad is None or model_b.stem.weight.grad is None:
        raise AssertionError("joint gradient does not reach both networks")
    print(
        json.dumps(
            {
                "condition": list(condition.shape),
                "regional_log_rate": list(regional_log_rate.shape),
                "regional_target": list(target.shape),
                "b_output": list(output["final_logit"].shape),
                "a_latent_grad": float(model_a.latent_head.weight.grad.float().norm()),
                "b_stem_grad": float(model_b.stem.weight.grad.float().norm()),
            },
            indent=2,
        )
    )


def _cosine_factor(step: int, total_steps: int) -> float:
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def stage_train(args, arm: str, *, continuation: bool = False):
    device = device_from_argument(args.device)
    split, sequences = load_project_data()
    config = ProfileWindowConfig(target_length=1024, context_flank=514, positive_branch_fraction=0.5)
    generator = ProfileWindowGenerator(split, sequences, part="train", config=config, seed=args.seed)
    risk_cache, risk_multipliers, risk_manifest = risk_state(args, generator, config)
    film_blocks = parse_film_blocks(args.film_blocks)
    model_b = build_b(args.seed, film_blocks, args.b_channels).to(device)
    model_a = None
    a_source = None
    if arm == "joint":
        model_a, a_source = load_a(args.a_checkpoint, device)
    base_contract = training_contract(args, arm, film_blocks, risk_manifest, args.a_checkpoint)
    if continuation:
        contract = {
            **base_contract,
            "phase": "20k-to-50k-cosine",
            "base_step": args.steps,
            "continuation_steps": args.continue_steps,
            "total_steps": args.steps + args.continue_steps,
            "b_continue_lr": args.b_continue_lr,
            "a_continue_lr": args.a_continue_lr if arm == "joint" else None,
            "cosine_final_factor": 0.1,
        }
    else:
        contract = base_contract

    groups = [{"params": model_b.parameters(), "lr": args.b_lr, "name": "B"}]
    max_lrs = [args.b_lr]
    if model_a is not None:
        groups.append({"params": model_a.parameters(), "lr": args.a_lr, "name": "A"})
        max_lrs.append(args.a_lr)
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scaler = gradient_scaler(device)
    if continuation:
        base_path = args.run_dir / arm / "latest.pt"
        checkpoint_path = args.run_dir / arm / "continuation_50k.pt"
        if not base_path.is_file():
            raise FileNotFoundError(base_path)
        base = torch.load(base_path, map_location="cpu", weights_only=False)
        if int(base["step"]) != args.steps or int(base["total_steps"]) != args.steps:
            raise RuntimeError(f"{arm} base checkpoint is not complete at {args.steps}")
        if base.get("training_contract") != base_contract:
            raise ValueError("20K base checkpoint contract differs from continuation command")
        model_b.load_state_dict(base["model_b_state_dict"])
        if model_a is not None:
            model_a.load_state_dict(base["model_a_state_dict"])
        optimizer.load_state_dict(base["optimizer_state_dict"])
        scaler.load_state_dict(base.get("scaler_state_dict", {}))
        generator.rng.bit_generator.state = base["sampler_rng_state"]
        continuation_lrs = [args.b_continue_lr]
        if model_a is not None:
            continuation_lrs.append(args.a_continue_lr)
        for group, learning_rate in zip(
            optimizer.param_groups, continuation_lrs, strict=True
        ):
            group["lr"] = learning_rate
            group["initial_lr"] = learning_rate
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[
                lambda step, total=args.continue_steps: _cosine_factor(step, total)
                for _ in optimizer.param_groups
            ],
        )
        start_phase_step = 0
        if checkpoint_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint.get("training_contract") != contract:
                raise ValueError("continuation checkpoint contract differs from this command")
            model_b.load_state_dict(checkpoint["model_b_state_dict"])
            if model_a is not None:
                model_a.load_state_dict(checkpoint["model_a_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            saved_lrs = [group["lr"] for group in optimizer.param_groups]
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            for group, learning_rate in zip(
                optimizer.param_groups, saved_lrs, strict=True
            ):
                group["lr"] = learning_rate
            scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
            generator.rng.bit_generator.state = checkpoint["sampler_rng_state"]
            start_phase_step = int(checkpoint["phase_step"])
            print(
                f"Resuming {arm} continuation from global step "
                f"{args.steps + start_phase_step}"
            )
        phase_steps = args.continue_steps
        global_offset = args.steps
    else:
        checkpoint_path = args.run_dir / arm / "latest.pt"
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lrs, total_steps=args.steps, pct_start=0.05
        )
        start_phase_step = 0
        if checkpoint_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint.get("training_contract") != contract:
                raise ValueError("checkpoint training contract differs from this command")
            model_b.load_state_dict(checkpoint["model_b_state_dict"])
            if model_a is not None:
                model_a.load_state_dict(checkpoint["model_a_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
            generator.rng.bit_generator.state = checkpoint["sampler_rng_state"]
            start_phase_step = int(checkpoint["step"])
            print(f"Resuming {arm} from step {start_phase_step}")
        phase_steps = args.steps
        global_offset = 0

    model_b.train()
    if model_a is not None:
        model_a.train()
    flank = config.context_flank
    crop = slice(flank, flank + config.target_length)
    started = time.monotonic()
    for phase_step in range(start_phase_step + 1, phase_steps + 1):
        global_step = global_offset + phase_step
        batch = generator.sample_batch(args.batch)
        raw = batch.raw_codes.to(device, non_blocking=True)
        plus_dna = encode_raw_dna_six_channel(raw)
        dna = torch.cat((plus_dna, reverse_complement_six_channel(plus_dna)), dim=0)
        if model_a is None:
            condition = torch.zeros(
                2 * args.batch, 17, config.input_length, device=device, dtype=dna.dtype
            )
            regional_aux = torch.zeros((), device=device)
        else:
            with amp_context(device):
                condition, regional_log_rate = online_regional_condition(
                    model_a,
                    batch.tiles,
                    sequences,
                    target_length=config.target_length,
                    context_flank=flank,
                    device=device,
                )
            regional_count, regional_target, regional_mask = regional_targets_from_profile(batch, device)
            regional_aux = masked_poisson_nll(regional_log_rate, regional_count, regional_mask)
            regional_aux = regional_aux + args.a_rank_weight * regional_pairwise_loss(
                regional_log_rate, regional_target, regional_mask
            )

        optimizer.zero_grad(set_to_none=True)
        with amp_context(device):
            output = model_b(dna, condition)
            final = output["final_logit"][..., crop]
            intensity = output["intensity"][..., crop]
            delta = output["delta_logit"][..., crop]
            plus_logit, minus_logit = final[: args.batch], final[args.batch :]
            plus_target = batch.plus_target.to(device)
            minus_target = batch.minus_target_reverse.to(device)
            plus_weight_cpu, minus_weight_cpu = presence_weights(
                batch, risk_cache, risk_multipliers
            )
            presence = weighted_profile_bce(
                plus_logit,
                minus_logit,
                plus_target,
                minus_target,
                plus_weight_cpu.to(device),
                minus_weight_cpu.to(device),
            )
            both_logit = torch.cat((plus_logit, minus_logit), dim=0)
            both_target = torch.cat((plus_target, minus_target), dim=0)
            both_mask = torch.cat(
                (batch.mask[:, 0:1], batch.mask[:, 1:2].flip(-1)), dim=0
            ).to(device)
            within = within_r16_pairwise_loss(both_logit, both_target, both_mask)
            plus_count = batch.count[:, 0:1].to(device)
            minus_count = batch.count[:, 1:2].flip(-1).to(device)
            both_count = torch.cat((plus_count, minus_count), dim=0)
            count_weight = torch.cat(
                (
                    2 * batch.plus_loss_weight * (batch.count[:, 0:1] > 0),
                    2 * batch.minus_loss_weight_reverse * (batch.count[:, 1:2].flip(-1) > 0),
                ),
                dim=0,
            ).to(device)
            count_element = F.smooth_l1_loss(
                intensity.float(), torch.log1p(both_count.float()), reduction="none"
            )
            count_loss = (count_element * count_weight.float()).sum()
            residual = delta.float().square().mean()
            loss = (
                presence
                + args.lambda_within * within
                + args.lambda_count * count_loss
                + args.lambda_residual * residual
                + args.a_aux_weight * regional_aux
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        parameters = list(model_b.parameters())
        if model_a is not None:
            parameters += list(model_a.parameters())
        torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if phase_step == 1 or phase_step % args.log_every == 0:
            elapsed = (time.monotonic() - started) / 60
            lrs = ",".join(f"{value:.2e}" for value in scheduler.get_last_lr())
            print(
                f"{arm} step={global_step}/{global_offset + phase_steps} "
                f"loss={float(loss.detach()):.6f} "
                f"presence={float(presence.detach()):.6f} within={float(within.detach()):.6f} "
                f"count={float(count_loss.detach()):.6f} Aaux={float(regional_aux.detach()):.6f} "
                f"lr={lrs} elapsed={elapsed:.1f}m",
                flush=True,
            )

        time_limit = args.max_minutes > 0 and time.monotonic() - started >= 60 * args.max_minutes
        if phase_step % args.save_every == 0 or phase_step == phase_steps or time_limit:
            atomic_torch_save(
                checkpoint_path,
                {
                    "experiment_id": EXPERIMENT_ID,
                    "arm": arm,
                    "step": global_step,
                    "total_steps": global_offset + phase_steps,
                    "phase_step": phase_step,
                    "phase_total_steps": phase_steps,
                    "model_b_state_dict": model_b.state_dict(),
                    "model_a_state_dict": model_a.state_dict() if model_a is not None else None,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "sampler_rng_state": generator.rng.bit_generator.state,
                    "training_contract": contract,
                    "profile_config": vars(config),
                    "a_source": {
                        "path": str(args.a_checkpoint),
                        "experiment_id": a_source.get("experiment_id"),
                        "step": a_source.get("step"),
                    }
                    if a_source is not None
                    else None,
                },
            )
            print("Saved", checkpoint_path, flush=True)
        if time_limit:
            print("Reached --max-minutes; checkpoint saved for resume.", flush=True)
            break
    if risk_cache is not None:
        del risk_cache


def dense_rank(values: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(values, return_inverse=True)
    return inverse.astype(np.float64) + 1.0 if len(unique) else np.empty(0)


class Metrics:
    def __init__(self):
        self.total = np.zeros(100_001, dtype=np.int64)
        self.positive = np.zeros(100_001, dtype=np.int64)
        self.positive_scores = []
        self.positive_counts = []

    def add(self, probability, target, count, mask):
        valid = np.asarray(mask, dtype=bool)
        target = np.asarray(target, dtype=bool)
        probability = np.asarray(probability, dtype=np.float32)
        score = np.rint(np.clip(probability, 0, 1) * 100_000).astype(np.int32)
        self.total += np.bincount(score[valid], minlength=100_001)
        positive = valid & target
        self.positive += np.bincount(score[positive], minlength=100_001)
        self.positive_scores.append(probability[positive])
        self.positive_counts.append(np.asarray(count, dtype=np.float32)[positive])

    def result(self):
        positives = int(self.positive.sum())
        pos_desc = self.positive[::-1]
        total_desc = self.total[::-1]
        precision = np.cumsum(pos_desc) / np.maximum(np.cumsum(total_desc), 1)
        ap = float(np.sum((pos_desc / max(positives, 1)) * precision))
        score = np.concatenate(self.positive_scores)
        count = np.concatenate(self.positive_counts)
        x, y = dense_rank(score), dense_rank(count)
        x -= x.mean()
        y -= y.mean()
        denominator = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
        spearman = float(np.dot(x, y) / denominator) if denominator else None
        return {"positions": int(self.total.sum()), "positives": positives, "ap5": ap, "epic_spearman": spearman}


@torch.no_grad()
def stage_evaluate(args, arm: str, *, continuation: bool = False):
    device = device_from_argument(args.device)
    checkpoint_path = args.run_dir / arm / (
        "continuation_50k.pt" if continuation else "latest.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint["step"]) != int(checkpoint["total_steps"]):
        raise RuntimeError(f"{arm} checkpoint is incomplete")
    contract = checkpoint["training_contract"]
    config = ProfileWindowConfig(**checkpoint["profile_config"])
    film_blocks = tuple(contract["film_blocks"])
    model_b = ConditionalLocalizerB(
        channels=int(contract["b_channels"]), film_after_blocks=film_blocks
    ).to(device)
    model_b.load_state_dict(checkpoint["model_b_state_dict"])
    model_b.eval()
    model_a = None
    if arm == "joint":
        model_a = RegionalRateDetectorA().to(device)
        model_a.load_state_dict(checkpoint["model_a_state_dict"])
        model_a.eval()

    split, sequences = load_project_data()
    generator = ProfileWindowGenerator(split, sequences, part="validation", config=config, seed=args.seed)
    overall = Metrics()
    per_contig: dict[str, Metrics] = {}
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
        if model_a is None:
            condition = torch.zeros(2 * len(selected), 17, config.input_length, device=device)
        else:
            with amp_context(device):
                condition, _ = online_regional_condition(
                    model_a,
                    batch.tiles,
                    sequences,
                    target_length=config.target_length,
                    context_flank=flank,
                    device=device,
                )
        with amp_context(device):
            logit = model_b(dna, condition)["final_logit"][..., crop]
        probability = torch.sigmoid(logit.float()).cpu().numpy()
        batch_size = len(selected)
        plus = probability[:batch_size, 0]
        minus = probability[batch_size:, 0, ::-1]
        for index, tile in enumerate(batch.tiles.itertuples(index=False)):
            contig = str(tile.contig)
            metric = per_contig.setdefault(contig, Metrics())
            for strand_index, values in enumerate((plus[index], minus[index])):
                arguments = (
                    values,
                    batch.target[index, strand_index].numpy(),
                    batch.count[index, strand_index].numpy(),
                    batch.mask[index, strand_index].numpy(),
                )
                overall.add(*arguments)
                metric.add(*arguments)
        if start == 0 or (start // args.eval_batch + 1) % args.log_every == 0:
            print(
                f"{arm} eval tiles={min(start + batch_size, len(tiles))}/{len(tiles)} "
                f"elapsed={(time.monotonic() - started)/60:.1f}m",
                flush=True,
            )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "arm": arm,
        "steps": int(checkpoint["step"]),
        "overall": overall.result(),
        "per_contig": {name: metric.result() for name, metric in sorted(per_contig.items())},
        "checkpoint": str(checkpoint_path),
    }
    print(json.dumps(result, indent=2))
    atomic_json(args.run_dir / arm / "validation_metrics.json", result)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "smoke",
            "train-control",
            "train-joint",
            "continue-control",
            "continue-joint",
            "evaluate-control",
            "evaluate-joint",
            "evaluate50-control",
            "evaluate50-joint",
        ),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--a-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=490049)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--b-channels", type=int, default=256)
    parser.add_argument("--b-lr", type=float, default=1e-3)
    parser.add_argument("--a-lr", type=float, default=1e-4)
    parser.add_argument("--continue-steps", type=int, default=30_000)
    parser.add_argument("--b-continue-lr", type=float, default=3e-4)
    parser.add_argument("--a-continue-lr", type=float, default=3e-5)
    parser.add_argument("--a-aux-weight", type=float, default=0.1)
    parser.add_argument("--a-rank-weight", type=float, default=0.1)
    parser.add_argument("--film-blocks", default="2,4,6,8")
    parser.add_argument("--lambda-within", type=float, default=0.1)
    parser.add_argument("--lambda-count", type=float, default=0.1)
    parser.add_argument("--lambda-residual", type=float, default=1e-4)
    parser.add_argument("--risk-cache", type=Path)
    parser.add_argument("--risk-manifest", type=Path)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-batch", type=int, default=1)
    parser.add_argument("--max-minutes", type=float, default=300)
    return parser


def main():
    args = build_parser().parse_args()
    args.run_dir = args.run_dir.resolve()
    args.a_checkpoint = args.a_checkpoint.resolve()
    if args.risk_cache is not None:
        args.risk_cache = args.risk_cache.resolve()
    if args.risk_manifest is not None:
        args.risk_manifest = args.risk_manifest.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    stages = {
        "smoke": stage_smoke,
        "train-control": lambda value: stage_train(value, "control"),
        "train-joint": lambda value: stage_train(value, "joint"),
        "continue-control": lambda value: stage_train(
            value, "control", continuation=True
        ),
        "continue-joint": lambda value: stage_train(
            value, "joint", continuation=True
        ),
        "evaluate-control": lambda value: stage_evaluate(value, "control"),
        "evaluate-joint": lambda value: stage_evaluate(value, "joint"),
        "evaluate50-control": lambda value: stage_evaluate(
            value, "control", continuation=True
        ),
        "evaluate50-joint": lambda value: stage_evaluate(
            value, "joint", continuation=True
        ),
    }
    stages[args.stage](args)


if __name__ == "__main__":
    main()
