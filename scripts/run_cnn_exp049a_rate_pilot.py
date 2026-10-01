"""Leakage-safe count-aware Network-A pilot for CNN-EXP-049 revision 1.

This runner deliberately stops at the smallest decisive experiment: one
held-out contig fold, checkpoints at registered milestones, and natural OOF
regional AP/Spearman.  It does not train Network B.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

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
    RegionalFeatureCache,
    RegionalRateDetectorA,
    RegionalWindowGenerator,
    predict_contig_to_regional_cache,
)
from scripts.run_cnn_exp049_cloud import (  # noqa: E402
    amp_context,
    atomic_json,
    atomic_torch_save,
    contract_sha256,
    device_from_argument,
    fold_assignment,
    gradient_scaler,
    load_project_data,
    train_contigs,
)


EXPERIMENT_ID = "CNN-EXP-049A-V1-RATE-PILOT"
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "experiments"
    / "CNN_EXP049A_V1_RATE_PILOT"
    / "run_001"
)


def parse_milestones(value: str, total_steps: int) -> tuple[int, ...]:
    milestones = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if any(step <= 0 or step > total_steps for step in milestones):
        raise ValueError("every milestone must lie in [1, --a-steps]")
    return milestones


def checkpoint_path(run_dir: Path, fold: int, step: int | None = None) -> Path:
    directory = run_dir / "network_a_rate" / f"fold_{fold:02d}"
    return directory / ("latest.pt" if step is None else f"step_{step:06d}.pt")


def masked_poisson_nll(
    log_rate: torch.Tensor,
    count: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    value = log_rate.float().clamp(-20.0, 15.0)
    element = torch.exp(value) - count.float() * value
    weight = mask.float()
    return (element * weight).sum() / weight.sum().clamp_min(1.0)


def regional_pairwise_loss(
    log_rate: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Rank every active R16 above valid empty bins in the same window."""

    z = log_rate.float()
    positive = target.bool() & mask.bool()
    negative = (~target.bool()) & mask.bool()
    pair_mask = positive.unsqueeze(-1) & negative.unsqueeze(-2)
    if not bool(pair_mask.any()):
        return z.sum() * 0.0
    difference = z.unsqueeze(-2) - z.unsqueeze(-1)
    return F.softplus(difference)[pair_mask].mean()


def poisson_active_probability(log_rate: torch.Tensor) -> torch.Tensor:
    rate = torch.exp(log_rate.float().clamp(-20.0, 15.0))
    return -torch.expm1(-rate)


def build_model() -> RegionalRateDetectorA:
    return RegionalRateDetectorA()


def training_contract(args, fit_contigs: list[str], held_out: list[str]) -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "architecture": "regional-rate-latent-bottleneck-v1",
        "fold": args.fold,
        "fold_count": args.folds,
        "fold_seed": args.fold_seed,
        "seed": args.seed,
        "fit_contigs": fit_contigs,
        "held_out_contigs": held_out,
        "total_steps": args.a_steps,
        "batch_size": args.a_batch,
        "learning_rate": args.a_lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "positive_window_fraction": args.a_positive_fraction,
        "lambda_rank": args.lambda_rank,
        "input_bp": 8192,
        "region_size": 16,
        "latent_channels": 16,
        "condition_channels": 17,
        "target": "strand-specific sum of pooled nucleotide counts per R16",
        "loss": "masked Poisson NLL plus within-window active-vs-empty pairwise",
    }


def checkpoint_payload(
    *,
    args,
    model,
    optimizer,
    scheduler,
    scaler,
    step: int,
    contract: dict,
) -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "network": "A-rate",
        "fold": args.fold,
        "step": step,
        "total_steps": args.a_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "model_config": {
            "architecture": "regional-rate-latent-bottleneck-v1",
            "input_bp": 8192,
            "region_size": 16,
            "condition_channels": model.condition_channels,
        },
        "training_contract": contract,
    }


def stage_inspect(args) -> None:
    split, sequences = load_project_data()
    folds = fold_assignment(train_contigs(split), args.folds, args.fold_seed)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "run_dir": str(args.run_dir),
        "sequence_count": len(sequences),
        "fold": args.fold,
        "held_out_contigs": folds[args.fold],
        "objective": "regional count log-rate: Poisson + pairwise ranking",
        "milestones": parse_milestones(args.a_milestones, args.a_steps),
    }
    print(json.dumps(payload, indent=2))
    atomic_json(args.run_dir / "pilot_inspect.json", payload)


def stage_smoke(args) -> None:
    device = device_from_argument(args.device)
    model = build_model().to(device).train()
    generator_split, sequences = load_project_data()
    folds = fold_assignment(train_contigs(generator_split), args.folds, args.fold_seed)
    fit = [name for name in train_contigs(generator_split) if name not in set(folds[args.fold])]
    generator = RegionalWindowGenerator(
        generator_split,
        sequences,
        part="train",
        contigs=fit,
        positive_window_fraction=0.0,
        seed=args.seed,
    )
    batch = generator.sample_batch(1)
    dna = generator.encode_oriented(batch, device)
    log_rate, latent = model(dna)
    loss = masked_poisson_nll(log_rate, batch.count.to(device), batch.mask.to(device))
    loss = loss + args.lambda_rank * regional_pairwise_loss(
        log_rate, batch.target.to(device), batch.mask.to(device)
    )
    loss.backward()
    if model.latent_head.weight.grad is None:
        raise AssertionError("latent head is disconnected from the supervised loss")
    print(
        "A-rate",
        tuple(log_rate.shape),
        tuple(latent.shape),
        "loss=",
        float(loss.detach()),
        "latent_grad_norm=",
        float(model.latent_head.weight.grad.float().norm()),
    )


def stage_train_a(args) -> None:
    device = device_from_argument(args.device)
    split, sequences = load_project_data()
    folds = fold_assignment(train_contigs(split), args.folds, args.fold_seed)
    if args.fold < 0 or args.fold >= args.folds:
        raise ValueError("the pilot requires one held-out --fold in [0, folds)")
    held_out = folds[args.fold]
    fit_contigs = [name for name in train_contigs(split) if name not in set(held_out)]
    generator = RegionalWindowGenerator(
        split,
        sequences,
        part="train",
        contigs=fit_contigs,
        window_bp=8192,
        positive_window_fraction=args.a_positive_fraction,
        seed=args.seed + args.fold,
    )
    contract = training_contract(args, fit_contigs, held_out)
    milestones = parse_milestones(args.a_milestones, args.a_steps)
    print(
        f"A-rate fold={args.fold} fit={fit_contigs} held_out={held_out} "
        f"allowed_R16={generator.allowed_region_count:,} "
        f"active_R16={generator.active_region_count:,} milestones={milestones}"
    )

    latest = checkpoint_path(args.run_dir, args.fold)
    model = build_model().to(device)
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
    if latest.is_file():
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        if checkpoint.get("training_contract") != contract:
            raise ValueError("checkpoint training contract differs from this pilot command")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
        start_step = int(checkpoint["step"])
        print("Resuming A-rate from", latest, "step", start_step)

    started = time.monotonic()
    model.train()
    for step in range(start_step + 1, args.a_steps + 1):
        batch = generator.sample_batch(args.a_batch)
        dna = generator.encode_oriented(batch, device)
        count = batch.count.to(device, non_blocking=True)
        target = batch.target.to(device, non_blocking=True)
        mask = batch.mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device):
            log_rate, _latent = model(dna)
            poisson = masked_poisson_nll(log_rate, count, mask)
            ranking = regional_pairwise_loss(log_rate, target, mask)
            loss = poisson + args.lambda_rank * ranking
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if step == 1 and model.latent_head.weight.grad is None:
            raise AssertionError("latent head is disconnected from the supervised loss")
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step == 1 or step % args.log_every == 0:
            probability = poisson_active_probability(log_rate.detach())
            prediction = probability >= 0.5
            positive = target.bool() & mask
            recall = float((prediction & positive).sum() / positive.sum().clamp_min(1))
            print(
                f"A-rate fold={args.fold} step={step}/{args.a_steps} "
                f"loss={float(loss.detach()):.6f} poisson={float(poisson.detach()):.6f} "
                f"rank={float(ranking.detach()):.6f} recall@0.5={recall:.5f} "
                f"lr={scheduler.get_last_lr()[0]:.3e} "
                f"elapsed={(time.monotonic()-started)/60:.1f}m",
                flush=True,
            )

        time_limit = args.max_minutes > 0 and (time.monotonic() - started) >= 60 * args.max_minutes
        save_latest = step % args.save_every == 0 or step == args.a_steps or time_limit
        save_milestone = step in milestones
        if save_latest or save_milestone:
            payload = checkpoint_payload(
                args=args,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                step=step,
                contract=contract,
            )
            if save_latest:
                atomic_torch_save(latest, payload)
                print("Saved", latest, flush=True)
            if save_milestone:
                snapshot = checkpoint_path(args.run_dir, args.fold, step)
                atomic_torch_save(snapshot, payload)
                print("Saved milestone", snapshot, flush=True)
        if time_limit:
            print("Reached --max-minutes; checkpoint saved for resume.", flush=True)
            break


def load_checkpoint_model(path: Path, device: torch.device) -> tuple[RegionalRateDetectorA, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"not an {EXPERIMENT_ID} checkpoint: {path}")
    model = build_model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def stage_cache_a(args) -> None:
    device = device_from_argument(args.device)
    split, sequences = load_project_data()
    folds = fold_assignment(train_contigs(split), args.folds, args.fold_seed)
    selected_step = args.a_checkpoint_step if args.a_checkpoint_step > 0 else None
    path = checkpoint_path(args.run_dir, args.fold, selected_step)
    if not path.is_file():
        raise FileNotFoundError(path)
    model, checkpoint = load_checkpoint_model(path, device)
    if selected_step is None and int(checkpoint["step"]) != int(checkpoint["total_steps"]):
        raise RuntimeError("latest A-rate checkpoint is incomplete; resume training first")
    if selected_step is not None and int(checkpoint["step"]) != selected_step:
        raise RuntimeError("milestone checkpoint step mismatch")

    cache = RegionalFeatureCache.create(
        args.cache_dir,
        channels=model.condition_channels,
        metadata={
            "experiment_id": EXPERIMENT_ID,
            "fold": args.fold,
            "checkpoint_step": int(checkpoint["step"]),
        },
    )
    source = {
        "fold": args.fold,
        "checkpoint": str(path),
        "checkpoint_step": int(checkpoint["step"]),
        "oof": True,
        "objective": "regional-rate-poisson-rank",
        "training_contract_sha256": contract_sha256(checkpoint["training_contract"]),
    }
    for index, contig in enumerate(folds[args.fold], start=1):
        if cache.has_contig(contig) and not args.overwrite_cache_contig:
            if cache.manifest["contigs"][contig].get("source") != source:
                raise RuntimeError(
                    f"cached {contig} has different source; use a separate cache-dir "
                    "or --overwrite-cache-contig"
                )
            print(f"[{index}/{len(folds[args.fold])}] skip cached {contig}")
            continue
        print(f"[{index}/{len(folds[args.fold])}] predict {contig}", flush=True)
        predict_contig_to_regional_cache(
            model,
            sequences[contig],
            contig=contig,
            cache=cache,
            device=device,
            source=source,
        )
    cache.close()
    print("Cache ready:", args.cache_dir)


def dense_rank(values: np.ndarray) -> np.ndarray:
    _unique, inverse = np.unique(values, return_inverse=True)
    return inverse.astype(np.float64) + 1.0


def dense_spearman(prediction: np.ndarray, target: np.ndarray) -> float | None:
    if len(prediction) < 2:
        return None
    x = dense_rank(prediction)
    y = dense_rank(target)
    x -= x.mean()
    y -= y.mean()
    denominator = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if denominator == 0.0:
        return None
    return float(np.dot(x, y) / denominator)


def stage_evaluate_a(args) -> None:
    split, sequences = load_project_data()
    folds = fold_assignment(train_contigs(split), args.folds, args.fold_seed)
    held_out = folds[args.fold]
    cache = RegionalFeatureCache(args.cache_dir)
    missing = [contig for contig in held_out if not cache.has_contig(contig)]
    if missing:
        raise RuntimeError(f"cache misses held-out contigs: {missing}")
    generator = RegionalWindowGenerator(
        split,
        sequences,
        part="train",
        contigs=held_out,
        window_bp=8192,
        positive_window_fraction=0.0,
        seed=args.seed,
    )

    total_hist = np.zeros(100_001, dtype=np.int64)
    active_hist = np.zeros(100_001, dtype=np.int64)
    active_log_rate: list[np.ndarray] = []
    active_probability: list[np.ndarray] = []
    active_count: list[np.ndarray] = []
    empty_log_rate: list[np.ndarray] = []
    for (contig, strand), allowed in generator.allowed.items():
        log_rate = np.asarray(cache._array(contig, strand)[:, 0], dtype=np.float32)
        active = generator.active[(contig, strand)] & allowed
        empty = (~generator.active[(contig, strand)]) & allowed
        if args.score_link == "poisson":
            rate = np.exp(np.clip(log_rate, -20.0, 15.0))
            probability = -np.expm1(-rate)
        else:
            probability = 1.0 / (1.0 + np.exp(-np.clip(log_rate, -30.0, 30.0)))
        quantized = np.rint(np.clip(probability, 0.0, 1.0) * 100_000).astype(np.int32)
        total_hist += np.bincount(quantized[allowed], minlength=100_001)
        active_hist += np.bincount(quantized[active], minlength=100_001)
        active_log_rate.append(log_rate[active].copy())
        active_probability.append(probability[active].copy())
        active_count.append(generator.count[(contig, strand)][active].copy())
        empty_log_rate.append(log_rate[empty].copy())

    positives = int(active_hist.sum())
    pos_desc = active_hist[::-1]
    total_desc = total_hist[::-1]
    precision = np.cumsum(pos_desc) / np.maximum(np.cumsum(total_desc), 1)
    regional_ap = float(np.sum((pos_desc / max(positives, 1)) * precision))
    positive_log_rate = np.concatenate(active_log_rate)
    positive_probability = np.concatenate(active_probability)
    positive_count = np.concatenate(active_count)
    empty_score = np.concatenate(empty_log_rate)
    threshold = float(np.quantile(positive_log_rate, 0.005, method="lower"))
    quantized_positive = np.rint(np.clip(positive_probability, 0.0, 1.0) * 100_000)
    result = {
        "experiment_id": EXPERIMENT_ID,
        "evaluation": "single held-out train-contig fold",
        "score_link": args.score_link,
        "fold": args.fold,
        "held_out_contigs": held_out,
        "allowed_regions": int(total_hist.sum()),
        "active_regions": positives,
        "active_prevalence": positives / max(int(total_hist.sum()), 1),
        "regional_ap": regional_ap,
        "ap_lift_over_prevalence": regional_ap
        / max(positives / max(int(total_hist.sum()), 1), 1e-12),
        "positive_count_dense_spearman": dense_spearman(
            quantized_positive, positive_count
        ),
        "threshold_for_0p995_active_recall": threshold,
        "realized_active_recall": float(np.mean(positive_log_rate >= threshold)),
        "empty_region_suppression_fraction": float(np.mean(empty_score < threshold)),
        "cache": str(args.cache_dir),
    }
    print(json.dumps(result, indent=2))
    atomic_json(
        args.cache_dir / f"held_out_metrics_{args.score_link}_fold_{args.fold:02d}.json",
        result,
    )
    cache.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inspect", "smoke", "train-a", "cache-a", "evaluate-a"))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_RUN_DIR / "cache_fold0_final")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=490149)
    parser.add_argument("--fold-seed", type=int, default=490049)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--a-steps", type=int, default=20_000)
    parser.add_argument("--a-batch", type=int, default=2)
    parser.add_argument("--a-lr", type=float, default=1e-3)
    parser.add_argument("--a-positive-fraction", type=float, default=0.0)
    parser.add_argument("--lambda-rank", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--a-milestones", default="5000,10000,20000")
    parser.add_argument("--a-checkpoint-step", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-minutes", type=float, default=0.0)
    parser.add_argument("--overwrite-cache-contig", action="store_true")
    parser.add_argument("--score-link", choices=("poisson", "sigmoid"), default="poisson")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.run_dir = args.run_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    stages = {
        "inspect": stage_inspect,
        "smoke": stage_smoke,
        "train-a": stage_train_a,
        "cache-a": stage_cache_a,
        "evaluate-a": stage_evaluate_a,
    }
    stages[args.stage](args)


if __name__ == "__main__":
    main()
