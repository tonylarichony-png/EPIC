"""Streaming full-population evaluation for EPIC CNN profile models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Mapping
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
import torch

from .model import (
    CNNBaseline,
    CNNPresenceIntensity,
    create_cnn_baseline,
    create_cnn_presence_intensity,
)


def _dense_rank_correlation(counts: np.ndarray, scores: np.ndarray) -> float:
    """EPIC dense-rank correlation without NumPy's BLAS-backed corrcoef."""

    counts = np.asarray(counts)
    scores = np.asarray(scores)
    if counts.shape != scores.shape:
        raise ValueError("counts and scores must have identical shapes")
    if not np.all(np.isfinite(counts)) or not np.all(np.isfinite(scores)):
        raise ValueError("counts and scores must be finite")
    left = rankdata(counts, method="dense").astype(np.float64, copy=False)
    right = rankdata(scores, method="dense").astype(np.float64, copy=False)
    if len(left) < 2:
        return 0.0
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(
        np.sqrt(np.sum(left * left) * np.sum(right * right))
    )
    if denominator == 0.0:
        return 0.0
    correlation = float(np.sum(left * right) / denominator)
    return float(np.clip(correlation, -1.0, 1.0))
from .profile import (
    ProfileWindowConfig,
    ProfileWindowGenerator,
    forward_both_strands,
    forward_both_strands_multitask,
)


@dataclass(frozen=True)
class LoadedCNNCheckpoint:
    """A baseline model and its saved metadata, restored on the requested device."""

    model: CNNBaseline
    profile_config: ProfileWindowConfig
    checkpoint: Mapping[str, Any]


@dataclass(frozen=True)
class LoadedPresenceIntensityCheckpoint:
    """An EXP-002 model and its saved metadata on the requested device."""

    model: CNNPresenceIntensity
    profile_config: ProfileWindowConfig
    checkpoint: Mapping[str, Any]


@dataclass(frozen=True)
class ProfileEvaluationResult:
    """Metrics and resource diagnostics from one complete profile scan."""

    metrics: pd.DataFrame
    elapsed_seconds: float
    positions_per_second: float
    processed_tiles: int
    processed_positions: int
    processed_positives: int
    score_decimals: int
    fallback_warnings: tuple[str, ...]

    @property
    def overall(self) -> pd.Series:
        return self.metrics.loc[self.metrics["segment"] == "overall"].iloc[0]


class _MetricAccumulator:
    def __init__(self, score_levels: int) -> None:
        self.total_by_score = np.zeros(score_levels, dtype=np.int64)
        self.positive_by_score = np.zeros(score_levels, dtype=np.int64)
        self.positive_counts: list[np.ndarray] = []
        self.positive_raw_scores: list[np.ndarray] = []
        self.positive_quantized_scores: list[np.ndarray] = []

    def add(
        self,
        probabilities: np.ndarray,
        quantized_scores: np.ndarray,
        mask: np.ndarray,
        counts: np.ndarray,
    ) -> None:
        positive_mask = mask & (counts > 0)
        valid_scores = quantized_scores[mask]
        positive_scores = quantized_scores[positive_mask]
        self.total_by_score += np.bincount(
            valid_scores,
            minlength=len(self.total_by_score),
        )
        self.positive_by_score += np.bincount(
            positive_scores,
            minlength=len(self.positive_by_score),
        )
        if positive_scores.size:
            self.positive_counts.append(
                counts[positive_mask].astype(np.float32, copy=True)
            )
            self.positive_raw_scores.append(
                probabilities[positive_mask].astype(np.float32, copy=True)
            )
            self.positive_quantized_scores.append(
                positive_scores.astype(np.int32, copy=True)
            )

    def merge(self, others: list["_MetricAccumulator"]) -> None:
        for other in others:
            self.total_by_score += other.total_by_score
            self.positive_by_score += other.positive_by_score
            self.positive_counts.extend(other.positive_counts)
            self.positive_raw_scores.extend(other.positive_raw_scores)
            self.positive_quantized_scores.extend(
                other.positive_quantized_scores
            )

    def metrics(self, segment: str) -> dict[str, object]:
        population = int(self.total_by_score.sum())
        positives = int(self.positive_by_score.sum())
        prevalence = positives / population if population else 0.0

        total_desc = self.total_by_score[::-1]
        positive_desc = self.positive_by_score[::-1]
        cumulative_total = np.cumsum(total_desc, dtype=np.int64)
        cumulative_positive = np.cumsum(positive_desc, dtype=np.int64)
        precision = np.divide(
            cumulative_positive,
            cumulative_total,
            out=np.zeros_like(cumulative_positive, dtype=np.float64),
            where=cumulative_total > 0,
        )
        average_precision = (
            float(np.sum((positive_desc / positives) * precision))
            if positives
            else 0.0
        )

        if positives:
            positive_counts = np.concatenate(self.positive_counts)
            raw_scores = np.concatenate(self.positive_raw_scores)
            quantized_scores = np.concatenate(
                self.positive_quantized_scores
            )
            epic_spearman = _dense_rank_correlation(
                positive_counts,
                raw_scores,
            )
            quantized_spearman = _dense_rank_correlation(
                positive_counts,
                quantized_scores,
            )
        else:
            epic_spearman = 0.0
            quantized_spearman = 0.0

        return {
            "segment": segment,
            "position_strand": population,
            "positives": positives,
            "prevalence": prevalence,
            "average_precision": average_precision,
            "ap_lift": average_precision / prevalence if prevalence else 0.0,
            "epic_spearman": epic_spearman,
            "quantized_spearman": quantized_spearman,
        }


def load_cnn_baseline_checkpoint(
    path: str | Path,
    *,
    device: Any,
) -> LoadedCNNCheckpoint:
    """Restore a trusted CNN baseline checkpoint without allocating it on GPU first."""

    checkpoint = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=False,
    )
    required = {"model_state_dict", "profile_config"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint misses fields: {sorted(missing)}")

    model = create_cnn_baseline()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    config = ProfileWindowConfig(**checkpoint["profile_config"])
    return LoadedCNNCheckpoint(model, config, checkpoint)


def load_cnn_presence_intensity_checkpoint(
    path: str | Path,
    *,
    device: Any,
) -> LoadedPresenceIntensityCheckpoint:
    """Restore a trusted EXP-002 presence/intensity checkpoint."""

    checkpoint = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=False,
    )
    required = {"model_state_dict", "profile_config"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint misses fields: {sorted(missing)}")
    model = create_cnn_presence_intensity()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    config = ProfileWindowConfig(**checkpoint["profile_config"])
    return LoadedPresenceIntensityCheckpoint(model, config, checkpoint)


def evaluate_profile_model(
    model: torch.nn.Module,
    generator: ProfileWindowGenerator,
    *,
    device: Any,
    batch_size: int = 8,
    score_decimals: int = 5,
    progress_every_batches: int = 500,
    progress: Callable[[str], None] | None = print,
    fail_on_cpu_fallback: bool = True,
) -> ProfileEvaluationResult:
    """Evaluate every allowed position-strand exactly once.

    AP is accumulated as an exact histogram after decimal score quantization.
    Raw positive scores are retained only for EPIC dense-rank Spearman, so the
    full validation population never has to reside in memory.
    """

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(score_decimals, int) or not 1 <= score_decimals <= 6:
        raise ValueError("score_decimals must be an integer from 1 to 6")
    if not isinstance(progress_every_batches, int) or progress_every_batches <= 0:
        raise ValueError("progress_every_batches must be positive")

    tiles = generator.tile_index
    contigs = tuple(str(value) for value in tiles["contig"].unique())
    score_scale = 10**score_decimals
    score_levels = score_scale + 1
    by_contig = {
        contig: _MetricAccumulator(score_levels)
        for contig in contigs
    }
    processed_positions = 0
    processed_positives = 0
    started_at = time.perf_counter()
    was_training = model.training

    try:
        model.eval()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.no_grad():
                for batch_number, start in enumerate(
                    range(0, len(tiles), batch_size),
                    start=1,
                ):
                    selected = tiles.iloc[start : start + batch_size]
                    batch = generator.make_batch(selected)
                    current_batch_size = len(selected)
                    x = batch.x_both_strands.to(device)
                    plus_logits, minus_reverse_logits = forward_both_strands(
                        model,
                        x,
                        batch_size=current_batch_size,
                        context_flank=generator.config.context_flank,
                        target_length=generator.config.target_length,
                    )

                    plus = plus_logits.detach().cpu().numpy()[:, 0, :]
                    minus = (
                        minus_reverse_logits.detach().cpu().numpy()[:, 0, ::-1]
                    )
                    probabilities = expit(np.stack((plus, minus), axis=1))
                    quantized = np.rint(
                        np.clip(probabilities, 0.0, 1.0) * score_scale
                    ).astype(np.int32)
                    mask = batch.mask.numpy()
                    counts = batch.count.numpy()

                    selected_contigs = selected["contig"].astype(str).to_numpy()
                    for contig in np.unique(selected_contigs):
                        rows = np.flatnonzero(selected_contigs == contig)
                        by_contig[contig].add(
                            probabilities[rows],
                            quantized[rows],
                            mask[rows],
                            counts[rows],
                        )

                    processed_positions += int(mask.sum())
                    processed_positives += int((mask & (counts > 0)).sum())
                    finished = start + current_batch_size >= len(tiles)
                    if progress is not None and (
                        batch_number == 1
                        or batch_number % progress_every_batches == 0
                        or finished
                    ):
                        elapsed = time.perf_counter() - started_at
                        progress(
                            f"tiles={start + current_batch_size:>6}/{len(tiles)} "
                            f"positions={processed_positions:>9} "
                            f"positions/s={processed_positions / elapsed:,.0f}"
                        )
        fallback_warnings = tuple(
            str(item.message)
            for item in caught
            if "fall back" in str(item.message).lower()
        )
    finally:
        model.train(was_training)

    if fallback_warnings and fail_on_cpu_fallback:
        raise RuntimeError(
            "DirectML CPU fallback detected: " + " | ".join(fallback_warnings)
        )

    expected_positions = int(tiles["allowed_position_strand"].sum())
    expected_positives = int(tiles["positive_positions"].sum())
    if processed_positions != expected_positions:
        raise AssertionError(
            f"Processed {processed_positions} positions, expected {expected_positions}"
        )
    if processed_positives != expected_positives:
        raise AssertionError(
            f"Processed {processed_positives} positives, expected {expected_positives}"
        )

    overall = _MetricAccumulator(score_levels)
    overall.merge(list(by_contig.values()))
    metric_rows = [overall.metrics("overall")]
    metric_rows.extend(
        by_contig[contig].metrics(contig)
        for contig in contigs
    )
    metrics = pd.DataFrame(metric_rows)
    elapsed_seconds = time.perf_counter() - started_at
    return ProfileEvaluationResult(
        metrics=metrics,
        elapsed_seconds=elapsed_seconds,
        positions_per_second=processed_positions / elapsed_seconds,
        processed_tiles=len(tiles),
        processed_positions=processed_positions,
        processed_positives=processed_positives,
        score_decimals=score_decimals,
        fallback_warnings=fallback_warnings,
    )


def evaluate_presence_intensity_model(
    model: torch.nn.Module,
    generator: ProfileWindowGenerator,
    *,
    device: Any,
    batch_size: int = 8,
    score_decimals: int = 5,
    progress_every_batches: int = 500,
    progress: Callable[[str], None] | None = print,
    fail_on_cpu_fallback: bool = True,
) -> ProfileEvaluationResult:
    """Evaluate presence, intensity and their fixed expected-signal combination."""

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(score_decimals, int) or not 1 <= score_decimals <= 6:
        raise ValueError("score_decimals must be an integer from 1 to 6")
    if not isinstance(progress_every_batches, int) or progress_every_batches <= 0:
        raise ValueError("progress_every_batches must be positive")

    score_names = (
        "presence_probability",
        "intensity_prediction",
        "expected_signal",
    )
    tiles = generator.tile_index
    contigs = tuple(str(value) for value in tiles["contig"].unique())
    score_scale = 10**score_decimals
    score_levels = score_scale + 1
    accumulators = {
        score_name: {
            contig: _MetricAccumulator(score_levels)
            for contig in contigs
        }
        for score_name in score_names
    }
    processed_positions = 0
    processed_positives = 0
    started_at = time.perf_counter()
    was_training = model.training

    try:
        model.eval()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.no_grad():
                for batch_number, start in enumerate(
                    range(0, len(tiles), batch_size),
                    start=1,
                ):
                    selected = tiles.iloc[start : start + batch_size]
                    batch = generator.make_batch(selected)
                    current_batch_size = len(selected)
                    outputs = forward_both_strands_multitask(
                        model,
                        batch.x_both_strands.to(device),
                        batch_size=current_batch_size,
                        context_flank=generator.config.context_flank,
                        target_length=generator.config.target_length,
                    )
                    plus_presence, minus_presence_reverse = outputs[:2]
                    plus_intensity, minus_intensity_reverse = outputs[2:]

                    presence_logits = np.stack(
                        (
                            plus_presence.detach().cpu().numpy()[:, 0, :],
                            minus_presence_reverse.detach().cpu().numpy()[
                                :, 0, ::-1
                            ],
                        ),
                        axis=1,
                    )
                    intensity_raw = np.stack(
                        (
                            plus_intensity.detach().cpu().numpy()[:, 0, :],
                            minus_intensity_reverse.detach().cpu().numpy()[
                                :, 0, ::-1
                            ],
                        ),
                        axis=1,
                    )
                    presence_probability = expit(presence_logits)
                    intensity_score = expit(intensity_raw)
                    expected_raw = presence_probability * np.expm1(
                        np.clip(intensity_raw, 0.0, 20.0)
                    )
                    expected_score = expected_raw / (1.0 + expected_raw)
                    scores = {
                        "presence_probability": presence_probability,
                        "intensity_prediction": intensity_score,
                        "expected_signal": expected_score,
                    }
                    mask = batch.mask.numpy()
                    counts = batch.count.numpy()
                    selected_contigs = selected["contig"].astype(str).to_numpy()

                    for score_name, score in scores.items():
                        quantized = np.rint(
                            np.clip(score, 0.0, 1.0) * score_scale
                        ).astype(np.int32)
                        for contig in np.unique(selected_contigs):
                            rows = np.flatnonzero(selected_contigs == contig)
                            accumulators[score_name][contig].add(
                                score[rows],
                                quantized[rows],
                                mask[rows],
                                counts[rows],
                            )

                    processed_positions += int(mask.sum())
                    processed_positives += int((mask & (counts > 0)).sum())
                    finished = start + current_batch_size >= len(tiles)
                    if progress is not None and (
                        batch_number == 1
                        or batch_number % progress_every_batches == 0
                        or finished
                    ):
                        elapsed = time.perf_counter() - started_at
                        progress(
                            f"tiles={start + current_batch_size:>6}/{len(tiles)} "
                            f"positions={processed_positions:>9} "
                            f"positions/s={processed_positions / elapsed:,.0f}"
                        )
        fallback_warnings = tuple(
            str(item.message)
            for item in caught
            if "fall back" in str(item.message).lower()
        )
    finally:
        model.train(was_training)

    if fallback_warnings and fail_on_cpu_fallback:
        raise RuntimeError(
            "DirectML CPU fallback detected: " + " | ".join(fallback_warnings)
        )
    expected_positions = int(tiles["allowed_position_strand"].sum())
    expected_positives = int(tiles["positive_positions"].sum())
    if processed_positions != expected_positions:
        raise AssertionError(
            f"Processed {processed_positions} positions, expected {expected_positions}"
        )
    if processed_positives != expected_positives:
        raise AssertionError(
            f"Processed {processed_positives} positives, expected {expected_positives}"
        )

    rows: list[dict[str, object]] = []
    for score_name in score_names:
        by_contig = accumulators[score_name]
        overall = _MetricAccumulator(score_levels)
        overall.merge(list(by_contig.values()))
        overall_row = overall.metrics("overall")
        overall_row["score"] = score_name
        rows.append(overall_row)
        for contig in contigs:
            row = by_contig[contig].metrics(contig)
            row["score"] = score_name
            rows.append(row)

    column_order = [
        "score",
        "segment",
        "position_strand",
        "positives",
        "prevalence",
        "average_precision",
        "ap_lift",
        "epic_spearman",
        "quantized_spearman",
    ]
    elapsed_seconds = time.perf_counter() - started_at
    return ProfileEvaluationResult(
        metrics=pd.DataFrame(rows)[column_order],
        elapsed_seconds=elapsed_seconds,
        positions_per_second=processed_positions / elapsed_seconds,
        processed_tiles=len(tiles),
        processed_positions=processed_positions,
        processed_positives=processed_positives,
        score_decimals=score_decimals,
        fallback_warnings=fallback_warnings,
    )


__all__ = [
    "LoadedCNNCheckpoint",
    "LoadedPresenceIntensityCheckpoint",
    "ProfileEvaluationResult",
    "evaluate_profile_model",
    "evaluate_presence_intensity_model",
    "load_cnn_baseline_checkpoint",
    "load_cnn_presence_intensity_checkpoint",
]
