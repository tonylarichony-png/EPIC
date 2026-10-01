"""Profile-window sampling for the EPIC CNN.

The training population is partitioned into fixed, non-overlapping genomic
target tiles.  Every sampled tile receives real sequence flanks, a two-strand
target profile, a whitelist mask, and importance weights for the class-balanced
population risk.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..epic_data import PreparedSplit
from ..sequence_context import UPPERCASE_BASE_CODE


@dataclass(frozen=True)
class ProfileWindowConfig:
    """Geometry and sampling mixture for profile training."""

    target_length: int = 1024
    context_flank: int = 130
    positive_branch_fraction: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.target_length, int) or self.target_length <= 0:
            raise ValueError("target_length должен быть положительным целым числом")
        if not isinstance(self.context_flank, int) or self.context_flank < 0:
            raise ValueError("context_flank должен быть неотрицательным целым числом")
        if not 0 < self.positive_branch_fraction < 1:
            raise ValueError("positive_branch_fraction должен находиться между 0 и 1")

    @property
    def input_length(self) -> int:
        return self.target_length + 2 * self.context_flank


@dataclass(frozen=True)
class ProfileBatch:
    """One CPU profile batch ready for transfer to an accelerator."""

    x: torch.Tensor
    x_reverse_complement: torch.Tensor
    x_both_strands: torch.Tensor
    raw_codes: torch.Tensor
    sequence_valid: torch.Tensor
    target: torch.Tensor
    count: torch.Tensor
    mask: torch.Tensor
    loss_weight: torch.Tensor
    plus_target: torch.Tensor
    minus_target_reverse: torch.Tensor
    plus_loss_weight: torch.Tensor
    minus_loss_weight_reverse: torch.Tensor
    sampling_probability: torch.Tensor
    tile_id: torch.Tensor
    sampling_branch: tuple[str, ...]
    tiles: pd.DataFrame


def build_profile_tile_index(
    prepared_split: PreparedSplit,
    part: str,
    config: ProfileWindowConfig = ProfileWindowConfig(),
) -> pd.DataFrame:
    """Partition one split into fixed genomic target tiles."""

    intervals = prepared_split.intervals(part)
    positives = prepared_split.positive_targets(part)
    if positives is None or positives.empty:
        raise ValueError(f"Для {part!r} отсутствуют положительные targets")

    plus = intervals.loc[
        intervals["strand"] == "+",
        ["contig", "start", "end"],
    ]
    if plus.empty:
        raise ValueError(f"Для {part!r} отсутствуют интервалы plus-цепи")

    target_length = config.target_length
    overlap_records: list[tuple[str, int, int]] = []
    for interval in plus.itertuples(index=False):
        interval_start = int(interval.start)
        interval_end = int(interval.end)
        first_tile = (interval_start // target_length) * target_length
        for target_start in range(first_tile, interval_end, target_length):
            overlap_start = max(interval_start, target_start)
            overlap_end = min(interval_end, target_start + target_length)
            if overlap_end > overlap_start:
                overlap_records.append(
                    (str(interval.contig), target_start, overlap_end - overlap_start)
                )

    tile_index = pd.DataFrame.from_records(
        overlap_records,
        columns=["contig", "target_start", "allowed_bp"],
    )
    tile_index = (
        tile_index.groupby(["contig", "target_start"], as_index=False, sort=False)
        .agg(allowed_bp=("allowed_bp", "sum"))
    )

    positive_rows = positives[
        ["contig", "coordinate_0based", "count"]
    ].copy()
    positive_rows["target_start"] = (
        positive_rows["coordinate_0based"] // target_length * target_length
    )
    positive_summary = (
        positive_rows.groupby(["contig", "target_start"], as_index=False, sort=False)
        .agg(
            positive_positions=("count", "size"),
            count_sum=("count", "sum"),
        )
    )
    tile_index = tile_index.merge(
        positive_summary,
        on=["contig", "target_start"],
        how="left",
        validate="one_to_one",
    )
    columns = ["positive_positions", "count_sum"]
    tile_index[columns] = tile_index[columns].fillna(0).astype("int64")
    tile_index["target_end"] = tile_index["target_start"] + target_length
    tile_index["allowed_position_strand"] = 2 * tile_index["allowed_bp"]
    tile_index["negative_positions"] = (
        tile_index["allowed_position_strand"] - tile_index["positive_positions"]
    )
    tile_index = tile_index.sort_values(
        ["contig", "target_start"], kind="stable"
    ).reset_index(drop=True)
    tile_index.insert(0, "tile_id", np.arange(len(tile_index), dtype=np.int64))

    if int(tile_index["positive_positions"].sum()) != len(positives):
        raise AssertionError("Не все положительные targets попали в tile index")
    if (tile_index["negative_positions"] < 0).any():
        raise AssertionError("Число отрицательных позиций не может быть отрицательным")
    expected_population = int(intervals["length_bp"].sum())
    if int(tile_index["allowed_position_strand"].sum()) != expected_population:
        raise AssertionError("Tile index не покрывает исходную position-strand популяцию")

    total_positive = int(tile_index["positive_positions"].sum())
    total_negative = int(tile_index["negative_positions"].sum())
    tile_index["q_positive"] = tile_index["positive_positions"] / total_positive
    tile_index["q_negative"] = tile_index["negative_positions"] / total_negative
    alpha = config.positive_branch_fraction
    tile_index["q_mixture"] = (
        alpha * tile_index["q_positive"]
        + (1.0 - alpha) * tile_index["q_negative"]
    )
    for column in ("q_positive", "q_negative", "q_mixture"):
        if not np.isclose(float(tile_index[column].sum()), 1.0):
            raise AssertionError(f"Вероятности {column} не суммируются в единицу")
    return tile_index


class ProfileWindowGenerator:
    """Sample and materialize fixed profile tiles without dense genome targets."""

    def __init__(
        self,
        prepared_split: PreparedSplit,
        sequences: Mapping[str, np.ndarray],
        *,
        part: str = "train",
        config: ProfileWindowConfig = ProfileWindowConfig(),
        seed: int = 42,
    ) -> None:
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed должен быть неотрицательным целым числом")
        self.prepared_split = prepared_split
        self.sequences = sequences
        self.part = part
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.tile_index = build_profile_tile_index(prepared_split, part, config)
        self.total_positive = int(self.tile_index["positive_positions"].sum())
        self.total_negative = int(self.tile_index["negative_positions"].sum())
        self._intervals = self._build_interval_lookup()
        self._positives = self._build_positive_lookup()
        self._validate_sequences()

    def _validate_sequences(self) -> None:
        for contig in self.tile_index["contig"].unique():
            if contig not in self.sequences:
                raise KeyError(f"Contig отсутствует в FASTA: {contig}")
            sequence = np.asarray(self.sequences[contig])
            if sequence.ndim != 1 or not len(sequence):
                raise ValueError(f"Некорректная FASTA-последовательность: {contig}")
            if not np.issubdtype(sequence.dtype, np.integer):
                raise TypeError(f"FASTA-коды должны быть целыми: {contig}")
            if np.any(sequence < 0) or np.any(sequence >= len(UPPERCASE_BASE_CODE)):
                raise ValueError(f"Неизвестный FASTA-код: {contig}")

    def _build_interval_lookup(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        intervals = self.prepared_split.intervals(self.part)
        intervals = intervals.loc[
            intervals["strand"] == "+", ["contig", "start", "end"]
        ].sort_values(["contig", "start"], kind="stable")
        return {
            str(contig): (
                frame["start"].to_numpy(dtype=np.int64, copy=True),
                frame["end"].to_numpy(dtype=np.int64, copy=True),
            )
            for contig, frame in intervals.groupby("contig", sort=False)
        }

    def _build_positive_lookup(
        self,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        positives = self.prepared_split.positive_targets(self.part)
        if positives is None:
            raise ValueError(f"Для {self.part!r} targets недоступны")
        positives = positives.sort_values(
            ["contig", "coordinate_0based", "strand"], kind="stable"
        )
        return {
            str(contig): (
                frame["coordinate_0based"].to_numpy(dtype=np.int64, copy=True),
                frame["strand"].to_numpy(copy=True),
                frame["count"].to_numpy(dtype=np.float32, copy=True),
            )
            for contig, frame in positives.groupby("contig", sort=False)
        }

    def sample_tiles(self, batch_size: int) -> pd.DataFrame:
        """Draw a fixed-count stratified batch and retain its mixture probability."""

        if not isinstance(batch_size, int) or batch_size <= 1:
            raise ValueError("batch_size должен быть целым числом больше единицы")
        alpha = self.config.positive_branch_fraction
        positive_draws = int(round(batch_size * alpha))
        if not np.isclose(positive_draws / batch_size, alpha):
            raise ValueError(
                "batch_size должен точно реализовывать positive_branch_fraction"
            )
        negative_draws = batch_size - positive_draws
        if not positive_draws or not negative_draws:
            raise ValueError("В батче должны присутствовать обе ветви sampling")

        positive_ids = self.rng.choice(
            len(self.tile_index),
            size=positive_draws,
            replace=True,
            p=self.tile_index["q_positive"].to_numpy(copy=True),
        )
        negative_ids = self.rng.choice(
            len(self.tile_index),
            size=negative_draws,
            replace=True,
            p=self.tile_index["q_negative"].to_numpy(copy=True),
        )
        tile_ids = np.concatenate((positive_ids, negative_ids))
        branches = np.array(
            ["positive"] * positive_draws + ["negative"] * negative_draws,
            dtype=object,
        )
        order = self.rng.permutation(batch_size)
        selected = self.tile_index.iloc[tile_ids[order]].copy().reset_index(drop=True)
        selected["sampling_branch"] = branches[order]
        return selected

    def sample_batch(self, batch_size: int) -> ProfileBatch:
        """Draw tiles and materialize one CPU batch."""

        return self.make_batch(self.sample_tiles(batch_size))

    def make_batch(self, selected_tiles: pd.DataFrame) -> ProfileBatch:
        """Materialize explicitly selected tile rows."""

        required = {
            "tile_id",
            "contig",
            "target_start",
            "allowed_position_strand",
            "positive_positions",
            "q_mixture",
        }
        missing = required - set(selected_tiles.columns)
        if missing:
            raise ValueError(f"В selected_tiles отсутствуют колонки: {sorted(missing)}")
        if selected_tiles.empty:
            raise ValueError("selected_tiles не должен быть пустым")

        batch_size = len(selected_tiles)
        target_length = self.config.target_length
        flank = self.config.context_flank
        input_length = self.config.input_length
        raw_codes = np.full((batch_size, input_length), 8, dtype=np.uint8)
        sequence_valid = np.zeros((batch_size, input_length), dtype=bool)
        counts = np.zeros((batch_size, 2, target_length), dtype=np.float32)
        target_mask = np.zeros((batch_size, 2, target_length), dtype=bool)

        for batch_index, tile in enumerate(selected_tiles.itertuples(index=False)):
            contig = str(tile.contig)
            target_start = int(tile.target_start)
            target_end = target_start + target_length
            input_start = target_start - flank
            input_end = target_end + flank
            contig_sequence = np.asarray(self.sequences[contig])
            source_start = max(input_start, 0)
            source_end = min(input_end, len(contig_sequence))
            destination_start = source_start - input_start
            destination_end = destination_start + source_end - source_start
            raw_codes[batch_index, destination_start:destination_end] = (
                contig_sequence[source_start:source_end]
            )
            sequence_valid[batch_index, destination_start:destination_end] = True

            interval_starts, interval_ends = self._intervals[contig]
            first_interval = int(
                np.searchsorted(interval_ends, target_start, side="right")
            )
            last_interval = int(
                np.searchsorted(interval_starts, target_end, side="left")
            )
            for interval_index in range(first_interval, last_interval):
                overlap_start = max(target_start, int(interval_starts[interval_index]))
                overlap_end = min(target_end, int(interval_ends[interval_index]))
                if overlap_end > overlap_start:
                    local_start = overlap_start - target_start
                    local_end = overlap_end - target_start
                    target_mask[batch_index, :, local_start:local_end] = True

            positive_data = self._positives.get(contig)
            if positive_data is not None:
                coordinates, strands, positive_counts = positive_data
                first_positive = int(
                    np.searchsorted(coordinates, target_start, side="left")
                )
                last_positive = int(
                    np.searchsorted(coordinates, target_end, side="left")
                )
                for positive_index in range(first_positive, last_positive):
                    local_position = int(coordinates[positive_index]) - target_start
                    strand_index = 0 if strands[positive_index] == "+" else 1
                    if counts[batch_index, strand_index, local_position] != 0:
                        raise AssertionError("Повтор position-strand среди positive targets")
                    counts[batch_index, strand_index, local_position] = positive_counts[
                        positive_index
                    ]

        uppercase_codes = UPPERCASE_BASE_CODE[raw_codes]
        x = (
            F.one_hot(torch.from_numpy(uppercase_codes).long(), num_classes=5)
            .permute(0, 2, 1)
            .float()
            .contiguous()
        )
        x_reverse_complement = x[:, [3, 2, 1, 0, 4], :].flip(-1).contiguous()
        # Concatenate on CPU.  torch-directml 0.2.5 can terminate the process
        # when a DML-side cat result is consumed by this Conv1d stack.
        x_both_strands = torch.cat((x, x_reverse_complement), dim=0).contiguous()
        count_tensor = torch.from_numpy(counts)
        target_tensor = (count_tensor > 0).float()
        mask_tensor = torch.from_numpy(target_mask)

        if not torch.all(target_tensor.bool() <= mask_tensor):
            raise AssertionError("Положительный target оказался вне whitelist mask")
        if int(mask_tensor.sum()) != int(
            selected_tiles["allowed_position_strand"].sum()
        ):
            raise AssertionError("Whitelist mask не совпадает с tile index")
        if int(target_tensor.sum()) != int(selected_tiles["positive_positions"].sum()):
            raise AssertionError("Positive profile не совпадает с tile index")

        probabilities = selected_tiles["q_mixture"].to_numpy(
            dtype=np.float32, copy=True
        )
        q = probabilities[:, None, None]
        positive_weight = 0.5 / (batch_size * q * self.total_positive)
        negative_weight = 0.5 / (batch_size * q * self.total_negative)
        loss_weight = np.where(counts > 0, positive_weight, negative_weight)
        loss_weight = np.where(target_mask, loss_weight, 0.0).astype(
            np.float32, copy=False
        )
        branches = (
            tuple(selected_tiles["sampling_branch"].astype(str))
            if "sampling_branch" in selected_tiles
            else tuple("explicit" for _ in range(batch_size))
        )
        return ProfileBatch(
            x=x,
            x_reverse_complement=x_reverse_complement,
            x_both_strands=x_both_strands,
            raw_codes=torch.from_numpy(raw_codes),
            sequence_valid=torch.from_numpy(sequence_valid),
            target=target_tensor,
            count=count_tensor,
            mask=mask_tensor,
            loss_weight=torch.from_numpy(loss_weight),
            plus_target=target_tensor[:, 0:1, :].contiguous(),
            minus_target_reverse=target_tensor[:, 1:2, :]
            .flip(-1)
            .contiguous(),
            plus_loss_weight=torch.from_numpy(
                loss_weight[:, 0:1, :].copy()
            ),
            minus_loss_weight_reverse=torch.from_numpy(
                loss_weight[:, 1:2, ::-1].copy()
            ),
            sampling_probability=torch.from_numpy(probabilities),
            tile_id=torch.from_numpy(
                selected_tiles["tile_id"].to_numpy(dtype=np.int64, copy=True)
            ),
            sampling_branch=branches,
            tiles=selected_tiles.copy(),
        )


def forward_both_strands(
    model: torch.nn.Module,
    x_both_strands: torch.Tensor,
    *,
    batch_size: int,
    context_flank: int,
    target_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict plus/minus profiles with shared weights and shared BN statistics."""

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size должен быть положительным целым числом")
    if x_both_strands.ndim != 3 or x_both_strands.shape[0] != 2 * batch_size:
        raise ValueError("x_both_strands должен иметь форму [2B,C,L]")
    expected_length = target_length + 2 * context_flank
    if x_both_strands.shape[-1] != expected_length:
        raise ValueError(
            "Длина входа должна быть "
            f"{expected_length}, получено {x_both_strands.shape[-1]}"
        )
    combined_logits = model(x_both_strands)
    if combined_logits.shape != (2 * batch_size, 1, expected_length):
        raise ValueError("Модель должна вернуть logits формы [2B,1,L]")
    plus_full = combined_logits[:batch_size]
    minus_reverse_full = combined_logits[batch_size:]
    output_slice = slice(context_flank, context_flank + target_length)
    return (
        plus_full[:, :, output_slice],
        minus_reverse_full[:, :, output_slice],
    )


def forward_both_strands_multitask(
    model: torch.nn.Module,
    x_both_strands: torch.Tensor,
    *,
    batch_size: int,
    context_flank: int,
    target_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align presence and intensity outputs for plus and RC-minus inputs."""

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size должен быть положительным целым числом")
    if x_both_strands.ndim != 3 or x_both_strands.shape[0] != 2 * batch_size:
        raise ValueError("x_both_strands должен иметь форму [2B,C,L]")
    expected_length = target_length + 2 * context_flank
    if x_both_strands.shape[-1] != expected_length:
        raise ValueError(
            "Длина входа должна быть "
            f"{expected_length}, получено {x_both_strands.shape[-1]}"
        )
    presence_all, intensity_all = model(x_both_strands)
    expected_shape = (2 * batch_size, 1, expected_length)
    if presence_all.shape != expected_shape or intensity_all.shape != expected_shape:
        raise ValueError("Обе головы модели должны вернуть тензоры [2B,1,L]")

    output_slice = slice(context_flank, context_flank + target_length)
    return (
        presence_all[:batch_size, :, output_slice],
        presence_all[batch_size:, :, output_slice],
        intensity_all[:batch_size, :, output_slice],
        intensity_all[batch_size:, :, output_slice],
    )


__all__ = [
    "ProfileBatch",
    "ProfileWindowConfig",
    "ProfileWindowGenerator",
    "build_profile_tile_index",
    "forward_both_strands",
    "forward_both_strands_multitask",
]
