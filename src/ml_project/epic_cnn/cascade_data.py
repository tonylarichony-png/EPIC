"""Data and cache utilities for the CNN-EXP-049 coarse-to-fine cascade."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .cascade import (
    REGION_SIZE,
    RegionalDetectorA,
    encode_raw_dna_six_channel,
    reverse_complement_six_channel,
)


def _safe_file_stem(value: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "contig"
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"{readable}.{digest}"


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@dataclass(frozen=True)
class RegionalBatch:
    """One strand-oriented dense R16 training batch for Network A."""

    raw_codes: torch.Tensor
    target: torch.Tensor
    count: torch.Tensor
    mask: torch.Tensor
    contigs: tuple[str, ...]
    strands: tuple[str, ...]
    start_regions: torch.Tensor


class RegionalWindowGenerator:
    """Sample exact, genomic-coordinate-aligned R16 windows.

    The generator uses true labels only for Network-A training targets and
    positive-window sampling.  It never creates features for Network B.
    """

    def __init__(
        self,
        split,
        sequences: Mapping[str, np.ndarray],
        *,
        part: str = "train",
        contigs: Sequence[str] | None = None,
        window_bp: int = 8192,
        positive_window_fraction: float = 0.5,
        seed: int = 490049,
    ) -> None:
        if window_bp <= 0 or window_bp % REGION_SIZE:
            raise ValueError("window_bp must be a positive multiple of 16")
        if not 0.0 <= positive_window_fraction <= 1.0:
            raise ValueError("positive_window_fraction must be in [0, 1]")
        self.window_bp = int(window_bp)
        self.window_regions = self.window_bp // REGION_SIZE
        self.positive_window_fraction = float(positive_window_fraction)
        self.rng = np.random.default_rng(seed)

        selected = set(map(str, contigs)) if contigs is not None else None
        intervals = split.intervals(part).copy()
        positives = split.positive_targets(part)
        if positives is None:
            raise ValueError(f"part {part!r} has no labels")
        positives = positives.copy()
        if selected is not None:
            intervals = intervals[intervals["contig"].astype(str).isin(selected)]
            positives = positives[positives["contig"].astype(str).isin(selected)]
        if intervals.empty:
            raise ValueError("no intervals remain for RegionalWindowGenerator")

        self.sequences = {str(name): np.asarray(value) for name, value in sequences.items()}
        self.allowed: dict[tuple[str, str], np.ndarray] = {}
        self.active: dict[tuple[str, str], np.ndarray] = {}
        self.count: dict[tuple[str, str], np.ndarray] = {}

        pair_rows = intervals[["contig", "strand"]].drop_duplicates()
        for row in pair_rows.itertuples(index=False):
            contig = str(row.contig)
            strand = str(row.strand)
            if contig not in self.sequences:
                raise KeyError(f"contig {contig!r} is absent from FASTA")
            region_count = (len(self.sequences[contig]) + REGION_SIZE - 1) // REGION_SIZE
            self.allowed[(contig, strand)] = np.zeros(region_count, dtype=bool)
            self.active[(contig, strand)] = np.zeros(region_count, dtype=bool)
            self.count[(contig, strand)] = np.zeros(region_count, dtype=np.float32)

        for row in intervals.itertuples(index=False):
            key = (str(row.contig), str(row.strand))
            first = int(row.start) // REGION_SIZE
            last = (int(row.end) - 1) // REGION_SIZE
            self.allowed[key][first : last + 1] = True

        for row in positives.itertuples(index=False):
            key = (str(row.contig), str(row.strand))
            if key not in self.active:
                continue
            region = int(row.coordinate_0based) // REGION_SIZE
            self.active[key][region] = True
            self.count[key][region] += float(row.count)

        self.allowed_indices = {
            key: np.flatnonzero(mask) for key, mask in self.allowed.items()
        }
        self.active_items = np.asarray(
            [
                (contig, strand, int(region))
                for (contig, strand), mask in self.active.items()
                for region in np.flatnonzero(mask)
            ],
            dtype=object,
        )
        if not len(self.active_items):
            raise ValueError("no active R16 regions remain")

        self.pairs = tuple(self.allowed_indices)
        counts = np.asarray(
            [len(self.allowed_indices[key]) for key in self.pairs], dtype=np.float64
        )
        if not np.all(counts > 0):
            raise ValueError("a contig/strand pair has no allowed R16 regions")
        self.pair_probability = counts / counts.sum()

    @property
    def active_region_count(self) -> int:
        return int(sum(mask.sum() for mask in self.active.values()))

    @property
    def allowed_region_count(self) -> int:
        return int(sum(mask.sum() for mask in self.allowed.values()))

    def _sample_anchor(self, positive: bool) -> tuple[str, str, int]:
        if positive:
            item = self.active_items[self.rng.integers(len(self.active_items))]
            return str(item[0]), str(item[1]), int(item[2])
        pair_index = int(self.rng.choice(len(self.pairs), p=self.pair_probability))
        key = self.pairs[pair_index]
        indices = self.allowed_indices[key]
        region = int(indices[self.rng.integers(len(indices))])
        return key[0], key[1], region

    def sample_batch(self, batch_size: int) -> RegionalBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        raw = np.full((batch_size, self.window_bp), 8, dtype=np.uint8)
        target = np.zeros((batch_size, 1, self.window_regions), dtype=np.float32)
        count = np.zeros((batch_size, 1, self.window_regions), dtype=np.float32)
        mask = np.zeros((batch_size, 1, self.window_regions), dtype=bool)
        start_regions = np.zeros(batch_size, dtype=np.int64)
        contigs: list[str] = []
        strands: list[str] = []

        for batch_index in range(batch_size):
            positive = bool(self.rng.random() < self.positive_window_fraction)
            contig, strand, anchor = self._sample_anchor(positive)
            total_regions = len(self.allowed[(contig, strand)])
            random_offset = int(self.rng.integers(self.window_regions))
            start_region = anchor - random_offset
            maximum_start = max(total_regions - self.window_regions, 0)
            start_region = min(max(start_region, 0), maximum_start)
            end_region = start_region + self.window_regions

            allowed_slice = self.allowed[(contig, strand)][start_region:end_region]
            active_slice = self.active[(contig, strand)][start_region:end_region]
            count_slice = self.count[(contig, strand)][start_region:end_region]
            mask[batch_index, 0, : len(allowed_slice)] = allowed_slice
            target[batch_index, 0, : len(active_slice)] = active_slice
            count[batch_index, 0, : len(count_slice)] = count_slice

            start_bp = start_region * REGION_SIZE
            sequence = self.sequences[contig]
            available = max(min(len(sequence) - start_bp, self.window_bp), 0)
            if available:
                raw[batch_index, :available] = sequence[start_bp : start_bp + available]

            if strand == "-":
                target[batch_index] = target[batch_index, :, ::-1].copy()
                count[batch_index] = count[batch_index, :, ::-1].copy()
                mask[batch_index] = mask[batch_index, :, ::-1].copy()

            start_regions[batch_index] = start_region
            contigs.append(contig)
            strands.append(strand)

        return RegionalBatch(
            raw_codes=torch.from_numpy(raw),
            target=torch.from_numpy(target),
            count=torch.from_numpy(count),
            mask=torch.from_numpy(mask),
            contigs=tuple(contigs),
            strands=tuple(strands),
            start_regions=torch.from_numpy(start_regions),
        )

    @staticmethod
    def encode_oriented(batch: RegionalBatch, device: torch.device) -> torch.Tensor:
        raw = batch.raw_codes.to(device, non_blocking=True)
        dna = encode_raw_dna_six_channel(raw)
        minus = torch.tensor(
            [strand == "-" for strand in batch.strands],
            dtype=torch.bool,
            device=device,
        )
        if bool(minus.any()):
            dna = dna.clone()
            dna[minus] = reverse_complement_six_channel(dna[minus])
        return dna


class RegionalFeatureCache:
    """Memory-mapped, absolute-coordinate R16 feature cache."""

    MANIFEST = "manifest.json"

    def __init__(self, root: Path | str):
        self.root = Path(root)
        manifest_path = self.root / self.MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(self.manifest["region_size"]) != REGION_SIZE:
            raise ValueError("regional cache does not use R16")
        self.channels = int(self.manifest["channels"])
        self._arrays: dict[tuple[str, str], np.ndarray] = {}

    @classmethod
    def create(
        cls,
        root: Path | str,
        *,
        channels: int,
        metadata: Mapping[str, object] | None = None,
    ) -> "RegionalFeatureCache":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / cls.MANIFEST
        payload: dict[str, object] = {
            "format_version": 1,
            "region_size": REGION_SIZE,
            "channels": int(channels),
            "dtype": "float16",
            "contigs": {},
            "metadata": dict(metadata or {}),
        }
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if int(existing["region_size"]) != REGION_SIZE or int(existing["channels"]) != channels:
                raise ValueError("existing cache manifest has an incompatible contract")
        else:
            _atomic_json(manifest_path, payload)
        return cls(root)

    def _reload_manifest(self) -> None:
        self.manifest = json.loads(
            (self.root / self.MANIFEST).read_text(encoding="utf-8")
        )

    def close(self) -> None:
        """Release memory maps so cache files can be replaced on Windows."""

        for array in self._arrays.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self._arrays.clear()

    def __enter__(self) -> "RegionalFeatureCache":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def write_contig(
        self,
        contig: str,
        plus: np.ndarray,
        minus: np.ndarray,
        *,
        length_bp: int,
        source: Mapping[str, object] | None = None,
    ) -> None:
        plus = np.asarray(plus, dtype=np.float16)
        minus = np.asarray(minus, dtype=np.float16)
        expected_regions = (int(length_bp) + REGION_SIZE - 1) // REGION_SIZE
        expected_shape = (expected_regions, self.channels)
        if plus.shape != expected_shape or minus.shape != expected_shape:
            raise ValueError(
                f"cache arrays for {contig} must have shape {expected_shape}, got "
                f"{plus.shape} and {minus.shape}"
            )
        stem = _safe_file_stem(contig)
        plus_name = f"{stem}.plus.npy"
        minus_name = f"{stem}.minus.npy"
        np.save(self.root / plus_name, plus, allow_pickle=False)
        np.save(self.root / minus_name, minus, allow_pickle=False)

        self._reload_manifest()
        contigs = dict(self.manifest.get("contigs", {}))
        contigs[contig] = {
            "length_bp": int(length_bp),
            "regions": expected_regions,
            "plus": plus_name,
            "minus": minus_name,
            "source": dict(source or {}),
        }
        updated = dict(self.manifest)
        updated["contigs"] = contigs
        _atomic_json(self.root / self.MANIFEST, updated)
        self.manifest = updated
        self._arrays.pop((contig, "+"), None)
        self._arrays.pop((contig, "-"), None)

    def has_contig(self, contig: str) -> bool:
        return contig in self.manifest.get("contigs", {})

    def _array(self, contig: str, strand: str) -> np.ndarray:
        key = (contig, strand)
        if key not in self._arrays:
            record = self.manifest["contigs"].get(contig)
            if record is None:
                raise KeyError(f"contig {contig!r} is absent from regional cache")
            field = "plus" if strand == "+" else "minus"
            self._arrays[key] = np.load(
                self.root / record[field], mmap_mode="r", allow_pickle=False
            )
        return self._arrays[key]

    def lookup_window(
        self,
        contig: str,
        strand: str,
        genomic_start: int,
        length: int,
        *,
        reverse_to_oriented: bool = False,
    ) -> np.ndarray:
        """Return ``[channels, length]`` aligned to base-resolution DNA."""

        if strand not in {"+", "-"}:
            raise ValueError("strand must be '+' or '-'")
        if length <= 0:
            raise ValueError("length must be positive")
        record = self.manifest["contigs"].get(contig)
        if record is None:
            raise KeyError(f"contig {contig!r} is absent from regional cache")
        contig_length = int(record["length_bp"])
        positions = genomic_start + np.arange(length, dtype=np.int64)
        valid = (positions >= 0) & (positions < contig_length)
        result = np.zeros((self.channels, length), dtype=np.float32)
        if valid.any():
            regions = positions[valid] // REGION_SIZE
            values = np.asarray(self._array(contig, strand)[regions], dtype=np.float32)
            result[:, valid] = values.T
        if reverse_to_oriented:
            result = result[:, ::-1].copy()
        return result

    def condition_for_profile_tiles(
        self,
        selected_tiles: pd.DataFrame,
        *,
        context_flank: int,
        target_length: int,
    ) -> torch.Tensor:
        """Build plus-then-minus condition order matching ProfileBatch."""

        length = target_length + 2 * context_flank
        plus: list[np.ndarray] = []
        minus: list[np.ndarray] = []
        for tile in selected_tiles.itertuples(index=False):
            contig = str(tile.contig)
            genomic_start = int(tile.target_start) - context_flank
            plus.append(self.lookup_window(contig, "+", genomic_start, length))
            minus.append(
                self.lookup_window(
                    contig,
                    "-",
                    genomic_start,
                    length,
                    reverse_to_oriented=True,
                )
            )
        return torch.from_numpy(np.stack((*plus, *minus), axis=0))


def _padded_raw_window(sequence: np.ndarray, start_bp: int, length: int) -> np.ndarray:
    result = np.full(length, 8, dtype=np.uint8)
    source_start = max(start_bp, 0)
    source_end = min(start_bp + length, len(sequence))
    if source_end > source_start:
        destination_start = source_start - start_bp
        destination_end = destination_start + source_end - source_start
        result[destination_start:destination_end] = sequence[source_start:source_end]
    return result


@torch.no_grad()
def predict_contig_to_regional_cache(
    model: RegionalDetectorA,
    sequence: np.ndarray,
    *,
    contig: str,
    cache: RegionalFeatureCache,
    device: torch.device,
    input_bp: int = 8192,
    output_bp: int = 4096,
    source: Mapping[str, object] | None = None,
) -> None:
    """Predict one contig with overlap and write genomic-aligned plus/minus R16 features."""

    if input_bp % REGION_SIZE or output_bp % REGION_SIZE:
        raise ValueError("input_bp and output_bp must be multiples of 16")
    if output_bp <= 0 or output_bp > input_bp or (input_bp - output_bp) % 2:
        raise ValueError("output_bp must define a symmetric central crop")
    context_bp = (input_bp - output_bp) // 2
    if context_bp % REGION_SIZE:
        raise ValueError("regional inference context must align to R16")

    sequence = np.asarray(sequence, dtype=np.uint8)
    region_count = (len(sequence) + REGION_SIZE - 1) // REGION_SIZE
    channels = model.condition_channels
    plus_all = np.zeros((region_count, channels), dtype=np.float16)
    minus_all = np.zeros((region_count, channels), dtype=np.float16)
    input_regions = input_bp // REGION_SIZE
    output_regions = output_bp // REGION_SIZE
    crop_regions = context_bp // REGION_SIZE

    was_training = model.training
    model.eval()
    try:
        for output_start_region in range(0, region_count, output_regions):
            input_start_region = output_start_region - crop_regions
            raw = _padded_raw_window(
                sequence,
                input_start_region * REGION_SIZE,
                input_bp,
            )
            raw_tensor = torch.from_numpy(raw[None, :]).to(device)
            plus_dna = encode_raw_dna_six_channel(raw_tensor)
            both = torch.cat(
                (plus_dna, reverse_complement_six_channel(plus_dna)), dim=0
            )
            condition = model.condition(both)
            if condition.shape[-1] != input_regions:
                raise AssertionError("Network A returned an unexpected regional length")
            plus_window = condition[0, :, crop_regions : crop_regions + output_regions]
            minus_genomic = condition[1].flip(-1)
            minus_window = minus_genomic[:, crop_regions : crop_regions + output_regions]
            write_count = min(output_regions, region_count - output_start_region)
            destination = slice(output_start_region, output_start_region + write_count)
            plus_all[destination] = (
                plus_window[:, :write_count].T.float().cpu().numpy().astype(np.float16)
            )
            minus_all[destination] = (
                minus_window[:, :write_count].T.float().cpu().numpy().astype(np.float16)
            )
    finally:
        model.train(was_training)

    cache.write_contig(
        contig,
        plus_all,
        minus_all,
        length_bp=len(sequence),
        source=source,
    )


__all__ = [
    "RegionalBatch",
    "RegionalFeatureCache",
    "RegionalWindowGenerator",
    "predict_contig_to_regional_cache",
]
