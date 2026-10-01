"""Reusable sequence-composition features for genomic position models.

The functions in this module only transform DNA sequence and coordinates.
They never inspect targets, fit estimators, or access validation state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sequence_context import UPPERCASE_BASE_CODE


@dataclass(frozen=True)
class GCWindowPrefix:
    """Inclusive-information prefix sums for one FASTA contig."""

    gc: np.ndarray
    valid: np.ndarray
    length: int


def canonical_base_codes(raw_sequence: np.ndarray) -> np.ndarray:
    """Map the case-preserving FASTA encoding to A/C/G/T/N codes."""

    raw = np.asarray(raw_sequence)
    if raw.ndim != 1 or not len(raw):
        raise ValueError("raw_sequence must be a non-empty one-dimensional array")
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("raw_sequence must contain integer FASTA codes")
    if np.any(raw < 0) or np.any(raw >= len(UPPERCASE_BASE_CODE)):
        raise ValueError("raw_sequence contains an unknown FASTA code")
    return UPPERCASE_BASE_CODE[raw.astype(np.int64, copy=False)]


def gc_window_prefix(raw_sequence: np.ndarray) -> GCWindowPrefix:
    """Build O(1)-query GC and canonical-base prefix sums for one contig."""

    return gc_prefix_from_base_codes(canonical_base_codes(raw_sequence))


def gc_prefix_from_base_codes(letters: np.ndarray) -> GCWindowPrefix:
    """Build prefix sums from uppercase A=0/C=1/G=2/T=3/N=4 codes."""

    letters = np.asarray(letters)
    if letters.ndim != 1 or not len(letters):
        raise ValueError("letters must be a non-empty one-dimensional array")
    if not np.issubdtype(letters.dtype, np.integer):
        raise TypeError("letters must contain integer base codes")
    if np.any(letters < 0) or np.any(letters > 4):
        raise ValueError("letters must use A=0, C=1, G=2, T=3, N=4")
    is_gc = (letters == 1) | (letters == 2)
    is_valid = letters < 4
    prefix_gc = np.empty(len(letters) + 1, dtype=np.int64)
    prefix_valid = np.empty(len(letters) + 1, dtype=np.int64)
    prefix_gc[0] = 0
    prefix_valid[0] = 0
    np.cumsum(is_gc, dtype=np.int64, out=prefix_gc[1:])
    np.cumsum(is_valid, dtype=np.int64, out=prefix_valid[1:])
    return GCWindowPrefix(prefix_gc, prefix_valid, len(letters))


def gc_counts_from_prefix(
    prefix: GCWindowPrefix,
    coordinates: np.ndarray,
    *,
    window_start: int = -100,
    window_end: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Count G/C and valid A/C/G/T bases in clipped inclusive windows."""

    if not isinstance(window_start, int) or not isinstance(window_end, int):
        raise TypeError("window bounds must be integers")
    if window_start > 0 or window_end < 0 or window_start > window_end:
        raise ValueError("window must contain offset zero and have ordered bounds")
    values = np.asarray(coordinates)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise TypeError("coordinates must be a one-dimensional integer array")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0) or np.any(values >= prefix.length):
        raise ValueError("coordinate is outside the contig")

    left = np.clip(values + window_start, 0, prefix.length)
    right = np.clip(values + window_end + 1, 0, prefix.length)
    gc_count = prefix.gc[right] - prefix.gc[left]
    valid_count = prefix.valid[right] - prefix.valid[left]
    if np.any(gc_count < 0) or np.any(gc_count > valid_count):
        raise AssertionError("invalid GC prefix-sum result")
    return gc_count.astype(np.int16), valid_count.astype(np.int16)


def gc_window_counts(
    raw_sequence: np.ndarray,
    coordinates: np.ndarray,
    *,
    window_start: int = -100,
    window_end: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper for one batch of coordinates on one contig."""

    return gc_counts_from_prefix(
        gc_window_prefix(raw_sequence),
        coordinates,
        window_start=window_start,
        window_end=window_end,
    )


def gc_fraction(
    gc_count: np.ndarray,
    valid_count: np.ndarray,
    *,
    missing_value: float = np.nan,
) -> np.ndarray:
    """Convert integer sufficient statistics to GC among valid A/C/G/T bases."""

    gc_values = np.asarray(gc_count)
    valid_values = np.asarray(valid_count)
    if gc_values.shape != valid_values.shape:
        raise ValueError("gc_count and valid_count must have identical shapes")
    if np.any(gc_values < 0) or np.any(valid_values < 0):
        raise ValueError("counts must be non-negative")
    if np.any(gc_values > valid_values):
        raise ValueError("gc_count cannot exceed valid_count")
    result = np.full(gc_values.shape, float(missing_value), dtype=np.float64)
    present = valid_values > 0
    result[present] = gc_values[present] / valid_values[present]
    return result


__all__ = [
    "GCWindowPrefix",
    "canonical_base_codes",
    "gc_counts_from_prefix",
    "gc_fraction",
    "gc_prefix_from_base_codes",
    "gc_window_counts",
    "gc_window_prefix",
]
