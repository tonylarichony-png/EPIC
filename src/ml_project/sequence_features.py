"""Deterministic sparse encoders for strand-oriented DNA contexts.

The encoders in this module have a fixed vocabulary and do not learn from the
target or from category frequencies.  They can therefore be shared by train,
validation and inference without fitting a separate vocabulary in every
experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import sparse

from .sequence_context import CONTEXT_BASES


DINUCLEOTIDE_ENCODING_VERSION = "dinucleotide_reference_v1"
POSITIONAL_ENCODING_VERSION = "positional_reference_v1"
CANONICAL_BASES = "ACGT"
DINUCLEOTIDE_LABELS = tuple(
    first + second for first in CANONICAL_BASES for second in CANONICAL_BASES
) + ("N/edge",)


@dataclass(frozen=True)
class EncodedSequenceFeatures:
    """Sparse feature matrix together with its reproducible schema."""

    matrix: sparse.csr_matrix
    names: tuple[str, ...]
    encoding: str
    reference: str
    version: str


def _validated_context(context, *, width: int | None = None) -> np.ndarray:
    values = np.asarray(context)
    if values.ndim != 2:
        raise TypeError("context must be a two-dimensional array")
    if width is not None and values.shape[1] != width:
        raise ValueError(f"context must contain exactly {width} positions")
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("context codes must be integers")
    if np.any(values < 0) or np.any(values >= len(CONTEXT_BASES)):
        raise ValueError("context contains a code outside A/C/G/T/N-edge")
    return values.astype(np.int8, copy=False)


def _validated_offsets(offsets: Sequence[int], *, width: int) -> tuple[int, ...]:
    values = tuple(offsets)
    if len(values) != width or any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in values
    ):
        raise ValueError(f"offsets must contain exactly {width} integers")
    return tuple(int(value) for value in values)


def _pair_prefix(offsets: tuple[int, int]) -> str:
    return f"pair[{offsets[0]},{offsets[1]}]"


def dinucleotide_category_codes(context) -> np.ndarray:
    """Return fixed category codes 0..15 for ACGT pairs and 16 for N/edge."""

    values = _validated_context(context, width=2)
    return np.where(
        (values[:, 0] < 4) & (values[:, 1] < 4),
        4 * values[:, 0] + values[:, 1],
        16,
    ).astype(np.int8, copy=False)


def encode_dinucleotide(
    context,
    *,
    offsets: Sequence[int] = (-1, 0),
    reference: str = "TT",
    unknown_label: str = "N/edge",
) -> EncodedSequenceFeatures:
    """Reference-code one oriented DNA pair as a sparse categorical feature.

    The 16 canonical A/C/G/T pairs form a fixed vocabulary.  ``reference`` is
    represented by an all-zero row; every other canonical pair gets one column.
    Any pair containing context code 4 is mapped to one stable unknown/edge
    column.  With the default TT reference this produces 16 columns: 15
    canonical contrasts plus ``N/edge``.
    """

    values = _validated_context(context, width=2)
    offset_values = _validated_offsets(offsets, width=2)
    reference = str(reference).upper()
    if len(reference) != 2 or any(base not in CANONICAL_BASES for base in reference):
        raise ValueError("reference must be one canonical two-base DNA pair")
    if not isinstance(unknown_label, str) or not unknown_label.strip():
        raise ValueError("unknown_label must be a non-empty string")

    reference_code = 4 * CANONICAL_BASES.index(reference[0]) + CANONICAL_BASES.index(
        reference[1]
    )
    canonical = dinucleotide_category_codes(values)
    active_codes = [code for code in range(16) if code != reference_code] + [16]
    column_by_code = np.full(17, -1, dtype=np.int16)
    column_by_code[active_codes] = np.arange(len(active_codes), dtype=np.int16)
    nonreference_rows = np.flatnonzero(canonical != reference_code)
    columns = column_by_code[canonical[nonreference_rows]]
    matrix = sparse.csr_matrix(
        (
            np.ones(len(nonreference_rows), dtype=np.float64),
            (nonreference_rows, columns),
        ),
        shape=(len(values), len(active_codes)),
    )

    prefix = _pair_prefix(offset_values)
    names = tuple(
        f"{prefix}={CANONICAL_BASES[code // 4]}{CANONICAL_BASES[code % 4]}"
        if code < 16
        else f"{prefix}={unknown_label}"
        for code in active_codes
    )
    return EncodedSequenceFeatures(
        matrix=matrix,
        names=names,
        encoding="reference_one_hot",
        reference=reference,
        version=DINUCLEOTIDE_ENCODING_VERSION,
    )


def encode_positional_bases(
    context,
    offsets: Sequence[int],
    *,
    reference: str = "T",
    unknown_label: str = "N/edge",
) -> EncodedSequenceFeatures:
    """Reference-code each oriented position independently as sparse features."""

    values = _validated_context(context)
    offset_values = _validated_offsets(offsets, width=values.shape[1])
    reference = str(reference).upper()
    if len(reference) != 1 or reference not in CANONICAL_BASES:
        raise ValueError("reference must be one canonical DNA base")
    if not isinstance(unknown_label, str) or not unknown_label.strip():
        raise ValueError("unknown_label must be a non-empty string")

    reference_code = CONTEXT_BASES.index(reference)
    active_codes = [code for code in range(len(CONTEXT_BASES)) if code != reference_code]
    column_by_code = np.full(len(CONTEXT_BASES), -1, dtype=np.int8)
    column_by_code[active_codes] = np.arange(len(active_codes), dtype=np.int8)
    rows, positions = np.nonzero(values != reference_code)
    columns = len(active_codes) * positions + column_by_code[values[rows, positions]]
    matrix = sparse.csr_matrix(
        (
            np.ones(len(rows), dtype=np.float64),
            (rows, columns),
        ),
        shape=(len(values), len(offset_values) * len(active_codes)),
    )

    labels = {
        code: unknown_label if code == 4 else CONTEXT_BASES[code]
        for code in active_codes
    }
    names = tuple(
        f"base[{offset:+d}]={labels[code]}"
        for offset in offset_values
        for code in active_codes
    )
    return EncodedSequenceFeatures(
        matrix=matrix,
        names=names,
        encoding="reference_one_hot_per_position",
        reference=reference,
        version=POSITIONAL_ENCODING_VERSION,
    )


__all__ = [
    "CANONICAL_BASES",
    "DINUCLEOTIDE_LABELS",
    "DINUCLEOTIDE_ENCODING_VERSION",
    "EncodedSequenceFeatures",
    "POSITIONAL_ENCODING_VERSION",
    "dinucleotide_category_codes",
    "encode_dinucleotide",
    "encode_positional_bases",
]
