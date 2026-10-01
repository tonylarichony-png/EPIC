"""Shared FASTA loading and strand-oriented sequence extraction for EPIC.

``strand`` remains part of the position key and submission metadata.  This
module uses it only to place every local DNA window in the same biological
orientation before an experiment builds model features.

The returned context uses five uppercase codes: A=0, C=1, G=2, T=3 and
N/other/contig-edge=4.  FASTA case is retained in the stored sequence codes so
that a separate feature builder can still derive lowercase-based features.
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import numpy as np
import pandas as pd


CONTEXT_BASES = "ACGTN"
UNKNOWN_CONTEXT_CODE = np.uint8(4)

# Raw FASTA codes preserve case: A/C/G/T=0..3, a/c/g/t=4..7, other=8.
FASTA_LOOKUP = np.full(256, 8, dtype=np.uint8)
for _code, _base in enumerate(b"ACGTacgt"):
    FASTA_LOOKUP[_base] = _code

# Convert the case-preserving FASTA representation to uppercase A/C/G/T/N.
UPPERCASE_BASE_CODE = np.array(
    [0, 1, 2, 3, 0, 1, 2, 3, 4],
    dtype=np.uint8,
)
COMPLEMENT_CODE = np.array([3, 2, 1, 0, 4], dtype=np.uint8)


def encode_fasta_sequence(sequence: bytes | str) -> np.ndarray:
    """Encode one FASTA sequence while retaining uppercase/lowercase state."""

    if isinstance(sequence, str):
        try:
            sequence = sequence.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("FASTA sequence must contain ASCII characters") from error
    if not isinstance(sequence, bytes):
        raise TypeError("sequence must be bytes or str")
    raw = np.frombuffer(sequence, dtype=np.uint8)
    return FASTA_LOOKUP[raw]


def read_fasta(path: Path | str) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Read a plain or gzip FASTA into compact, case-preserving arrays."""

    fasta_path = Path(path)
    opener = gzip.open if fasta_path.suffix.lower() == ".gz" else open
    sequences: dict[str, np.ndarray] = {}
    alphabet = np.zeros(256, dtype=np.int64)
    name: str | None = None
    chunks: list[bytes] = []

    def flush() -> None:
        nonlocal name, chunks
        if name is None:
            return
        if name in sequences:
            raise ValueError(f"Duplicate FASTA identifier: {name}")
        payload = b"".join(chunks)
        if not payload:
            raise ValueError(f"Empty FASTA record: {name}")
        raw = np.frombuffer(payload, dtype=np.uint8)
        alphabet[:] += np.bincount(raw, minlength=256)
        sequences[name] = FASTA_LOOKUP[raw]

    with opener(fasta_path, "rb") as stream:
        for line in stream:
            if line.startswith(b">"):
                flush()
                fields = line[1:].split()
                if not fields:
                    raise ValueError("Empty FASTA header")
                try:
                    name = fields[0].decode("ascii")
                except UnicodeDecodeError as error:
                    raise ValueError("FASTA identifiers must be ASCII") from error
                chunks = []
            else:
                if name is None:
                    raise ValueError("Sequence before FASTA header")
                chunks.append(line.strip())
        flush()

    if not sequences:
        raise ValueError("FASTA contains no records")
    observed = {chr(i): int(count) for i, count in enumerate(alphabet) if count}
    return sequences, observed


def _integer_vector(values, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must be a one-dimensional integer vector")
    return array.astype(np.int64, copy=False)


def extract_oriented_context_from_arrays(
    sequences: Mapping[str, np.ndarray],
    contigs,
    coordinates,
    strands,
    offsets,
) -> np.ndarray:
    """Extract A/C/G/T/N context with offsets interpreted in strand space.

    Offset zero always denotes ``coordinate_0based``.  For the plus strand an
    offset is added to the genomic coordinate.  For the minus strand it is
    subtracted and the selected base is complemented.  Consequently ``-1``
    means the preceding base in the oriented sequence for both strands.
    """

    contig_array = np.asarray(contigs, dtype=object)
    coordinate_array = _integer_vector(coordinates, name="coordinates")
    strand_array = np.asarray(strands)
    offset_array = _integer_vector(offsets, name="offsets")

    if contig_array.ndim != 1 or strand_array.ndim != 1:
        raise TypeError("contigs and strands must be one-dimensional vectors")
    size = len(contig_array)
    if len(coordinate_array) != size or len(strand_array) != size:
        raise ValueError("contigs, coordinates and strands must have equal length")
    if not len(offset_array):
        raise ValueError("offsets must not be empty")
    if not np.all(np.isin(strand_array, ("+", "-"))):
        raise ValueError("strand must be '+' or '-'")

    result = np.full(
        (size, len(offset_array)),
        UNKNOWN_CONTEXT_CODE,
        dtype=np.uint8,
    )
    minus = strand_array == "-"

    for contig_value in pd.unique(contig_array):
        contig = str(contig_value)
        if contig not in sequences:
            raise KeyError(f"Contig is absent from FASTA: {contig}")
        sequence = np.asarray(sequences[contig])
        if sequence.ndim != 1 or not len(sequence):
            raise ValueError(f"FASTA sequence must be a non-empty vector: {contig}")
        if not np.issubdtype(sequence.dtype, np.integer):
            raise TypeError(f"FASTA sequence codes must be integers: {contig}")
        if np.any(sequence < 0) or np.any(sequence >= len(UPPERCASE_BASE_CODE)):
            raise ValueError(f"Unknown FASTA sequence code in contig: {contig}")

        take = np.flatnonzero(contig_array == contig_value)
        centers = coordinate_array[take]
        if np.any(centers < 0) or np.any(centers >= len(sequence)):
            raise ValueError(f"Center coordinate is outside FASTA contig: {contig}")

        genomic_coordinates = centers[:, None] + np.where(
            minus[take, None],
            -offset_array,
            offset_array,
        )
        valid = (genomic_coordinates >= 0) & (genomic_coordinates < len(sequence))
        clipped = np.clip(genomic_coordinates, 0, len(sequence) - 1)
        values = UPPERCASE_BASE_CODE[sequence[clipped].astype(np.int64, copy=False)]
        values = np.where(minus[take, None], COMPLEMENT_CODE[values], values)
        result[take] = np.where(valid, values, UNKNOWN_CONTEXT_CODE)

    return result


def extract_oriented_context(
    sequences: Mapping[str, np.ndarray],
    rows: pd.DataFrame,
    offsets,
    *,
    contig_column: str = "contig",
    coordinate_column: str = "coordinate_0based",
    strand_column: str = "strand",
) -> np.ndarray:
    """Extract oriented context for a standard EPIC position batch."""

    required = (contig_column, coordinate_column, strand_column)
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise KeyError("Context rows are missing columns: " + ", ".join(missing))
    return extract_oriented_context_from_arrays(
        sequences,
        rows[contig_column].to_numpy(),
        rows[coordinate_column].to_numpy(),
        rows[strand_column].to_numpy(),
        offsets,
    )


def iter_oriented_context_batches(
    sequences: Mapping[str, np.ndarray],
    batches: Iterable[pd.DataFrame],
    offsets,
) -> Iterator[tuple[pd.DataFrame, np.ndarray]]:
    """Attach an oriented context matrix to each streamed position batch."""

    for batch in batches:
        yield batch, extract_oriented_context(sequences, batch, offsets)


__all__ = [
    "COMPLEMENT_CODE",
    "CONTEXT_BASES",
    "FASTA_LOOKUP",
    "UNKNOWN_CONTEXT_CODE",
    "UPPERCASE_BASE_CODE",
    "encode_fasta_sequence",
    "extract_oriented_context",
    "extract_oriented_context_from_arrays",
    "iter_oriented_context_batches",
    "read_fasta",
]
