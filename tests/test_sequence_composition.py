"""Tests for reusable GC-window sequence features."""

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.sequence_composition import (
    gc_counts_from_prefix,
    gc_fraction,
    gc_window_prefix,
)
from ml_project.sequence_context import encode_fasta_sequence


class TestSequenceComposition(unittest.TestCase):
    def test_clipped_windows_count_gc_and_ignore_unknown_bases(self):
        sequence = encode_fasta_sequence("ACGTN")
        prefix = gc_window_prefix(sequence)
        gc_count, valid_count = gc_counts_from_prefix(
            prefix,
            np.array([0, 2, 4]),
            window_start=-1,
            window_end=1,
        )
        np.testing.assert_array_equal(gc_count, [1, 2, 0])
        np.testing.assert_array_equal(valid_count, [2, 3, 1])
        np.testing.assert_allclose(gc_fraction(gc_count, valid_count), [0.5, 2 / 3, 0])

    def test_lowercase_bases_are_canonical_for_gc(self):
        sequence = encode_fasta_sequence("acgtn")
        prefix = gc_window_prefix(sequence)
        gc_count, valid_count = gc_counts_from_prefix(
            prefix,
            np.array([2]),
            window_start=-2,
            window_end=2,
        )
        self.assertEqual(int(gc_count[0]), 2)
        self.assertEqual(int(valid_count[0]), 4)

    def test_zero_valid_window_uses_explicit_missing_value(self):
        result = gc_fraction(
            np.array([0, 1]),
            np.array([0, 2]),
            missing_value=-1.0,
        )
        np.testing.assert_allclose(result, [-1.0, 0.5])

    def test_invalid_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            gc_fraction(np.array([2]), np.array([1]))


if __name__ == "__main__":
    unittest.main()
