"""Contract tests for shared sparse DNA feature encoders."""
from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.sequence_features import (
    DINUCLEOTIDE_LABELS,
    DINUCLEOTIDE_ENCODING_VERSION,
    POSITIONAL_ENCODING_VERSION,
    dinucleotide_category_codes,
    encode_dinucleotide,
    encode_positional_bases,
)


class TestSequenceFeatures(unittest.TestCase):
    def test_dinucleotide_category_codes_have_fixed_17_value_vocabulary(self):
        canonical = np.array(
            [[first, second] for first in range(4) for second in range(4)],
            dtype=np.uint8,
        )
        unknown_or_edge = np.array(
            [[4, 0], [0, 4], [4, 4]],
            dtype=np.uint8,
        )
        actual = dinucleotide_category_codes(
            np.vstack((canonical, unknown_or_edge))
        )

        np.testing.assert_array_equal(actual[:16], np.arange(16))
        np.testing.assert_array_equal(actual[16:], [16, 16, 16])
        self.assertEqual(
            DINUCLEOTIDE_LABELS,
            (
                "AA",
                "AC",
                "AG",
                "AT",
                "CA",
                "CC",
                "CG",
                "CT",
                "GA",
                "GC",
                "GG",
                "GT",
                "TA",
                "TC",
                "TG",
                "TT",
                "N/edge",
            ),
        )

    def test_dinucleotide_uses_tt_reference_and_stable_unknown_column(self):
        encoded = encode_dinucleotide(
            np.array(
                [
                    [3, 3],  # TT reference
                    [1, 0],  # CA
                    [2, 3],  # GT
                    [4, 0],  # N/edge-A
                ],
                dtype=np.uint8,
            )
        )
        self.assertEqual(encoded.matrix.shape, (4, 16))
        self.assertEqual(encoded.matrix[0].nnz, 0)
        self.assertEqual(encoded.matrix[1].nnz, 1)
        self.assertEqual(
            encoded.names[int(encoded.matrix[1].indices[0])],
            "pair[-1,0]=CA",
        )
        self.assertEqual(
            encoded.names[int(encoded.matrix[3].indices[0])],
            "pair[-1,0]=N/edge",
        )
        self.assertNotIn("pair[-1,0]=TT", encoded.names)
        self.assertEqual(encoded.reference, "TT")
        self.assertEqual(encoded.version, DINUCLEOTIDE_ENCODING_VERSION)

    def test_dinucleotide_schema_does_not_depend_on_observed_categories(self):
        first = encode_dinucleotide(np.array([[0, 0]], dtype=np.uint8))
        second = encode_dinucleotide(np.array([[1, 2]], dtype=np.uint8))
        self.assertEqual(first.names, second.names)
        self.assertEqual(first.matrix.shape[1], second.matrix.shape[1])

    def test_positional_encoder_matches_reference_coding_contract(self):
        encoded = encode_positional_bases(
            np.array([[3, 3], [0, 4], [1, 2]], dtype=np.uint8),
            offsets=[-1, 0],
            reference="T",
        )
        self.assertEqual(encoded.matrix.shape, (3, 8))
        self.assertEqual(encoded.matrix[0].nnz, 0)
        self.assertEqual(
            encoded.names,
            (
                "base[-1]=A",
                "base[-1]=C",
                "base[-1]=G",
                "base[-1]=N/edge",
                "base[+0]=A",
                "base[+0]=C",
                "base[+0]=G",
                "base[+0]=N/edge",
            ),
        )
        np.testing.assert_array_equal(
            encoded.matrix.toarray()[1],
            [1, 0, 0, 0, 0, 0, 0, 1],
        )
        self.assertEqual(encoded.reference, "T")
        self.assertEqual(encoded.version, POSITIONAL_ENCODING_VERSION)

    def test_invalid_context_reference_and_offsets_are_rejected(self):
        bad_calls = [
            lambda: encode_dinucleotide(np.array([[0, 1, 2]], dtype=np.uint8)),
            lambda: encode_dinucleotide(np.array([[0, 5]], dtype=np.uint8)),
            lambda: encode_dinucleotide(np.array([[0, 1]], dtype=np.uint8), reference="NN"),
            lambda: encode_positional_bases(
                np.array([[0, 1]], dtype=np.uint8), offsets=[0], reference="T"
            ),
        ]
        for call in bad_calls:
            with self.subTest(call=call), self.assertRaises((TypeError, ValueError)):
                call()


if __name__ == "__main__":
    unittest.main()
