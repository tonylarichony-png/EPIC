"""Contract tests for shared strand-oriented FASTA context extraction."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project import sequence_context


class TestSequenceContext(unittest.TestCase):
    def setUp(self):
        self.sequences = {
            "c1": sequence_context.encode_fasta_sequence("ACGTA"),
            "c2": sequence_context.encode_fasta_sequence("aN"),
        }

    def test_plus_and_minus_share_one_offset_convention(self):
        rows = pd.DataFrame(
            {
                "contig": ["c1", "c1"],
                "coordinate_0based": [2, 2],
                "strand": ["+", "-"],
            }
        )
        actual = sequence_context.extract_oriented_context(
            self.sequences,
            rows,
            offsets=[-2, -1, 0, 1, 2],
        )
        # +: A C [G] T A; -: reverse-complement T A [C] G T.
        np.testing.assert_array_equal(actual[0], [0, 1, 2, 3, 0])
        np.testing.assert_array_equal(actual[1], [3, 0, 1, 2, 3])

    def test_context_is_padded_at_contig_edges_and_never_wraps(self):
        rows = pd.DataFrame(
            {
                "contig": ["c2", "c2"],
                "coordinate_0based": [0, 1],
                "strand": ["+", "-"],
            }
        )
        actual = sequence_context.extract_oriented_context(
            self.sequences,
            rows,
            offsets=[-1, 0, 1],
        )
        np.testing.assert_array_equal(actual[0], [4, 0, 4])
        np.testing.assert_array_equal(actual[1], [4, 4, 3])

    def test_streamed_batches_keep_strand_metadata_outside_context(self):
        rows = pd.DataFrame(
            {
                "template_index": [10, 11],
                "contig": ["c1", "c1"],
                "coordinate_0based": [1, 1],
                "strand": ["+", "-"],
            }
        )
        [(same_rows, context)] = list(
            sequence_context.iter_oriented_context_batches(
                self.sequences,
                [rows],
                offsets=[-1, 0],
            )
        )
        self.assertIs(same_rows, rows)
        self.assertIn("strand", same_rows)
        self.assertEqual(context.shape, (2, 2))
        np.testing.assert_array_equal(context, [[0, 1], [1, 2]])

    def test_invalid_keys_and_coordinates_fail_loudly(self):
        invalid_rows = [
            pd.DataFrame(
                {"contig": ["missing"], "coordinate_0based": [0], "strand": ["+"]}
            ),
            pd.DataFrame(
                {"contig": ["c1"], "coordinate_0based": [9], "strand": ["+"]}
            ),
            pd.DataFrame(
                {"contig": ["c1"], "coordinate_0based": [0], "strand": ["?"]}
            ),
        ]
        for rows in invalid_rows:
            with self.subTest(rows=rows.to_dict("records")), self.assertRaises(
                (KeyError, ValueError)
            ):
                sequence_context.extract_oriented_context(
                    self.sequences,
                    rows,
                    offsets=[0],
                )

    def test_plain_and_gzip_fasta_use_the_same_encoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plain = root / "genome.fa"
            plain.write_bytes(b">c1 description\nACgtN\n")
            import gzip

            compressed = root / "genome.fa.gz"
            with gzip.open(compressed, "wb") as stream:
                stream.write(plain.read_bytes())
            left, left_alphabet = sequence_context.read_fasta(plain)
            right, right_alphabet = sequence_context.read_fasta(compressed)
        np.testing.assert_array_equal(left["c1"], right["c1"])
        self.assertEqual(left_alphabet, right_alphabet)
        self.assertEqual(left_alphabet, {"A": 1, "C": 1, "N": 1, "g": 1, "t": 1})


if __name__ == "__main__":
    unittest.main()
