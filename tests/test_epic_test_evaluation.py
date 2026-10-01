"""Contracts for the explicit labelled-demo-test loader."""

import gzip
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.epic_test_evaluation import (
    audit_test_template,
    read_dense_test_ground_truth,
)


def tiny_template() -> pd.DataFrame:
    rows = [
        ("c1", 10, 13, "+", 0, 3),
        ("c2", 20, 22, "+", 3, 5),
        ("c1", 10, 13, "-", 5, 8),
        ("c2", 20, 22, "-", 8, 10),
    ]
    frame = pd.DataFrame(rows, columns=[
        "contig", "start", "end", "strand", "template_start", "template_stop",
    ])
    frame["length_bp"] = frame["end"] - frame["start"]
    frame["source_split"] = "test"
    frame["split"] = "test"
    return frame


class TestEpicTestEvaluation(unittest.TestCase):
    def test_template_and_dense_target_preserve_official_order(self):
        intervals = tiny_template()
        contract = audit_test_template(intervals)
        self.assertEqual(contract["test_positions"], 10)
        self.assertEqual(contract["plus_positions"], 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truth.txt.gz"
            with gzip.open(path, "wt", encoding="ascii") as stream:
                stream.write("0\n2\n0\n0\n3\n4\n0\n0\n0\n5\n")
            positives = read_dense_test_ground_truth(path, intervals, chunksize=3)
        self.assertEqual(positives["template_index"].tolist(), [1, 4, 5, 9])
        self.assertEqual(positives["coordinate_0based"].tolist(), [11, 21, 10, 21])
        self.assertEqual(positives["strand"].tolist(), ["+", "+", "-", "-"])
        self.assertEqual(positives["count"].tolist(), [2, 3, 4, 5])

    def test_dense_target_length_must_equal_template(self):
        intervals = tiny_template()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short.txt.gz"
            with gzip.open(path, "wt", encoding="ascii") as stream:
                stream.write("0\n" * 9)
            with self.assertRaisesRegex(ValueError, "length"):
                read_dense_test_ground_truth(path, intervals, chunksize=4)


if __name__ == "__main__":
    unittest.main()
