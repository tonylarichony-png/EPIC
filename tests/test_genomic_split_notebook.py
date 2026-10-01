"""Check the shared EPIC data contract against small independent examples."""

import gzip
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project import epic_data


class TestGenomicDataContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.lengths = {"a": 20, "b": 20, "c": 20}
        self.whitelist = epic_data.read_whitelist(
            self.gz("w.bed.gz", "a\t2\t4\na\t7\t8\nb\t1\t3\n"),
            self.lengths,
        )
        self.template_text = (
            "a\t2\t4\t.\t0\t+\na\t7\t8\t.\t0\t+\nb\t1\t3\t.\t0\t+\n"
            "a\t2\t4\t.\t0\t-\na\t7\t8\t.\t0\t-\nb\t1\t3\t.\t0\t-\n"
        )
        self.template = epic_data.read_template(
            self.gz("template.bed.gz", self.template_text),
            self.whitelist,
            "train",
        )
        self.template["split"] = self.template["contig"].map(
            {"a": "train", "b": "validation"}
        )

    def gz(self, name, content):
        path = self.root / name
        with gzip.open(path, "wt", encoding="ascii") as stream:
            stream.write(content)
        return path

    def signals(self):
        first = epic_data.read_sparse_signal(
            self.gz("r1.bed.gz", "a\t2\t4\tx\t2\t+\nb\t2\t3\tx\t5\t-\n"),
            self.whitelist,
            "r1_count",
        )
        second = epic_data.read_sparse_signal(
            self.gz("r2.bed.gz", "a\t3\t4\tx\t3\t+\na\t7\t8\tx\t4\t-\n"),
            self.whitelist,
            "r2_count",
        )
        return epic_data.pool_targets(first, second, self.whitelist)

    def test_contigs_disjoint_and_reproducible_independent_of_input_order(self):
        assign = epic_data.assign_contigs
        left = assign(["a", "b", "a"], ["c"], 0.5, 42)
        right = assign(["b", "a"], ["c"], 0.5, 42)
        pd.testing.assert_frame_equal(left, right)
        self.assertTrue(left.contig.is_unique)
        self.assertEqual(set(left.split), {"train", "validation", "test"})
        self.assertEqual(left.loc[left.contig == "c", "split"].item(), "test")
        with self.assertRaises(ValueError):
            assign(["a", "b"], ["b"], 0.5, 42)
        with self.assertRaises(ValueError):
            assign(["a"], ["c"], 0.5, 42)

    def test_intervals_expand_and_replicates_sum_by_position_and_strand(self):
        merged = self.signals()
        self.assertEqual(merged.template_index.tolist(), [0, 1, 7, 9])
        self.assertEqual(merged["count"].tolist(), [2, 5, 4, 5])
        self.assertEqual(merged.target.tolist(), [1, 1, 1, 1])

    def test_batches_keep_original_indices_across_gaps_and_both_strands(self):
        targets = self.signals()
        intervals = self.template.loc[self.template.split == "train"]
        targets = targets.loc[targets.contig == "a"]
        batches = list(
            epic_data.iter_position_batches(intervals, targets, batch_size=4)
        )
        self.assertEqual([len(batch) for batch in batches], [4, 2])
        rows = pd.concat(batches, ignore_index=True)
        self.assertEqual(rows.template_index.tolist(), [0, 1, 2, 5, 6, 7])
        self.assertEqual(rows.coordinate_0based.tolist(), [2, 3, 7, 2, 3, 7])
        self.assertEqual(rows.strand.tolist(), ["+", "+", "+", "-", "-", "-"])
        self.assertEqual(rows["count"].tolist(), [2, 5, 0, 0, 0, 4])
        self.assertEqual(rows.target.tolist(), [1, 1, 0, 0, 0, 1])

    def test_test_rows_have_no_answers_and_reject_labels(self):
        intervals = self.template.loc[self.template.contig == "b"].copy()
        intervals["split"] = "test"
        intervals["source_split"] = "test"
        batch = next(epic_data.iter_position_batches(intervals, batch_size=10))
        self.assertNotIn("count", batch)
        self.assertNotIn("target", batch)
        with self.assertRaises(ValueError):
            next(epic_data.iter_position_batches(intervals, self.signals()))

    def test_sparse_signal_rejects_duplicates_and_outside_whitelist(self):
        bad = [
            "a\t3\t5\tx\t1\t+\n",
            "a\t1\t2\tx\t1\t+\n",
            "c\t2\t3\tx\t1\t+\n",
            "a\t2\t3\tx\t1\t+\na\t2\t3\tx\t2\t+\n",
        ]
        for i, text in enumerate(bad):
            with self.subTest(i=i), self.assertRaises(ValueError):
                epic_data.read_sparse_signal(
                    self.gz(f"bad{i}.bed.gz", text),
                    self.whitelist,
                    "r1_count",
                )

    def test_whitelist_overlap_and_template_wrong_strand_rejected(self):
        with self.assertRaises(ValueError):
            epic_data.read_whitelist(
                self.gz("overlap.gz", "a\t1\t4\na\t3\t7\n"),
                self.lengths,
            )
        with self.assertRaises(AssertionError):
            epic_data.read_template(
                self.gz("wrong.gz", self.template_text.replace("+", "-")),
                self.whitelist,
                "train",
            )

    def test_empty_positive_table_gives_all_negative_examples(self):
        intervals = self.template.loc[self.template.split == "train"]
        empty = pd.DataFrame(
            {
                "template_index": pd.Series(dtype="int64"),
                "count": pd.Series(dtype="int64"),
            }
        )
        rows = pd.concat(
            epic_data.iter_position_batches(intervals, empty, batch_size=2)
        )
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows["count"].sum(), 0)
        self.assertEqual(rows.target.sum(), 0)
        with self.assertRaises(ValueError):
            next(
                epic_data.iter_position_batches(
                    intervals,
                    empty,
                    batch_size=0,
                )
            )


if __name__ == "__main__":
    unittest.main()
