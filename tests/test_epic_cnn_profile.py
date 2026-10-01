"""Contract tests for fixed-tile EPIC CNN profile sampling."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.epic_cnn import (
    ProfileWindowConfig,
    ProfileWindowGenerator,
    create_cnn_baseline,
    create_cnn_presence_intensity,
    dml_weighted_huber,
    dml_weighted_profile_bce_with_logits,
    evaluate_profile_model,
    evaluate_presence_intensity_model,
    forward_both_strands,
    forward_both_strands_multitask,
)
from ml_project.sequence_context import encode_fasta_sequence


class FakeSplit:
    def __init__(self):
        plus = pd.DataFrame(
            {
                "contig": ["c1", "c1"],
                "start": [1, 8],
                "end": [5, 11],
                "strand": ["+", "+"],
                "length_bp": [4, 3],
            }
        )
        minus = plus.assign(strand="-")
        self._intervals = pd.concat([plus, minus], ignore_index=True)
        self._positives = pd.DataFrame(
            {
                "contig": ["c1", "c1"],
                "coordinate_0based": [2, 9],
                "strand": ["+", "-"],
                "count": [2, 3],
            }
        )

    def intervals(self, part):
        if part != "train":
            raise ValueError(part)
        return self._intervals

    def positive_targets(self, part):
        if part != "train":
            raise ValueError(part)
        return self._positives


class TestProfileWindowGenerator(unittest.TestCase):
    def setUp(self):
        self.config = ProfileWindowConfig(
            target_length=4,
            context_flank=2,
            positive_branch_fraction=0.5,
        )
        self.generator = ProfileWindowGenerator(
            FakeSplit(),
            {"c1": encode_fasta_sequence("ACgtNACGTacg")},
            config=self.config,
            seed=42,
        )

    def test_tile_index_exactly_preserves_population(self):
        tiles = self.generator.tile_index
        self.assertEqual(int(tiles.allowed_position_strand.sum()), 14)
        self.assertEqual(int(tiles.positive_positions.sum()), 2)
        self.assertEqual(int(tiles.negative_positions.sum()), 12)
        self.assertTrue(np.isclose(tiles.q_positive.sum(), 1.0))
        self.assertTrue(np.isclose(tiles.q_negative.sum(), 1.0))
        self.assertTrue(np.isclose(tiles.q_mixture.sum(), 1.0))

    def test_materialized_profiles_preserve_masks_counts_and_fasta_case(self):
        batch = self.generator.make_batch(self.generator.tile_index)
        self.assertEqual(tuple(batch.x.shape), (3, 5, 8))
        self.assertEqual(tuple(batch.x_both_strands.shape), (6, 5, 8))
        self.assertEqual(tuple(batch.target.shape), (3, 2, 4))
        self.assertEqual(int(batch.mask.sum()), 14)
        self.assertEqual(int(batch.target.sum()), 2)
        self.assertEqual(float(batch.count.sum()), 5.0)
        self.assertTrue(torch.all(batch.x.sum(dim=1) == 1))
        self.assertTrue(torch.any((batch.raw_codes >= 4) & (batch.raw_codes <= 7)))
        self.assertTrue(torch.all(batch.target.bool() <= batch.mask))
        self.assertTrue(torch.all(batch.loss_weight[~batch.mask] == 0))
        torch.testing.assert_close(batch.plus_target, batch.target[:, 0:1, :])
        torch.testing.assert_close(
            batch.minus_target_reverse,
            batch.target[:, 1:2, :].flip(-1),
        )

    def test_sampling_and_two_strand_forward(self):
        batch = self.generator.sample_batch(batch_size=2)
        self.assertCountEqual(batch.sampling_branch, ("positive", "negative"))
        model = create_cnn_baseline(channels=4, block_settings=())
        plus_logits, minus_reverse_logits = forward_both_strands(
            model,
            batch.x_both_strands,
            batch_size=len(batch.tiles),
            context_flank=self.config.context_flank,
            target_length=self.config.target_length,
        )
        self.assertEqual(plus_logits.shape, (2, 1, 4))
        self.assertEqual(minus_reverse_logits.shape, (2, 1, 4))
        loss = dml_weighted_profile_bce_with_logits(
            plus_logits,
            minus_reverse_logits,
            batch.plus_target,
            batch.minus_target_reverse,
            batch.plus_loss_weight,
            batch.minus_loss_weight_reverse,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.backbone.stem.weight.grad)

    def test_streaming_evaluation_covers_population_and_restores_mode(self):
        class TinyProfileModel(torch.nn.Module):
            def forward(self, x):
                return 4.0 * x[:, 0:1, :] - 2.0

        model = TinyProfileModel()
        model.train()
        result = evaluate_profile_model(
            model,
            self.generator,
            device=torch.device("cpu"),
            batch_size=2,
            score_decimals=5,
            progress=None,
        )

        self.assertTrue(model.training)
        self.assertEqual(result.processed_tiles, 3)
        self.assertEqual(result.processed_positions, 14)
        self.assertEqual(result.processed_positives, 2)
        self.assertEqual(result.fallback_warnings, ())
        self.assertEqual(result.metrics.segment.tolist(), ["overall", "c1"])
        self.assertEqual(int(result.overall.position_strand), 14)
        self.assertEqual(int(result.overall.positives), 2)
        self.assertTrue(np.isfinite(float(result.overall.average_precision)))
        self.assertTrue(np.isfinite(float(result.overall.epic_spearman)))

    def test_multitask_forward_and_weighted_huber(self):
        batch = self.generator.sample_batch(batch_size=2)
        model = create_cnn_presence_intensity(
            channels=4,
            block_settings=(),
        )
        outputs = forward_both_strands_multitask(
            model,
            batch.x_both_strands,
            batch_size=2,
            context_flank=self.config.context_flank,
            target_length=self.config.target_length,
        )
        for output in outputs:
            self.assertEqual(output.shape, (2, 1, 4))

        plus_intensity = torch.log1p(batch.count[:, 0:1, :])
        plus_mask = (batch.count[:, 0:1, :] > 0).float()
        plus_weight = 2.0 * batch.plus_loss_weight * plus_mask
        loss = dml_weighted_huber(
            outputs[2],
            plus_intensity,
            plus_weight,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.profile_head.weight.grad)

        result = evaluate_presence_intensity_model(
            model,
            self.generator,
            device=torch.device("cpu"),
            batch_size=2,
            progress=None,
        )
        self.assertEqual(
            set(result.metrics.score),
            {
                "presence_probability",
                "intensity_prediction",
                "expected_signal",
            },
        )
        self.assertEqual(len(result.metrics), 6)


if __name__ == "__main__":
    unittest.main()
