"""Shape, alignment and zero-initialisation tests for CNN-EXP-049."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.epic_cnn import (  # noqa: E402
    ConditionalLocalizerB,
    FiLMAdapter,
    RegionalFeatureCache,
    RegionalWindowGenerator,
    RegionalDetectorA,
    RegionalRateDetectorA,
    broadcast_r16_condition,
    encode_raw_dna_six_channel,
    reverse_complement_six_channel,
    theoretical_receptive_field,
)
from ml_project.epic_cnn.cascade_data import predict_contig_to_regional_cache  # noqa: E402
from ml_project.sequence_context import encode_fasta_sequence  # noqa: E402
from scripts.run_cnn_exp049_cloud import (  # noqa: E402
    fold_assignment,
    require_completed_checkpoint,
    validate_oof_cache,
)
from scripts.run_cnn_exp049b_joint_pilot import (  # noqa: E402
    _cosine_factor,
    _gather_regions,
    aligned_long_raw,
    regional_targets_from_profile,
)


class FakeRegionalSplit:
    def __init__(self):
        self._intervals = pd.DataFrame(
            {
                "contig": ["c1", "c1"],
                "start": [0, 0],
                "end": [256, 256],
                "strand": ["+", "-"],
            }
        )
        self._positives = pd.DataFrame(
            {
                "contig": ["c1", "c1"],
                "coordinate_0based": [17, 130],
                "strand": ["+", "-"],
                "count": [1, 2],
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


class FakeOofSplit:
    def __init__(self):
        self._intervals = pd.DataFrame(
            {
                "contig": ["c1", "c2", "c3", "c4"],
                "start": [0, 0, 0, 0],
                "end": [16, 16, 16, 16],
                "strand": ["+", "+", "+", "+"],
            }
        )

    def intervals(self, part):
        if part != "train":
            raise ValueError(part)
        return self._intervals


class TestCascadeEncoding(unittest.TestCase):
    def test_six_channel_encoding_and_reverse_complement(self):
        raw = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.uint8)
        encoded = encode_raw_dna_six_channel(raw)
        self.assertEqual(tuple(encoded.shape), (1, 6, 9))
        torch.testing.assert_close(encoded[:, :4, :].sum(dim=1), (raw < 8).float())
        torch.testing.assert_close(encoded[:, 4, :], ((raw >= 4) & (raw < 8)).float())
        torch.testing.assert_close(encoded[:, 5, :], (raw == 8).float())
        round_trip = reverse_complement_six_channel(
            reverse_complement_six_channel(encoded)
        )
        torch.testing.assert_close(round_trip, encoded)

    def test_exact_r16_broadcast(self):
        condition = torch.tensor([[[1.0, 2.0], [10.0, 20.0]]])
        broadcast = broadcast_r16_condition(condition)
        self.assertEqual(tuple(broadcast.shape), (1, 2, 32))
        self.assertTrue(torch.all(broadcast[:, :, :16] == condition[:, :, 0:1]))
        self.assertTrue(torch.all(broadcast[:, :, 16:] == condition[:, :, 1:2]))


class TestCascadeModels(unittest.TestCase):
    def test_regional_detector_shapes(self):
        model = RegionalDetectorA(
            local_channels=8,
            tower_channels=(8, 12, 16, 20),
            latent_channels=4,
        )
        x = torch.randn(2, 6, 256)
        activity, latent = model(x)
        self.assertEqual(tuple(activity.shape), (2, 1, 16))
        self.assertEqual(tuple(latent.shape), (2, 4, 16))

    def test_regional_rate_latent_is_on_supervised_gradient_path(self):
        model = RegionalRateDetectorA(
            local_channels=8,
            tower_channels=(8, 12, 16, 20),
            latent_channels=4,
        ).eval()
        log_rate, latent = model(torch.randn(1, 6, 256))
        self.assertEqual(tuple(log_rate.shape), (1, 1, 16))
        self.assertEqual(tuple(latent.shape), (1, 4, 16))
        log_rate.square().mean().backward()
        self.assertIsNotNone(model.latent_head.weight.grad)
        self.assertGreater(float(model.latent_head.weight.grad.norm()), 0.0)

    def test_zero_initialised_film_is_identity(self):
        adapter = FiLMAdapter(5, 8)
        x = torch.randn(2, 8, 31)
        condition = torch.randn(2, 5, 31)
        torch.testing.assert_close(adapter(x, condition), x)

    def test_localizer_zero_residual_and_shapes(self):
        model = ConditionalLocalizerB(
            channels=8,
            condition_channels=5,
            film_after_blocks=(2, 4, 6, 8),
        )
        dna = torch.randn(2, 6, 96)
        condition = torch.randn(2, 5, 96)
        output = model(dna, condition)
        self.assertEqual(model.receptive_field, 1029)
        self.assertEqual(theoretical_receptive_field(), 1029)
        for name in ("final_logit", "base_logit", "delta_logit", "intensity"):
            self.assertEqual(tuple(output[name].shape), (2, 1, 96))
        torch.testing.assert_close(output["delta_logit"], torch.zeros_like(output["delta_logit"]))
        torch.testing.assert_close(output["final_logit"], output["base_logit"])


class TestCascadeData(unittest.TestCase):
    def test_joint_continuation_cosine_contract(self):
        self.assertAlmostEqual(_cosine_factor(0, 30_000), 1.0)
        self.assertAlmostEqual(_cosine_factor(30_000, 30_000), 0.1)
        self.assertGreater(_cosine_factor(15_000, 30_000), 0.1)

    def test_joint_long_window_alignment_and_regional_gather(self):
        tiles = pd.DataFrame({"contig": ["c1"], "target_start": [1024]})
        sequence = (np.arange(10_000, dtype=np.int64) % 9).astype(np.uint8)
        raw, starts = aligned_long_raw(
            tiles,
            {"c1": sequence},
            target_length=1024,
        )
        self.assertEqual(tuple(raw.shape), (1, 8192))
        self.assertEqual(int(starts[0]) % 16, 0)
        target_offset = 1024 - int(starts[0])
        np.testing.assert_array_equal(
            raw[0, target_offset : target_offset + 1024].numpy(),
            sequence[1024:2048],
        )

        features = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)
        indices = torch.tensor([[0, 0, 2, 5]])
        gathered = _gather_regions(features, indices)
        self.assertEqual(tuple(gathered.shape), (1, 2, 4))
        torch.testing.assert_close(gathered[0, 0], torch.tensor([0.0, 0.0, 2.0, 5.0]))
        torch.testing.assert_close(gathered[0, 1], torch.tensor([6.0, 6.0, 8.0, 11.0]))

    def test_joint_regional_targets_preserve_strands_and_counts(self):
        class Batch:
            tiles = pd.DataFrame({"tile_id": [0]})
            count = torch.zeros(1, 2, 32)
            mask = torch.ones(1, 2, 32, dtype=torch.bool)

        batch = Batch()
        batch.count[0, 0, 1] = 2
        batch.count[0, 0, 17] = 3
        batch.count[0, 1, 31] = 5
        count, target, mask = regional_targets_from_profile(batch, torch.device("cpu"))
        self.assertEqual(tuple(count.shape), (2, 1, 2))
        torch.testing.assert_close(count[:, 0], torch.tensor([[2.0, 3.0], [0.0, 5.0]]))
        torch.testing.assert_close(target, (count > 0).float())
        self.assertTrue(bool(mask.all()))

    def test_regional_generator_orients_minus_labels(self):
        sequence = encode_fasta_sequence("ACGT" * 64)
        generator = RegionalWindowGenerator(
            FakeRegionalSplit(),
            {"c1": sequence},
            window_bp=256,
            positive_window_fraction=1.0,
            seed=1,
        )
        batch = generator.sample_batch(4)
        self.assertEqual(tuple(batch.raw_codes.shape), (4, 256))
        self.assertEqual(tuple(batch.target.shape), (4, 1, 16))
        self.assertEqual(tuple(batch.count.shape), (4, 1, 16))
        torch.testing.assert_close(batch.target, (batch.count > 0).float())
        self.assertTrue(torch.all(batch.target.bool() <= batch.mask))
        oriented = generator.encode_oriented(batch, torch.device("cpu"))
        self.assertEqual(tuple(oriented.shape), (4, 6, 256))

    def test_cache_lookup_and_model_prediction_alignment(self):
        sequence = encode_fasta_sequence("ACGT" * 64)
        with tempfile.TemporaryDirectory() as temporary:
            cache = RegionalFeatureCache.create(temporary, channels=5)
            plus = np.arange(16 * 5, dtype=np.float16).reshape(16, 5)
            minus = plus + 1000
            cache.write_contig("c1", plus, minus, length_bp=256)
            lookup = cache.lookup_window("c1", "+", 14, 5)
            self.assertEqual(tuple(lookup.shape), (5, 5))
            np.testing.assert_array_equal(lookup[:, 0], plus[0])
            np.testing.assert_array_equal(lookup[:, -1], plus[1])
            reverse = cache.lookup_window(
                "c1", "-", 14, 5, reverse_to_oriented=True
            )
            np.testing.assert_array_equal(reverse[:, 0], minus[1])
            cache.close()

        with tempfile.TemporaryDirectory() as temporary:
            model = RegionalDetectorA(
                local_channels=8,
                tower_channels=(8, 12, 16, 20),
                latent_channels=4,
            )
            cache = RegionalFeatureCache.create(temporary, channels=5)
            predict_contig_to_regional_cache(
                model,
                sequence,
                contig="c1",
                cache=cache,
                device=torch.device("cpu"),
                input_bp=256,
                output_bp=128,
            )
            self.assertTrue(cache.has_contig("c1"))
            self.assertEqual(cache._array("c1", "+").shape, (16, 5))
            cache.close()

    def test_oof_cache_contract_guard(self):
        split = FakeOofSplit()
        folds = fold_assignment(["c1", "c2", "c3", "c4"], 2, 7)
        expected = {
            contig: fold for fold, contigs in enumerate(folds) for contig in contigs
        }
        with tempfile.TemporaryDirectory() as temporary:
            cache = RegionalFeatureCache.create(temporary, channels=5)
            values = np.zeros((1, 5), dtype=np.float16)
            for contig, fold in expected.items():
                cache.write_contig(
                    contig,
                    values,
                    values,
                    length_bp=16,
                    source={
                        "fold": fold,
                        "oof": True,
                        "checkpoint_step": 2,
                        "training_contract_sha256": "a" * 64,
                    },
                )
            validate_oof_cache(cache, split, fold_count=2, fold_seed=7)
            cache.manifest["contigs"]["c1"]["source"]["oof"] = False
            with self.assertRaisesRegex(RuntimeError, "not marked oof"):
                validate_oof_cache(cache, split, fold_count=2, fold_seed=7)
            cache.close()

    def test_incomplete_checkpoint_guard(self):
        require_completed_checkpoint({"step": 2, "total_steps": 2}, "tiny")
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            require_completed_checkpoint({"step": 1, "total_steps": 2}, "tiny")


if __name__ == "__main__":
    unittest.main()
