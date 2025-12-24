"""Tests for the pretraining module."""
import sys
from pathlib import Path
from unittest import TestCase, main, skipIf

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from enose_uci_dataset.pretrain.model import (
    EnoseVQVAE,
    EnoseEncoder,
    EnoseDecoder,
    VectorQuantizer,
    PatchEmbedding,
    SensorEmbedding,
    PositionalEncoding,
)

DATA_ROOT = Path(".cache")


def has_dataset(name: str) -> bool:
    return (DATA_ROOT / name).exists()


class TestPositionalEncoding(TestCase):
    """Tests for PositionalEncoding."""
    
    def test_output_shape(self):
        pe = PositionalEncoding(d_model=64, max_len=1000)
        x = torch.randn(2, 100, 64)
        out = pe(x)
        self.assertEqual(out.shape, x.shape)
    
    def test_encoding_unique(self):
        pe = PositionalEncoding(d_model=64, max_len=100, dropout=0)
        x = torch.zeros(1, 50, 64)
        out = pe(x)
        # Each position should have unique encoding
        for i in range(out.size(1) - 1):
            self.assertFalse(torch.allclose(out[0, i], out[0, i+1]))


class TestSensorEmbedding(TestCase):
    """Tests for SensorEmbedding."""
    
    def test_known_sensor(self):
        emb = SensorEmbedding(d_model=32)
        idx = emb.get_sensor_idx("TGS2602")
        self.assertGreater(idx, 0)
        self.assertLess(idx, len(emb.SENSOR_MODELS))
    
    def test_unknown_sensor(self):
        emb = SensorEmbedding(d_model=32)
        idx = emb.get_sensor_idx("UnknownSensor")
        unknown_idx = emb.model_to_idx["UNKNOWN"]
        self.assertEqual(idx, unknown_idx)
    
    def test_embedding_output(self):
        emb = SensorEmbedding(d_model=32)
        indices = torch.tensor([0, 1, 2])
        out = emb(indices)
        self.assertEqual(out.shape, (3, 32))


class TestVectorQuantizer(TestCase):
    """Tests for VectorQuantizer."""
    
    def test_output_shape(self):
        vq = VectorQuantizer(num_embeddings=64, embedding_dim=32)
        z = torch.randn(2, 10, 32)
        z_q, loss, perp = vq(z)
        self.assertEqual(z_q.shape, z.shape)
        self.assertEqual(loss.dim(), 0)  # Scalar
        self.assertEqual(perp.dim(), 0)  # Scalar
    
    def test_straight_through(self):
        vq = VectorQuantizer(num_embeddings=64, embedding_dim=32)
        z = torch.randn(2, 10, 32, requires_grad=True)
        z_q, loss, _ = vq(z)
        # Should be able to backprop through z_q
        loss.backward()
        self.assertIsNotNone(z.grad)


class TestPatchEmbedding(TestCase):
    """Tests for PatchEmbedding."""
    
    def test_output_shape(self):
        pe = PatchEmbedding(patch_size=16, in_channels=1, d_model=64)
        x = torch.randn(2, 8, 256)  # [B, C, T]
        patches = pe(x)
        # Expected: [B, C, num_patches, D]
        self.assertEqual(patches.shape, (2, 8, 256 // 16, 64))


class TestEnoseEncoder(TestCase):
    """Tests for EnoseEncoder."""
    
    def test_output_shape(self):
        enc = EnoseEncoder(
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            patch_size=16,
            max_channels=10,
        )
        x = torch.randn(2, 8, 256)
        z = enc(x)
        # [B, C, num_patches, D]
        self.assertEqual(z.shape, (2, 8, 256 // 16, 64))
    
    def test_with_mask(self):
        enc = EnoseEncoder(d_model=64, nhead=4, num_layers=2, patch_size=16, max_channels=10)
        x = torch.randn(2, 8, 256)
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[0, 0] = False  # Mask first channel of first batch
        z = enc(x, mask)
        self.assertEqual(z.shape, (2, 8, 256 // 16, 64))
    
    def test_with_sensor_indices(self):
        enc = EnoseEncoder(d_model=64, nhead=4, num_layers=2, patch_size=16, max_channels=10)
        x = torch.randn(2, 8, 256)
        sensor_idx = torch.zeros(2, 8, dtype=torch.long)
        z = enc(x, sensor_indices=sensor_idx)
        self.assertEqual(z.shape, (2, 8, 256 // 16, 64))


class TestEnoseDecoder(TestCase):
    """Tests for EnoseDecoder."""
    
    def test_output_shape(self):
        dec = EnoseDecoder(
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            patch_size=16,
            max_channels=10,
        )
        z = torch.randn(2, 8, 16, 64)  # [B, C, num_patches, D]
        x_recon = dec(z)
        self.assertEqual(x_recon.shape, (2, 8, 16 * 16))


class TestEnoseVQVAE(TestCase):
    """Tests for EnoseVQVAE Lightning Module."""
    
    @classmethod
    def setUpClass(cls):
        cls.model = EnoseVQVAE(
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            patch_size=16,
            num_embeddings=32,
            max_channels=10,
        )
    
    def test_forward(self):
        x = torch.randn(2, 8, 256)
        outputs = self.model(x)
        self.assertIn("x_recon", outputs)
        self.assertIn("vq_loss", outputs)
        self.assertIn("perplexity", outputs)
        self.assertEqual(outputs["x_recon"].shape, x.shape)
    
    def test_forward_with_mask(self):
        x = torch.randn(2, 8, 256)
        mask = torch.ones(2, 8, dtype=torch.bool)
        sensor_idx = torch.zeros(2, 8, dtype=torch.long)
        outputs = self.model(x, mask, sensor_idx)
        self.assertEqual(outputs["x_recon"].shape, x.shape)
    
    def test_training_step(self):
        batch = {
            "data": torch.randn(4, 8, 256),
            "sensor_indices": torch.zeros(4, 8, dtype=torch.long),
        }
        loss = self.model._compute_loss(batch, "train")
        self.assertIn("loss", loss)
        self.assertIn("recon_loss", loss)
        self.assertIn("vq_loss", loss)
    
    def test_get_latent(self):
        x = torch.randn(2, 8, 256)
        z_q = self.model.get_latent(x)
        self.assertEqual(z_q.shape[:2], (2, 8))


@skipIf(not has_dataset("twin_gas_sensor_arrays"), "Dataset not available")
class TestEnosePretrainingDataModule(TestCase):
    """Tests for EnosePretrainingDataModule."""
    
    @classmethod
    def setUpClass(cls):
        from enose_uci_dataset.pretrain.datamodule import EnosePretrainingDataModule
        cls.dm = EnosePretrainingDataModule(
            root=str(DATA_ROOT),
            datasets=["twin_gas_sensor_arrays"],
            batch_size=4,
            num_workers=0,
            max_length=256,
            max_channels=10,
        )
        cls.dm.setup()
    
    def test_train_dataloader(self):
        loader = self.dm.train_dataloader()
        batch = next(iter(loader))
        self.assertIn("data", batch)
        self.assertIn("sensor_indices", batch)
        self.assertIn("channel_mask", batch)
        self.assertEqual(batch["data"].shape[1:], (10, 256))
    
    def test_val_dataloader(self):
        loader = self.dm.val_dataloader()
        batch = next(iter(loader))
        self.assertEqual(batch["data"].shape[1:], (10, 256))


@skipIf(not has_dataset("twin_gas_sensor_arrays"), "Dataset not available")
class TestEndToEndTraining(TestCase):
    """End-to-end training test."""
    
    def test_training_loop(self):
        import lightning as L
        from enose_uci_dataset.pretrain import EnoseVQVAE, EnosePretrainingDataModule
        
        dm = EnosePretrainingDataModule(
            root=str(DATA_ROOT),
            datasets=["twin_gas_sensor_arrays"],
            batch_size=4,
            num_workers=0,
            max_length=128,
            max_channels=8,
        )
        
        model = EnoseVQVAE(
            d_model=16,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
            patch_size=16,
            num_embeddings=16,
            max_channels=8,
        )
        
        trainer = L.Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
            limit_train_batches=2,
            limit_val_batches=1,
        )
        
        trainer.fit(model, datamodule=dm)
        self.assertEqual(trainer.current_epoch, 1)


if __name__ == "__main__":
    main()
