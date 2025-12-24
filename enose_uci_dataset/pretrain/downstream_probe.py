"""Downstream task probe for monitoring pretraining effectiveness.

This module provides a callback that evaluates the pretrained model's
utility for channel imputation on the Twin Gas dataset during validation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.callbacks import Callback

# Lazy imports to avoid circular dependencies
def _get_twin_gas_dataset():
    from ..datasets import TwinGasSensorArrays
    return TwinGasSensorArrays

def _get_sensor_embedding():
    from .model import SensorEmbedding
    return SensorEmbedding


class DownstreamProbeCallback(Callback):
    """Callback to evaluate pretrained model on downstream channel imputation task.
    
    During validation, this callback:
    1. Loads Twin Gas test samples
    2. Masks specified channels
    3. Uses the pretrained VQ-VAE to impute masked channels
    4. Evaluates imputation quality (MSE) and optionally downstream R² if TCN provided
    
    This provides early feedback on whether the pretraining is useful for the
    intended downstream task.
    """
    
    def __init__(
        self,
        twin_gas_root: str = ".cache",
        tcn_checkpoint: Optional[str] = None,
        probe_every_n_epochs: int = 10,
        num_probe_samples: int = 50,
        mask_channels: List[int] = [0, 1],  # Channels to mask for probing
        seq_len: int = 1000,
        device: Optional[str] = None,
    ):
        """Initialize the downstream probe.
        
        Args:
            twin_gas_root: Root directory for Twin Gas dataset
            tcn_checkpoint: Path to trained TCN checkpoint (optional)
            probe_every_n_epochs: Run probe every N epochs
            num_probe_samples: Number of samples to use for probing
            mask_channels: Which channels to mask during probing
            seq_len: Sequence length for downstream task (default 1000)
            device: Device to run probe on
        """
        super().__init__()
        self.twin_gas_root = Path(twin_gas_root)
        self.tcn_checkpoint = tcn_checkpoint
        self.probe_every_n_epochs = probe_every_n_epochs
        self.num_probe_samples = num_probe_samples
        self.mask_channels = mask_channels
        self.seq_len = seq_len
        self._device = device
        
        self._twin_gas = None
        self._tcn_model = None
        self._probe_data = None
        self._sensor_embed = None
        
    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        """Load Twin Gas dataset and optionally TCN model."""
        if stage != "fit":
            return
            
        # Determine device
        if self._device is None:
            self._device = pl_module.device
        
        # Load Twin Gas dataset
        TwinGasSensorArrays = _get_twin_gas_dataset()
        try:
            self._twin_gas = TwinGasSensorArrays(str(self.twin_gas_root), download=True)
            print(f"[DownstreamProbe] Loaded Twin Gas dataset: {len(self._twin_gas)} samples")
        except Exception as e:
            print(f"[DownstreamProbe] Failed to load Twin Gas dataset: {e}")
            return
        
        # Prepare probe data
        self._prepare_probe_data()
        
        # Load TCN model if checkpoint provided
        if self.tcn_checkpoint and Path(self.tcn_checkpoint).exists():
            self._load_tcn_model()
    
    def _prepare_probe_data(self) -> None:
        """Prepare probe data from Twin Gas dataset."""
        if self._twin_gas is None:
            return
            
        # Get sensor embedding for index lookup
        SensorEmbedding = _get_sensor_embedding()
        self._sensor_embed = SensorEmbedding(1)
        
        # Sample indices
        n_samples = min(self.num_probe_samples, len(self._twin_gas))
        indices = np.random.choice(len(self._twin_gas), n_samples, replace=False)
        
        probe_data = []
        probe_labels = []
        
        for idx in indices:
            data, meta = self._twin_gas.get_normalized_sample(idx)
            # data: [C, T] normalized
            
            # Downsample to seq_len
            C, T = data.shape
            if T > self.seq_len:
                # Uniform downsample
                indices_t = np.linspace(0, T - 1, self.seq_len, dtype=int)
                data = data[:, indices_t]
            elif T < self.seq_len:
                # Pad
                pad_width = ((0, 0), (0, self.seq_len - T))
                data = np.pad(data, pad_width, mode='constant', constant_values=0)
            
            probe_data.append(data)
            
            # Get labels if available
            target = meta.get("target", {})
            if isinstance(target, dict):
                probe_labels.append({
                    "gas": target.get("gas", 0),
                    "ppm": target.get("ppm", 0.0),
                })
            else:
                probe_labels.append({"gas": 0, "ppm": 0.0})
        
        self._probe_data = {
            "data": np.stack(probe_data, axis=0),  # [N, C, T]
            "labels": probe_labels,
            "channel_models": self._twin_gas.channel_models,  # ['TGS2611', 'TGS2612', ...]
            "sample_rate": self._twin_gas.sample_rate_hz,
        }
        print(f"[DownstreamProbe] Prepared {n_samples} probe samples, shape: {self._probe_data['data'].shape}")
    
    def _load_tcn_model(self) -> None:
        """Load pretrained TCN model for downstream evaluation."""
        try:
            # Import here to avoid dependency
            import sys
            sys.path.insert(0, str(Path(__file__).parents[2] / "examples"))
            from train_twin_timeseries import TwinGasLitModule
            
            self._tcn_model = TwinGasLitModule.load_from_checkpoint(
                self.tcn_checkpoint,
                map_location=self._device,
            )
            self._tcn_model.eval()
            self._tcn_model.freeze()
            print(f"[DownstreamProbe] Loaded TCN model from {self.tcn_checkpoint}")
        except Exception as e:
            print(f"[DownstreamProbe] Failed to load TCN model: {e}")
            self._tcn_model = None
    
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Run downstream probe at the end of validation epoch."""
        if self._probe_data is None:
            return
            
        # Only run every N epochs
        current_epoch = trainer.current_epoch
        if current_epoch % self.probe_every_n_epochs != 0:
            return
        
        print(f"\n[DownstreamProbe] Running probe at epoch {current_epoch}...")
        
        # Run imputation and evaluate
        metrics = self._evaluate_imputation(pl_module)
        
        # Log metrics
        for name, value in metrics.items():
            pl_module.log(f"probe/{name}", value, on_step=False, on_epoch=True)
            print(f"  {name}: {value:.4f}")
    
    @torch.no_grad()
    def _evaluate_imputation(self, pl_module: L.LightningModule) -> Dict[str, float]:
        """Evaluate imputation quality on probe data."""
        pl_module.eval()
        device = pl_module.device
        
        data = torch.tensor(self._probe_data["data"], dtype=torch.float32, device=device)
        N, C, T = data.shape
        
        # Get sensor indices
        sensor_indices = torch.zeros(N, C, dtype=torch.long, device=device)
        for i, model in enumerate(self._probe_data["channel_models"]):
            sensor_indices[:, i] = self._sensor_embed.get_sensor_idx(model)
        
        # Sample rate (approximate after downsampling)
        original_rate = self._probe_data["sample_rate"]
        # After downsampling from original length to seq_len
        approx_rate = original_rate * (self.seq_len / 60000)  # ~60000 original timesteps
        sample_rate = torch.full((N,), approx_rate, dtype=torch.float32, device=device)
        
        # Create mask (True = visible, False = masked)
        channel_mask = torch.ones(N, C, dtype=torch.bool, device=device)
        for ch in self.mask_channels:
            if ch < C:
                channel_mask[:, ch] = False
        
        # Mask the data
        data_masked = data.clone()
        for ch in self.mask_channels:
            if ch < C:
                data_masked[:, ch, :] = 0
        
        # Run through pretrained model
        # Need to pad to max_channels if necessary
        max_channels = pl_module.hparams.get("max_channels", 20)
        if C < max_channels:
            pad_c = max_channels - C
            data_masked = F.pad(data_masked, (0, 0, 0, pad_c), value=0)
            channel_mask = F.pad(channel_mask, (0, pad_c), value=False)
            sensor_indices = F.pad(sensor_indices, (0, pad_c), value=0)
        
        # Pad time dimension if needed
        patch_size = pl_module.hparams.get("patch_size", 16)
        max_length = pl_module.hparams.get("max_length", 4096)
        if T < max_length:
            data_masked = F.pad(data_masked, (0, max_length - T), value=0)
        elif T > max_length:
            data_masked = data_masked[:, :, :max_length]
        
        # Forward pass
        outputs = pl_module(data_masked, channel_mask, sensor_indices, sample_rate)
        x_recon = outputs["x_recon"][:, :C, :T]  # Trim back to original size
        
        # Calculate imputation MSE (only on masked channels)
        mse_masked = 0.0
        count = 0
        for ch in self.mask_channels:
            if ch < C:
                mse_ch = F.mse_loss(x_recon[:, ch, :], data[:, ch, :]).item()
                mse_masked += mse_ch
                count += 1
        mse_masked = mse_masked / max(count, 1)
        
        # Calculate reconstruction MSE on visible channels
        mse_visible = 0.0
        count = 0
        for ch in range(C):
            if ch not in self.mask_channels:
                mse_ch = F.mse_loss(x_recon[:, ch, :], data[:, ch, :]).item()
                mse_visible += mse_ch
                count += 1
        mse_visible = mse_visible / max(count, 1)
        
        metrics = {
            "imputation_mse": mse_masked,
            "visible_mse": mse_visible,
            "num_masked_channels": len(self.mask_channels),
        }
        
        # If TCN model available, evaluate downstream R²
        if self._tcn_model is not None:
            r2_metrics = self._evaluate_downstream_r2(data, x_recon)
            metrics.update(r2_metrics)
        
        pl_module.train()
        return metrics
    
    @torch.no_grad()
    def _evaluate_downstream_r2(
        self, 
        data_original: torch.Tensor, 
        data_imputed: torch.Tensor,
    ) -> Dict[str, float]:
        """Evaluate R² on downstream TCN model.
        
        Args:
            data_original: [N, C, T] original data
            data_imputed: [N, C, T] data with imputed channels
            
        Returns:
            Dictionary with R² metrics
        """
        device = next(self._tcn_model.parameters()).device
        
        # Move data to TCN device
        data_original = data_original.to(device)
        data_imputed = data_imputed.to(device)
        
        # Get ground truth labels
        y_reg = torch.tensor(
            [label["ppm"] for label in self._probe_data["labels"]],
            dtype=torch.float32,
            device=device,
        )
        
        # Evaluate with original data (baseline)
        _, reg_original = self._tcn_model(data_original)
        
        # Evaluate with imputed data
        # Replace masked channels with imputed values
        data_with_imputation = data_original.clone()
        for ch in self.mask_channels:
            if ch < data_original.shape[1]:
                data_with_imputation[:, ch, :] = data_imputed[:, ch, :]
        
        _, reg_imputed = self._tcn_model(data_with_imputation)
        
        # Evaluate with zeroed channels (degraded baseline)
        data_zeroed = data_original.clone()
        for ch in self.mask_channels:
            if ch < data_original.shape[1]:
                data_zeroed[:, ch, :] = 0
        
        _, reg_zeroed = self._tcn_model(data_zeroed)
        
        # Calculate R² scores
        from torchmetrics import R2Score
        r2_metric = R2Score().to(device)
        
        r2_original = r2_metric(reg_original, y_reg).item()
        r2_metric.reset()
        r2_imputed = r2_metric(reg_imputed, y_reg).item()
        r2_metric.reset()
        r2_zeroed = r2_metric(reg_zeroed, y_reg).item()
        
        # Recovery rate
        if r2_original > 0:
            recovery_rate = r2_imputed / r2_original
        else:
            recovery_rate = 0.0
        
        return {
            "r2_original": r2_original,
            "r2_imputed": r2_imputed,
            "r2_zeroed": r2_zeroed,
            "r2_recovery_rate": recovery_rate,
        }
