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
        """Prepare probe data from Twin Gas dataset.
        
        IMPORTANT: Must match TCN downstream data processing EXACTLY:
        1. Load raw sensor data (no timestamp)
        2. Truncate to gas response segment (truncate_ratio, truncate_start_ratio)
        3. Downsample to seq_len
        4. Apply per-channel z-score normalization
        5. Transpose to [C, T]
        """
        if self._twin_gas is None:
            return
            
        # Get sensor embedding for index lookup
        SensorEmbedding = _get_sensor_embedding()
        self._sensor_embed = SensorEmbedding(1)
        
        # TCN training parameters - MUST match training config
        truncate_ratio = 0.25
        truncate_start_ratio = 0.0666
        
        # Sample indices
        n_samples = min(self.num_probe_samples, len(self._twin_gas))
        indices = np.random.choice(len(self._twin_gas), n_samples, replace=False)
        
        probe_data = []
        probe_labels = []
        
        for idx in indices:
            # Get raw sample (not normalized) - mimic TCN data loading
            rec = self._twin_gas._samples[idx]
            df, target = self._twin_gas._load_sample(rec)
            
            # Convert to numpy [T, C]
            data_raw = df.values.astype(np.float32)  # [T, C]
            orig_len = data_raw.shape[0]
            
            # Step 1: Truncate to gas response segment (SAME as TCN training)
            start_idx = int(orig_len * truncate_start_ratio)
            end_idx = int(orig_len * truncate_ratio)
            start_idx = max(0, min(start_idx, orig_len - 1))
            end_idx = max(start_idx + 1, min(end_idx, orig_len))
            data_raw = data_raw[start_idx:end_idx]
            
            T, C = data_raw.shape
            
            # Step 2: Downsample to seq_len (like TCN _eval_downsample)
            if T > self.seq_len:
                idx_t = np.linspace(0, T - 1, self.seq_len, dtype=np.float32)
                idx_t = np.rint(idx_t).astype(np.int64)
                idx_t = np.clip(idx_t, 0, T - 1)
                data_raw = data_raw[idx_t]
            elif T < self.seq_len:
                pad = np.zeros((self.seq_len - T, C), dtype=np.float32)
                data_raw = np.concatenate([data_raw, pad], axis=0)
            
            # Step 3: per-channel z-score normalization AFTER truncate+downsample
            mean = data_raw.mean(axis=0, keepdims=True)
            std = data_raw.std(axis=0, keepdims=True)
            std = np.where(std < 1e-6, 1.0, std)
            data_norm = (data_raw - mean) / std
            
            # Step 4: Transpose to [C, T]
            data_norm = data_norm.T.astype(np.float32)
            
            probe_data.append(data_norm)
            
            # Get labels
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
            "channel_models": self._twin_gas.channel_models,
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
        
        data_np = self._probe_data["data"]
        N, C, T = data_np.shape
        
        # Get model hyperparams
        max_channels = pl_module.hparams.get("max_channels", 16)
        max_length = pl_module.hparams.get("max_length", 4096)
        
        # PERF: Use much shorter sequence for probe - 1000 is enough for Twin Gas
        # Padding to 4096 causes massive memory usage in Transformer attention
        patch_size = pl_module.hparams.get("patch_size", 16)
        # Round up to nearest multiple of patch_size for model compatibility
        probe_length_padded = ((min(T, 1024) + patch_size - 1) // patch_size) * patch_size
        # Keep original length for data slicing
        probe_length = min(T, 1024)
        
        # Sample rate (approximate after downsampling)
        original_rate = self._probe_data["sample_rate"]
        approx_rate = original_rate * (probe_length / 60000)
        
        # PERF: Small batch size to limit GPU memory
        batch_size = 4
        all_recon = []
        
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            batch_data = torch.tensor(
                data_np[batch_start:batch_end, :, :probe_length], 
                dtype=torch.float32, device=device
            )
            B = batch_data.shape[0]
            
            # Get sensor indices
            sensor_indices = torch.zeros(B, C, dtype=torch.long, device=device)
            for i, model in enumerate(self._probe_data["channel_models"]):
                if i < C:
                    sensor_indices[:, i] = self._sensor_embed.get_sensor_idx(model)
            
            sample_rate = torch.full((B,), approx_rate, dtype=torch.float32, device=device)
            
            # Create TWO masks:
            # 1. valid_channel_mask: True = valid channel (not padding)
            # 2. training_mask: True = visible, False = masked for imputation
            valid_channel_mask = torch.ones(B, C, dtype=torch.bool, device=device)
            training_mask = torch.ones(B, C, dtype=torch.bool, device=device)
            for ch in self.mask_channels:
                if ch < C:
                    training_mask[:, ch] = False  # Mark as masked for imputation
            
            # Mask the data (zero out masked channels)
            batch_masked = batch_data.clone()
            for ch in self.mask_channels:
                if ch < C:
                    batch_masked[:, ch, :] = 0
            
            # Pad channels if needed
            if C < max_channels:
                pad_c = max_channels - C
                batch_masked = F.pad(batch_masked, (0, 0, 0, pad_c), value=0)
                valid_channel_mask = F.pad(valid_channel_mask, (0, pad_c), value=False)  # Padding is not valid
                training_mask = F.pad(training_mask, (0, pad_c), value=True)  # Padding not masked for training
                sensor_indices = F.pad(sensor_indices, (0, pad_c), value=0)
            elif C > max_channels:
                batch_masked = batch_masked[:, :max_channels, :]
                valid_channel_mask = valid_channel_mask[:, :max_channels]
                training_mask = training_mask[:, :max_channels]
                sensor_indices = sensor_indices[:, :max_channels]
            
            # PERF: Only pad to probe_length_padded, not max_length (huge memory saving)
            curr_T = batch_masked.shape[2]
            if curr_T < probe_length_padded:
                batch_masked = F.pad(batch_masked, (0, probe_length_padded - curr_T), value=0)
            
            # Forward pass with autocast for memory efficiency
            # Pass BOTH valid_channel_mask (for padding) and training_mask (for mask_token)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = pl_module(batch_masked, valid_channel_mask, sensor_indices, sample_rate, training_mask=training_mask)
            x_recon = outputs["x_recon"][:, :min(C, max_channels), :probe_length].float()
            all_recon.append(x_recon.cpu())
            
            # PERF: Clear intermediate tensors
            del batch_masked, outputs
            torch.cuda.empty_cache()
        
        # Concatenate all batches - keep on CPU for MSE calculation to save GPU memory
        x_recon = torch.cat(all_recon, dim=0)
        data = torch.tensor(data_np[:, :min(C, max_channels), :probe_length], dtype=torch.float32)
        
        # CRITICAL: Scale correction for imputed channels
        # VQ-VAE output range is compressed, need to rescale to match original data distribution
        # For each masked channel, rescale recon to match the mean/std of visible channels in original data
        x_recon_scaled = x_recon.clone()
        for ch in self.mask_channels:
            if ch < min(C, max_channels):
                # Get stats from original data (all samples, this channel)
                orig_mean = data[:, ch, :].mean()
                orig_std = data[:, ch, :].std()
                # Get stats from reconstructed data
                recon_mean = x_recon[:, ch, :].mean()
                recon_std = x_recon[:, ch, :].std()
                # Rescale: (x - recon_mean) / recon_std * orig_std + orig_mean
                if recon_std > 1e-6:
                    x_recon_scaled[:, ch, :] = (x_recon[:, ch, :] - recon_mean) / recon_std * orig_std + orig_mean
        
        # Calculate imputation MSE on CPU (memory efficient) - use scaled version
        mse_masked = 0.0
        count = 0
        for ch in self.mask_channels:
            if ch < C:
                mse_ch = F.mse_loss(x_recon[:, ch, :], data[:, ch, :]).item()
                mse_masked += mse_ch
                count += 1
        mse_masked = mse_masked / max(count, 1)
        
        # Calculate reconstruction MSE on visible channels
        C_eff = min(C, max_channels)
        mse_visible = 0.0
        count = 0
        for ch in range(C_eff):
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
        
        # If TCN model available, evaluate downstream R² using SCALED reconstruction
        if self._tcn_model is not None:
            r2_metrics = self._evaluate_downstream_r2(data, x_recon_scaled)
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
            data_original: [N, C, T] original data (on CPU)
            data_imputed: [N, C, T] data with imputed channels (on CPU)
            
        Returns:
            Dictionary with R² metrics
        """
        device = next(self._tcn_model.parameters()).device
        N = data_original.shape[0]
        
        # Prepare data variants on CPU first
        data_with_imputation = data_original.clone()
        for ch in self.mask_channels:
            if ch < data_original.shape[1]:
                data_with_imputation[:, ch, :] = data_imputed[:, ch, :]
        
        data_zeroed = data_original.clone()
        for ch in self.mask_channels:
            if ch < data_original.shape[1]:
                data_zeroed[:, ch, :] = 0
        
        # Get ground truth labels
        y_reg = torch.tensor(
            [label["ppm"] for label in self._probe_data["labels"]],
            dtype=torch.float32,
        )
        
        # PERF: Process TCN inference in batches to avoid OOM
        batch_size = 16  # TCN is much lighter than VQ-VAE
        reg_original_list = []
        reg_imputed_list = []
        reg_zeroed_list = []
        
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            
            # Original data
            batch_orig = data_original[i:j].to(device)
            _, reg = self._tcn_model(batch_orig)
            reg_original_list.append(reg.cpu())
            
            # Imputed data
            batch_imp = data_with_imputation[i:j].to(device)
            _, reg = self._tcn_model(batch_imp)
            reg_imputed_list.append(reg.cpu())
            
            # Zeroed data
            batch_zero = data_zeroed[i:j].to(device)
            _, reg = self._tcn_model(batch_zero)
            reg_zeroed_list.append(reg.cpu())
            
            # Clear GPU memory
            del batch_orig, batch_imp, batch_zero
            torch.cuda.empty_cache()
        
        reg_original = torch.cat(reg_original_list)
        reg_imputed = torch.cat(reg_imputed_list)
        reg_zeroed = torch.cat(reg_zeroed_list)
        
        # Calculate R² scores on CPU
        from torchmetrics import R2Score
        r2_metric = R2Score()
        
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
