"""Baseline models for e-nose pretraining comparison.

Implements MLP and TCN autoencoders with the same training interface as EnoseVQVAE.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import MeanMetric, MinMetric


class BaseAutoencoder(L.LightningModule):
    """Base class for autoencoder models with shared training logic.
    
    Data flow:
    1. Input raw data -> InputBN -> normalized data (internal processing)
    2. Encoder -> latent -> Decoder -> reconstructed normalized data
    3. Inverse BN -> reconstructed raw data (output)
    
    This allows the model to handle raw sensor data directly without external normalization.
    """
    
    def __init__(
        self,
        max_channels: int = 16,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        mask_ratio: float = 0.25,
        lambda_visible: float = 1.0,
        lambda_masked: float = 1.0,
        loss_type: str = "mse",
        huber_delta: float = 1.0,
        optimizer_type: str = "adamw",
        lr_scheduler: str = "cosine_warmup",
        lr_warmup_steps: int = 1000,
        lr_T_mult: int = 2,
        lr_min: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Input normalization (BatchNorm without learnable params for reversibility)
        # affine=False means no gamma/beta, making inverse BN simple: x = y * std + mean
        self.input_bn = nn.BatchNorm1d(max_channels, affine=False, track_running_stats=True)
        
        # Metrics
        self.train_loss = MeanMetric()
        self.train_recon_loss = MeanMetric()
        self.train_recon_visible = MeanMetric()
        self.train_recon_masked = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_recon_visible = MeanMetric()
        self.val_recon_masked = MeanMetric()
        self.val_best_loss = MinMetric()
    
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input using BatchNorm.
        
        Args:
            x: [B, C, T] raw input data
            
        Returns:
            Normalized data [B, C, T]
        """
        return self.input_bn(x)
    
    def denormalize_output(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize output back to raw scale using BN running stats.
        
        Since affine=False, the inverse is simply: x = x_norm * std + mean
        
        Args:
            x_norm: [B, C, T] normalized reconstruction
            
        Returns:
            Denormalized data [B, C, T] in original raw scale
        """
        # Get running stats from BN layer
        mean = self.input_bn.running_mean  # [C]
        var = self.input_bn.running_var    # [C]
        std = torch.sqrt(var + self.input_bn.eps)  # [C]
        
        # Reshape for broadcasting: [C] -> [1, C, 1]
        mean = mean.view(1, -1, 1)
        std = std.view(1, -1, 1)
        
        # Inverse normalization: x = x_norm * std + mean
        return x_norm * std + mean
    
    def _apply_random_mask(
        self,
        x: torch.Tensor,
        valid_channel_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random channel masking only to valid (non-padded) channels."""
        B, C, T = x.shape
        mask = torch.ones(B, C, dtype=torch.bool, device=x.device)
        
        for b in range(B):
            if valid_channel_mask is not None:
                valid_indices = torch.where(valid_channel_mask[b])[0]
                num_valid = len(valid_indices)
                num_mask = int(num_valid * self.hparams.mask_ratio)
                if num_mask > 0 and num_valid > 0:
                    perm = torch.randperm(num_valid, device=x.device)[:num_mask]
                    mask_indices = valid_indices[perm]
                    mask[b, mask_indices] = False
            else:
                num_mask = int(C * self.hparams.mask_ratio)
                if num_mask > 0:
                    mask_indices = torch.randperm(C, device=x.device)[:num_mask]
                    mask[b, mask_indices] = False
        
        x_masked = x.clone()
        x_masked[~mask.unsqueeze(-1).expand_as(x)] = 0
        
        return x_masked, mask
    
    def _compute_recon_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss based on configured loss type."""
        loss_type = self.hparams.loss_type
        
        if loss_type == "mse":
            return F.mse_loss(pred, target)
        elif loss_type == "mae":
            return F.l1_loss(pred, target)
        elif loss_type == "huber":
            return F.huber_loss(pred, target, delta=self.hparams.huber_delta)
        elif loss_type == "cosine":
            if pred.dim() == 1:
                pred = pred.unsqueeze(0)
                target = target.unsqueeze(0)
            cos_sim = F.cosine_similarity(pred, target, dim=-1)
            return 1 - cos_sim.mean()
        elif loss_type == "correlation":
            pred_centered = pred - pred.mean()
            target_centered = target - target.mean()
            pred_std = pred_centered.std() + 1e-8
            target_std = target_centered.std() + 1e-8
            correlation = (pred_centered * target_centered).mean() / (pred_std * target_std)
            return 1 - correlation
        elif loss_type == "mse_corr":
            mse = F.mse_loss(pred, target)
            pred_centered = pred - pred.mean()
            target_centered = target - target.mean()
            pred_std = pred_centered.std() + 1e-8
            target_std = target_centered.std() + 1e-8
            correlation = (pred_centered * target_centered).mean() / (pred_std * target_std)
            return mse + 0.5 * (1 - correlation)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _compute_loss(
        self,
        batch: Dict[str, Any],
        stage: str = "train",
    ) -> Dict[str, torch.Tensor]:
        x_raw = batch["data"]  # [B, C, T] - raw scale
        valid_channel_mask = batch.get("channel_mask")
        
        # Apply mask to raw data
        x_masked, train_mask = self._apply_random_mask(x_raw, valid_channel_mask)
        
        # Forward pass - subclass implements this (same signature as EnoseVQVAE)
        # Input is raw, output includes both raw and normalized reconstruction
        sensor_indices = batch.get("sensor_indices")
        sample_rate = batch.get("sample_rate")
        outputs = self(x_masked, valid_channel_mask, sensor_indices, sample_rate, training_mask=train_mask)
        x_recon_norm = outputs["x_recon_norm"]  # Use normalized output for loss
        
        # Normalize target for loss computation (same as internal BN)
        # Note: We need to normalize the FULL data (not masked) to get the target
        B, C, T = x_raw.shape
        if C < self.hparams.max_channels:
            x_padded = torch.zeros(B, self.hparams.max_channels, T, device=x_raw.device, dtype=x_raw.dtype)
            x_padded[:, :C, :] = x_raw
            x_norm = self.normalize_input(x_padded)[:, :C, :]
        else:
            x_norm = self.normalize_input(x_raw)
        
        train_mask_expanded = train_mask.unsqueeze(-1).expand_as(x_raw)
        
        if valid_channel_mask is not None:
            valid_expanded = valid_channel_mask.unsqueeze(-1).expand_as(x_raw)
            visible_mask = train_mask_expanded & valid_expanded
            masked_mask = (~train_mask_expanded) & valid_expanded
        else:
            visible_mask = train_mask_expanded
            masked_mask = ~train_mask_expanded
        
        # Compute loss in NORMALIZED space
        if visible_mask.any():
            loss_visible = self._compute_recon_loss(x_recon_norm[visible_mask], x_norm[visible_mask])
        else:
            loss_visible = torch.tensor(0.0, device=x_raw.device)
        
        if masked_mask.any():
            loss_masked = self._compute_recon_loss(x_recon_norm[masked_mask], x_norm[masked_mask])
        else:
            loss_masked = torch.tensor(0.0, device=x_raw.device)
        
        recon_loss = (self.hparams.lambda_visible * loss_visible + 
                      self.hparams.lambda_masked * loss_masked)
        
        return {
            "loss": recon_loss,
            "recon_loss": recon_loss,
            "loss_visible": loss_visible,
            "loss_masked": loss_masked,
        }
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        losses = self._compute_loss(batch, stage="train")
        
        self.train_loss(losses["loss"].detach())
        self.train_recon_loss(losses["recon_loss"].detach())
        self.train_recon_visible(losses["loss_visible"].detach())
        self.train_recon_masked(losses["loss_masked"].detach())
        
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/recon_loss", self.train_recon_loss, on_step=False, on_epoch=True)
        self.log("train/recon_visible", self.train_recon_visible, on_step=False, on_epoch=True)
        self.log("train/recon_masked", self.train_recon_masked, on_step=False, on_epoch=True)
        
        return losses["loss"]
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        losses = self._compute_loss(batch, stage="val")
        
        self.val_loss(losses["loss"].detach())
        self.val_recon_visible(losses["loss_visible"].detach())
        self.val_recon_masked(losses["loss_masked"].detach())
        
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recon_loss", losses["recon_loss"].detach(), on_step=False, on_epoch=True)
        self.log("val/recon_visible", self.val_recon_visible, on_step=False, on_epoch=True)
        self.log("val/recon_masked", self.val_recon_masked, on_step=False, on_epoch=True)
        
        return losses["loss"]
    
    def on_validation_epoch_end(self) -> None:
        val_loss = self.val_loss.compute()
        self.val_best_loss(val_loss)
        self.log("val/best_loss", self.val_best_loss.compute(), prog_bar=True)
    
    def configure_optimizers(self):
        optimizer_type = self.hparams.optimizer_type
        
        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif optimizer_type == "soap":
            from enose_uci_dataset.pretrain.optimizers import SOAP
            optimizer = SOAP(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                precondition_frequency=10,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        
        lr_scheduler_type = self.hparams.lr_scheduler
        
        if lr_scheduler_type == "cosine_warmup":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.hparams.lr_warmup_steps,
                T_mult=self.hparams.lr_T_mult,
                eta_min=self.hparams.lr_min,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        elif lr_scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs if self.trainer else 500,
                eta_min=self.hparams.lr_min,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        else:
            return optimizer


class MLPAutoencoder(BaseAutoencoder):
    """MLP-based autoencoder for e-nose time series.
    
    Per-channel MLP processing to avoid memory explosion.
    Architecture: [B, C, T] -> per-channel encode -> cross-channel mixing -> decode -> [B, C, T]
    Target: ~2M parameters
    """
    
    def __init__(
        self,
        d_model: int = 384,
        num_layers: int = 6,
        max_length: int = 1024,
        max_channels: int = 16,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(max_channels=max_channels, **kwargs)
        self.save_hyperparameters()
        
        self.max_length = max_length
        self.max_channels = max_channels
        
        # Per-channel encoder: T -> d_model (shared across channels)
        # Target ~2M total params with d_model=256
        self.channel_encoder = nn.Sequential(
            nn.Linear(max_length, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        # Cross-channel mixing: [B, C, d_model] -> [B, C, d_model]
        mixing_layers = []
        for _ in range(num_layers):
            mixing_layers.extend([
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.channel_mixing = nn.Sequential(*mixing_layers)
        
        # Per-channel decoder: d_model -> T (shared across channels)
        self.channel_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, max_length),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
        sensor_indices: Optional[torch.Tensor] = None,
        sample_rate: Optional[torch.Tensor] = None,
        training_mask: Optional[torch.Tensor] = None,
        return_raw: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, C, T] input time series (RAW scale, not normalized)
            channel_mask: [B, C] padding mask (True = valid)
            sensor_indices: [B, C] sensor model indices (unused)
            sample_rate: [B] sample rate in Hz (unused)
            training_mask: [B, C] training mask (True = visible)
            return_raw: If True, denormalize output to raw scale
            
        Returns:
            Dict with 'x_recon' (raw scale) and 'x_recon_norm' (normalized)
        """
        B, C, T = x.shape
        orig_T = T
        orig_C = C
        
        # Pad channels to max_channels BEFORE normalization
        if C < self.max_channels:
            x_padded = torch.zeros(B, self.max_channels, T, device=x.device, dtype=x.dtype)
            x_padded[:, :C, :] = x
            x = x_padded
        
        # Pad time to max_length
        if T < self.max_length:
            x = F.pad(x, (0, self.max_length - T), value=0)
        elif T > self.max_length:
            x = x[:, :, :self.max_length]
            orig_T = self.max_length
        
        # Input normalization (internal BN)
        x = self.normalize_input(x)
        
        # Per-channel encoding: [B, C, T] -> [B, C, d_model]
        z = self.channel_encoder(x)
        
        # Cross-channel mixing (flatten B*C for efficiency)
        z = self.channel_mixing(z)
        
        # Per-channel decoding: [B, C, d_model] -> [B, C, T]
        x_recon_norm = self.channel_decoder(z)
        
        # Output denormalization (inverse BN) to raw scale
        x_recon = self.denormalize_output(x_recon_norm)
        
        # Crop to original size
        x_recon = x_recon[:, :orig_C, :orig_T]
        x_recon_norm = x_recon_norm[:, :orig_C, :orig_T]
        
        return {
            "x_recon": x_recon,           # Raw scale (for downstream tasks)
            "x_recon_norm": x_recon_norm,  # Normalized scale (for loss computation)
        }


class TemporalBlock(nn.Module):
    """TCN residual block with dilated causal convolution."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.padding = padding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        return self.relu(out + residual)


class TCNAutoencoder(BaseAutoencoder):
    """TCN-based autoencoder for e-nose time series.
    
    Lightweight TCN with dilated convolutions.
    Target: ~2M parameters
    """
    
    def __init__(
        self,
        d_model: int = 160,
        num_layers: int = 6,
        kernel_size: int = 3,
        max_length: int = 1024,
        max_channels: int = 16,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(max_channels=max_channels, **kwargs)
        self.save_hyperparameters()
        
        self.max_length = max_length
        self.max_channels = max_channels
        hidden_ch = d_model
        
        # Encoder: TCN blocks with increasing dilation
        # Target ~2M total params with d_model=128, num_layers=6
        encoder_blocks = []
        in_ch = max_channels
        
        for i in range(num_layers):
            dilation = 2 ** i
            encoder_blocks.append(
                TemporalBlock(in_ch, hidden_ch, kernel_size, dilation, dropout)
            )
            in_ch = hidden_ch
        
        self.encoder = nn.Sequential(*encoder_blocks)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(hidden_ch, hidden_ch, 1),
            nn.BatchNorm1d(hidden_ch),
            nn.ReLU(),
        )
        
        # Decoder: TCN blocks with decreasing dilation
        decoder_blocks = []
        in_ch = hidden_ch
        
        for i in range(num_layers - 1, -1, -1):
            dilation = 2 ** i
            out_ch = max_channels if i == 0 else hidden_ch
            decoder_blocks.append(
                TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )
            in_ch = out_ch
        
        self.decoder = nn.Sequential(*decoder_blocks)
        
        # Final projection
        self.output_proj = nn.Conv1d(max_channels, max_channels, 1)
    
    def forward(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
        sensor_indices: Optional[torch.Tensor] = None,
        sample_rate: Optional[torch.Tensor] = None,
        training_mask: Optional[torch.Tensor] = None,
        return_raw: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, C, T] input time series (RAW scale, not normalized)
            channel_mask: [B, C] padding mask (True = valid)
            sensor_indices: [B, C] sensor model indices (unused, for API compatibility)
            sample_rate: [B] sample rate in Hz (unused, for API compatibility)
            training_mask: [B, C] training mask (True = visible)
            return_raw: If True, denormalize output to raw scale; if False, return normalized
            
        Returns:
            Dict with 'x_recon' (raw scale) and optionally 'x_recon_norm' (normalized)
        """
        B, C, T = x.shape
        orig_T = T
        orig_C = C
        
        # Pad channels to max_channels BEFORE normalization
        if C < self.max_channels:
            x_padded = torch.zeros(B, self.max_channels, T, device=x.device, dtype=x.dtype)
            x_padded[:, :C, :] = x
            x = x_padded
        
        # Pad time to max_length
        if T < self.max_length:
            x_padded = torch.zeros(B, self.max_channels, self.max_length, device=x.device, dtype=x.dtype)
            x_padded[:, :, :T] = x
            x = x_padded
        elif T > self.max_length:
            x = x[:, :, :self.max_length]
            orig_T = self.max_length
        
        # Input normalization (internal BN)
        x = self.normalize_input(x)
        
        # Encode
        z = self.encoder(x)
        
        # Bottleneck
        z = self.bottleneck(z)
        
        # Decode
        x_recon_norm = self.decoder(z)
        
        # Output projection
        x_recon_norm = self.output_proj(x_recon_norm)
        
        # Output denormalization (inverse BN) to raw scale
        x_recon = self.denormalize_output(x_recon_norm)
        
        # Crop to original size
        x_recon = x_recon[:, :orig_C, :orig_T]
        x_recon_norm = x_recon_norm[:, :orig_C, :orig_T]
        
        return {
            "x_recon": x_recon,           # Raw scale (for downstream tasks)
            "x_recon_norm": x_recon_norm,  # Normalized scale (for loss computation)
        }


# Model registry for easy access
BASELINE_MODELS = {
    "mlp": MLPAutoencoder,
    "tcn": TCNAutoencoder,
}


def create_baseline_model(model_type: str, **kwargs) -> BaseAutoencoder:
    """Create a baseline model by type.
    
    Args:
        model_type: One of 'mlp', 'tcn'
        **kwargs: Model-specific arguments
        
    Returns:
        Initialized model
    """
    if model_type not in BASELINE_MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(BASELINE_MODELS.keys())}")
    
    return BASELINE_MODELS[model_type](**kwargs)
