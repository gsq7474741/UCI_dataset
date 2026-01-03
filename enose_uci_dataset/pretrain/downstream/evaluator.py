"""Downstream task evaluator.

Compares downstream model performance with:
1. Original inputs (ground truth)
2. Reconstructed inputs (from backbone model)
3. Zeroed inputs (baseline)

This measures how well the backbone preserves task-relevant information.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import lightning as L
from torchmetrics.functional import r2_score

from .base import BaseDownstreamModel, BaseDownstreamTask


class DownstreamEvaluator:
    """Evaluator for comparing original vs reconstructed inputs on downstream tasks.
    
    Usage:
        evaluator = DownstreamEvaluator(
            backbone_model=vqvae,
            downstream_model=regressor,
            task=concentration_task,
        )
        results = evaluator.evaluate()
        # results = {
        #     "r2_original": 0.85,
        #     "r2_reconstructed": 0.82,
        #     "r2_zeroed": 0.10,
        #     "r2_degradation": 0.03,
        #     "r2_recovery_rate": 0.96,
        # }
    """
    
    def __init__(
        self,
        backbone_model: L.LightningModule,
        downstream_model: BaseDownstreamModel,
        task: BaseDownstreamTask,
        mask_channels: Optional[List[int]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize evaluator.
        
        Args:
            backbone_model: Pretrained backbone (VQ-VAE, MLP, TCN autoencoder)
            downstream_model: Trained downstream model
            task: Downstream task with test data
            mask_channels: Channels to mask (None = no masking, just reconstruction)
            device: Device to run evaluation on
        """
        self.backbone = backbone_model
        self.downstream = downstream_model
        self.task = task
        self.mask_channels = mask_channels or []
        self.device = device
        
        self.backbone.eval()
        self.downstream.eval()
        self.backbone.to(device)
        self.downstream.to(device)
    
    @torch.no_grad()
    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Evaluate downstream performance with different inputs.
        
        Args:
            dataloader: DataLoader to evaluate on (default: task's test dataloader)
            
        Returns:
            Dict with metrics for original, reconstructed, and zeroed inputs
        """
        if dataloader is None:
            dataloader = self.task.test_dataloader()
        
        all_pred_original = []
        all_pred_recon = []
        all_pred_zeroed = []
        all_targets = []
        
        for batch in dataloader:
            x = batch["data"].to(self.device)  # [B, C, T]
            target = batch["target"].to(self.device)
            
            B, C, T = x.shape
            
            # 1. Original input prediction
            pred_original = self.downstream(x)
            all_pred_original.append(pred_original.cpu())
            all_targets.append(target.cpu())
            
            # 2. Reconstructed input prediction
            x_recon = self._reconstruct(x)
            pred_recon = self.downstream(x_recon)
            all_pred_recon.append(pred_recon.cpu())
            
            # 3. Zeroed (masked) input prediction (baseline)
            x_zeroed = x.clone()
            if self.mask_channels:
                for ch in self.mask_channels:
                    if ch < C:
                        x_zeroed[:, ch, :] = 0
            pred_zeroed = self.downstream(x_zeroed)
            all_pred_zeroed.append(pred_zeroed.cpu())
        
        # Concatenate all predictions
        pred_original = torch.cat(all_pred_original, dim=0).flatten()
        pred_recon = torch.cat(all_pred_recon, dim=0).flatten()
        pred_zeroed = torch.cat(all_pred_zeroed, dim=0).flatten()
        targets = torch.cat(all_targets, dim=0).flatten()
        
        # Compute R² scores
        r2_original = r2_score(pred_original, targets).item()
        r2_recon = r2_score(pred_recon, targets).item()
        r2_zeroed = r2_score(pred_zeroed, targets).item()
        
        # Compute degradation and recovery metrics
        r2_degradation = r2_original - r2_recon
        
        # Recovery rate: how much of the original R² is preserved
        if r2_original > 0:
            r2_recovery_rate = r2_recon / r2_original
        else:
            r2_recovery_rate = 0.0
        
        # MSE comparison
        mse_original = F.mse_loss(pred_original, targets).item()
        mse_recon = F.mse_loss(pred_recon, targets).item()
        mse_zeroed = F.mse_loss(pred_zeroed, targets).item()
        
        return {
            "r2_original": r2_original,
            "r2_reconstructed": r2_recon,
            "r2_zeroed": r2_zeroed,
            "r2_degradation": r2_degradation,
            "r2_recovery_rate": r2_recovery_rate,
            "mse_original": mse_original,
            "mse_reconstructed": mse_recon,
            "mse_zeroed": mse_zeroed,
        }
    
    def _reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input using backbone model.
        
        Args:
            x: Original input [B, C, T]
            
        Returns:
            Reconstructed input [B, C, T]
        """
        B, C, T = x.shape
        
        # Apply masking if specified
        x_input = x.clone()
        if self.mask_channels:
            for ch in self.mask_channels:
                if ch < C:
                    x_input[:, ch, :] = 0
        
        # Create channel mask (all valid)
        channel_mask = torch.ones(B, C, dtype=torch.bool, device=x.device)
        
        # Create training mask (True = visible)
        training_mask = torch.ones(B, C, dtype=torch.bool, device=x.device)
        if self.mask_channels:
            for ch in self.mask_channels:
                if ch < C:
                    training_mask[:, ch] = False
        
        # Get model's max_length for padding
        max_length = getattr(self.backbone, 'max_length', T)
        if T < max_length:
            x_input = F.pad(x_input, (0, max_length - T), value=0)
        
        # Forward pass through backbone
        outputs = self.backbone(
            x_input,
            channel_mask=channel_mask,
            sensor_indices=None,
            sample_rate=None,
            training_mask=training_mask,
        )
        
        x_recon = outputs["x_recon"]
        
        # Crop to original size
        x_recon = x_recon[:, :C, :T]
        
        return x_recon
    
    @torch.no_grad()
    def evaluate_masked_imputation(
        self,
        dataloader: Optional[DataLoader] = None,
        mask_channels: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """Evaluate downstream performance specifically for masked channel imputation.
        
        This tests the backbone's ability to impute masked channels in a way
        that preserves downstream task performance.
        
        Args:
            dataloader: DataLoader to evaluate on
            mask_channels: Override default mask channels
            
        Returns:
            Detailed metrics for imputation quality
        """
        if dataloader is None:
            dataloader = self.task.test_dataloader()
        
        mask_ch = mask_channels if mask_channels is not None else self.mask_channels
        
        all_pred_full = []
        all_pred_imputed = []
        all_pred_zeroed = []
        all_targets = []
        
        # Also track reconstruction MSE on masked channels
        all_recon_mse = []
        
        for batch in dataloader:
            x = batch["data"].to(self.device)  # [B, C, T]
            target = batch["target"].to(self.device)
            
            B, C, T = x.shape
            
            # 1. Full input (no masking)
            pred_full = self.downstream(x)
            all_pred_full.append(pred_full.cpu())
            all_targets.append(target.cpu())
            
            # 2. Mask and impute
            x_masked = x.clone()
            for ch in mask_ch:
                if ch < C:
                    x_masked[:, ch, :] = 0
            
            x_imputed = self._reconstruct_with_mask(x_masked, mask_ch)
            pred_imputed = self.downstream(x_imputed)
            all_pred_imputed.append(pred_imputed.cpu())
            
            # Compute reconstruction MSE on masked channels only
            if mask_ch:
                mse_masked = 0
                for ch in mask_ch:
                    if ch < C:
                        mse_masked += F.mse_loss(x_imputed[:, ch, :], x[:, ch, :]).item()
                mse_masked /= len(mask_ch)
                all_recon_mse.append(mse_masked)
            
            # 3. Zeroed input (baseline)
            pred_zeroed = self.downstream(x_masked)
            all_pred_zeroed.append(pred_zeroed.cpu())
        
        pred_full = torch.cat(all_pred_full, dim=0).flatten()
        pred_imputed = torch.cat(all_pred_imputed, dim=0).flatten()
        pred_zeroed = torch.cat(all_pred_zeroed, dim=0).flatten()
        targets = torch.cat(all_targets, dim=0).flatten()
        
        r2_full = r2_score(pred_full, targets).item()
        r2_imputed = r2_score(pred_imputed, targets).item()
        r2_zeroed = r2_score(pred_zeroed, targets).item()
        
        avg_recon_mse = np.mean(all_recon_mse) if all_recon_mse else 0.0
        
        return {
            "r2_full_input": r2_full,
            "r2_imputed": r2_imputed,
            "r2_zeroed": r2_zeroed,
            "r2_recovery_vs_full": r2_imputed / r2_full if r2_full > 0 else 0,
            "r2_gain_over_zeroed": r2_imputed - r2_zeroed,
            "masked_channel_recon_mse": avg_recon_mse,
            "mask_channels": mask_ch,
        }
    
    def _reconstruct_with_mask(
        self,
        x_masked: torch.Tensor,
        mask_channels: List[int],
    ) -> torch.Tensor:
        """Reconstruct with explicit mask channels."""
        B, C, T = x_masked.shape
        
        channel_mask = torch.ones(B, C, dtype=torch.bool, device=x_masked.device)
        training_mask = torch.ones(B, C, dtype=torch.bool, device=x_masked.device)
        for ch in mask_channels:
            if ch < C:
                training_mask[:, ch] = False
        
        max_length = getattr(self.backbone, 'max_length', T)
        if T < max_length:
            x_padded = F.pad(x_masked, (0, max_length - T), value=0)
        else:
            x_padded = x_masked
        
        outputs = self.backbone(
            x_padded,
            channel_mask=channel_mask,
            sensor_indices=None,
            sample_rate=None,
            training_mask=training_mask,
        )
        
        x_recon = outputs["x_recon"][:, :C, :T]
        return x_recon
