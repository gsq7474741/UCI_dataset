"""Downstream model implementations.

All models:
- Accept raw sensor data [B, C, T]
- Handle their own normalization internally
- Output task-specific predictions
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import r2_score

from .base import BaseDownstreamModel


class MLPRegressor(BaseDownstreamModel):
    """MLP-based regressor for concentration prediction.
    
    Architecture:
    - BatchNorm on input
    - Per-channel feature extraction
    - Global average pooling
    - MLP head
    """
    
    def __init__(
        self,
        in_channels: int = 8,
        seq_len: int = 1000,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        **kwargs,
    ):
        super().__init__(in_channels, seq_len, learning_rate, **kwargs)
        self.save_hyperparameters()
        
        # Input normalization - per channel
        self.input_bn = nn.BatchNorm1d(in_channels)
        
        # Per-channel encoder
        self.channel_encoder = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # MLP head
        head_layers = []
        for i in range(num_layers - 1):
            head_layers.extend([
                nn.Linear(hidden_dim if i == 0 else hidden_dim // 2, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        head_layers.append(nn.Linear(hidden_dim // 2 if num_layers > 1 else hidden_dim, 1))
        self.head = nn.Sequential(*head_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw sensor data [B, C, T]
        Returns:
            Concentration prediction [B, 1]
        """
        # Input normalization
        x = self.input_bn(x)  # [B, C, T]
        
        # Per-channel encoding
        z = self.channel_encoder(x)  # [B, C, hidden_dim]
        
        # Global average over channels
        z = z.mean(dim=1)  # [B, hidden_dim]
        
        # Prediction
        return self.head(z)  # [B, 1]
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        return F.mse_loss(pred, target)
    
    def compute_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        r2 = r2_score(pred.flatten(), target.flatten())
        return {"r2": r2.item()}


class TCNRegressor(BaseDownstreamModel):
    """TCN-based regressor for concentration prediction.
    
    Architecture:
    - BatchNorm on input
    - TCN encoder with dilated convolutions
    - Global average pooling
    - MLP head
    """
    
    def __init__(
        self,
        in_channels: int = 8,
        seq_len: int = 1000,
        hidden_dim: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        **kwargs,
    ):
        super().__init__(in_channels, seq_len, learning_rate, **kwargs)
        self.save_hyperparameters()
        
        # Input normalization
        self.input_bn = nn.BatchNorm1d(in_channels)
        
        # TCN encoder
        tcn_layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else hidden_dim
            tcn_layers.append(
                TemporalBlock(in_ch, hidden_dim, kernel_size, dilation, dropout)
            )
        self.tcn = nn.Sequential(*tcn_layers)
        
        # MLP head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw sensor data [B, C, T]
        Returns:
            Concentration prediction [B, 1]
        """
        # Input normalization
        x = self.input_bn(x)  # [B, C, T]
        
        # TCN encoding
        z = self.tcn(x)  # [B, hidden_dim, T]
        
        # Global average pooling over time
        z = z.mean(dim=-1)  # [B, hidden_dim]
        
        # Prediction
        return self.head(z)  # [B, 1]
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        return F.mse_loss(pred, target)
    
    def compute_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        r2 = r2_score(pred.flatten(), target.flatten())
        return {"r2": r2.item()}


class TemporalBlock(nn.Module):
    """TCN temporal block with dilated causal convolution."""
    
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
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]  # Causal: trim to input length
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = out[:, :, :x.size(2)]
        out = self.bn2(out)
        out = self.dropout(out)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        return self.relu(out + residual)
