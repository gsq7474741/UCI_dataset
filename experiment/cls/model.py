"""1D CNN classification model for e-nose gas classification.

Architecture: 1D CNN with residual connections and global pooling.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassConfusionMatrix,
)


class Conv1dBlock(nn.Module):
    """1D Convolutional block with BatchNorm and residual connection."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.residual = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.dropout(x)
        
        return x + residual


class CNN1DEncoder(nn.Module):
    """1D CNN encoder for time series classification.
    
    Architecture:
    - Input projection
    - Stack of Conv1d blocks with increasing dilation
    - Global average pooling
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        
        # Convolutional layers
        layers = []
        for i in range(num_layers):
            # Optionally use strided conv for downsampling
            stride = 2 if i % 2 == 1 and i > 0 else 1
            layers.append(Conv1dBlock(
                hidden_dim, hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                dropout=dropout,
            ))
        self.layers = nn.Sequential(*layers)
        
        self.output_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] input tensor
        Returns:
            [B, hidden_dim] pooled representation
        """
        x = self.input_proj(x)
        x = self.layers(x)
        # Global average pooling
        x = x.mean(dim=-1)
        return x


class TCNBlock(nn.Module):
    """Temporal Convolutional Network block with dilated causal convolution.
    
    Features:
    - Dilated convolution for exponentially growing receptive field
    - Causal padding (no future information leakage)
    - Residual connection
    - Weight normalization for stable training
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Causal padding: (kernel_size - 1) * dilation on left side only
        self.padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        )
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] input tensor
        Returns:
            [B, out_channels, T] output tensor (same length)
        """
        residual = self.residual(x)
        
        # First conv with causal padding
        out = F.pad(x, (self.padding, 0))  # Pad left only
        out = self.conv1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        # Second conv with causal padding
        out = F.pad(out, (self.padding, 0))
        out = self.conv2(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        return out + residual


class TCNEncoder(nn.Module):
    """Temporal Convolutional Network encoder for time series.
    
    Architecture:
    - Input projection
    - Stack of TCN blocks with exponentially increasing dilation
    - Global average pooling
    
    Receptive field grows exponentially: 2^num_layers * kernel_size
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        
        # TCN blocks with exponentially increasing dilation
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i  # 1, 2, 4, 8, ...
            self.blocks.append(TCNBlock(
                hidden_dim, hidden_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            ))
        
        self.output_dim = hidden_dim
        
        # Calculate receptive field
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2 ** num_layers - 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] input tensor
        Returns:
            [B, hidden_dim] pooled representation
        """
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        # Global average pooling
        x = x.mean(dim=-1)
        return x
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return features before pooling for Grad-CAM.
        
        Args:
            x: [B, C, T] input tensor
        Returns:
            [B, hidden_dim, T] feature maps
        """
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention.
    
    Adaptively recalibrates channel-wise feature responses.
    Does NOT mix temporal features, only scales channels.
    """
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),  # Changed from ReLU to GELU to avoid dead neurons/gradient truncation
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MultiScaleGroupedBlock(nn.Module):
    """Multi-scale grouped convolution block.
    
    Uses multiple kernel sizes to capture different temporal patterns
    while keeping channels separate (grouped conv).
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = [3, 5, 7],
        stride: int = 1,
        dropout: float = 0.1,
        groups: int = 1,
    ):
        super().__init__()
        self.branches = nn.ModuleList()
        
        # Ensure out_channels is divisible by number of branches
        assert out_channels % len(kernel_sizes) == 0
        branch_channels = out_channels // len(kernel_sizes)
        
        # Each branch uses a different kernel size
        for k in kernel_sizes:
            padding = k // 2
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_channels, branch_channels, k, stride, padding, groups=groups),
                nn.BatchNorm1d(branch_channels),
                nn.GELU(),
                nn.Dropout(dropout)
            ))
            
        # 1x1 conv to fuse branches (keeping groups to preserve channel independence)
        self.fuse = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, groups=groups),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply each branch
        outs = [branch(x) for branch in self.branches]
        # Concatenate along channel dimension
        out = torch.cat(outs, dim=1)
        # Fuse
        return self.fuse(out)


class ChannelWiseCNN1DEncoder(nn.Module):
    """Channel-wise 1D CNN encoder with Multi-scale + SE.
    
    Key design for interpretability AND performance:
    1. Grouped convolutions: Preserve channel independence deep into the network.
    2. Multi-scale kernels: Capture short & long-term patterns per channel.
    3. SE Block: Allow channel interaction via importance scaling (not mixing waveforms).
    
    Architecture:
    - Per-channel feature extraction (grouped conv)
    - Stack of MultiScaleGroupedBlock + SEBlock
    - Final aggregation with learned channel weights
    """
    
    def __init__(
        self,
        in_channels: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 4,
        kernel_size: int = 7,  # Base kernel size, will use [k-2, k, k+2]
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # Per-channel feature dimension
        self.per_channel_dim = hidden_dim // in_channels
        assert self.per_channel_dim * in_channels == hidden_dim, \
            f"hidden_dim ({hidden_dim}) must be divisible by in_channels ({in_channels})"
        
        # Multi-scale kernels
        k_mid = kernel_size
        k_small = max(3, k_mid - 2)
        k_large = k_mid + 2
        kernel_sizes = [k_small, k_mid, k_large]
        
        # Per-channel input projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1, groups=in_channels),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        
        # Encoder layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            stride = 2 if i % 2 == 1 and i > 0 else 1
            
            # 1. Multi-scale grouped convolution
            # Constraint: branch_channels must be divisible by groups (in_channels)
            # branch_channels = hidden_dim / num_scales
            # So (hidden_dim / num_scales) % in_channels == 0
            # This implies (hidden_dim / in_channels) % num_scales == 0
            # i.e., per_channel_dim % num_scales == 0
            
            if self.per_channel_dim % 3 == 0:
                ks = kernel_sizes  # [k-2, k, k+2]
            elif self.per_channel_dim % 2 == 0:
                ks = [kernel_sizes[0], kernel_sizes[-1]]  # [k-2, k+2]
            else:
                ks = [kernel_size] # Fallback to single scale
            
            block = nn.Sequential(
                MultiScaleGroupedBlock(
                    hidden_dim, hidden_dim, 
                    kernel_sizes=ks, 
                    stride=stride, 
                    dropout=dropout, 
                    groups=in_channels # Crucial: Keep channels separate!
                ),
                # 2. SE Block for channel interaction (re-weighting)
                SEBlock(hidden_dim, reduction=4)
            )
            self.layers.append(block)
        
        # Final channel attention (for aggregation)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, in_channels),
            nn.Sigmoid(),
        )
        
        self.output_dim = hidden_dim
        
        # Store intermediate features for CAM
        self._feature_maps = None
        self._channel_weights = None
    
    def forward(self, x: torch.Tensor, return_cam_features: bool = False) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] input tensor (C = in_channels = 8)
            return_cam_features: If True, also return features for CAM
        Returns:
            [B, hidden_dim] pooled representation
            (optional) per-channel feature maps for CAM
        """
        B, C, T = x.shape
        
        # Per-channel projection
        x = self.input_proj(x)  # [B, hidden_dim, T]
        
        # Apply layers
        for layer in self.layers:
            x = layer(x)
        
        # Store feature maps before pooling (for CAM)
        self._feature_maps = x.detach()  # [B, hidden_dim, T']
        
        # Compute channel attention weights
        channel_weights = self.channel_attention(x)  # [B, in_channels]
        self._channel_weights = channel_weights.detach()
        
        # Reshape to [B, in_channels, per_channel_dim, T']
        T_out = x.shape[-1]
        x_reshaped = x.view(B, self.in_channels, self.per_channel_dim, T_out)
        
        # Apply channel attention (Aggregation)
        # weight: [B, C] -> [B, C, 1, 1]
        x_weighted = x_reshaped * channel_weights.unsqueeze(-1).unsqueeze(-1)
        
        # Flatten back and global pool
        x = x_weighted.view(B, self.hidden_dim, T_out)
        pooled = x.mean(dim=-1)  # [B, hidden_dim]
        
        if return_cam_features:
            return pooled, x, channel_weights
        return pooled
    
    def get_channel_cam(self, x: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
        """Compute per-channel CAM (Class Activation Map)."""
        B, C, T = x.shape
        
        # Forward to get features
        _, features, channel_attn = self.forward(x, return_cam_features=True)
        # features: [B, hidden_dim, T']
        
        T_out = features.shape[-1]
        
        # Reshape features to per-channel: [B, in_channels, per_channel_dim, T']
        features_per_ch = features.view(B, self.in_channels, self.per_channel_dim, T_out)
        
        # Compute CAM per channel (weighted sum across feature dim)
        # Use channel attention as importance weight
        cam_per_channel = features_per_ch.mean(dim=2)  # [B, in_channels, T']
        
        # Weight by channel attention
        cam_weighted = cam_per_channel * channel_attn.unsqueeze(-1)  # [B, in_channels, T']
        
        return cam_weighted


class CNN1DClassifier(L.LightningModule):
    """1D CNN classifier for e-nose gas classification.
    
    Uses CrossEntropyLoss for multi-class classification.
    Supports class weights for imbalanced data.
    """
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.1,
        learning_rate: float = 3e-3,
        weight_decay: float = 1e-4,
        class_weights: Optional[List[float]] = None,
        label_smoothing: float = 0.0,
        class_names: Optional[List[str]] = None,
        channel_wise: bool = False,  # Use channel-wise encoder for interpretability
        encoder_type: str = "cnn",  # "cnn" or "tcn"
        mixup_alpha: float = 0.0,  # Mixup alpha (0 = disabled)
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['class_names'])
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.channel_wise = channel_wise
        self.encoder_type = encoder_type
        self.mixup_alpha = mixup_alpha
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
        # Encoder - choose based on encoder_type and channel_wise flag
        if channel_wise:
            self.encoder = ChannelWiseCNN1DEncoder(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                kernel_size=kernel_size,
                dropout=dropout,
            )
        elif encoder_type == "tcn":
            self.encoder = TCNEncoder(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                kernel_size=kernel_size,
                dropout=dropout,
            )
        else:  # default: cnn
            self.encoder = CNN1DEncoder(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                kernel_size=kernel_size,
                dropout=dropout,
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
        # Loss
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights))
        else:
            self.class_weights = None
        self.label_smoothing = label_smoothing
        
        # Metrics
        metrics = MetricCollection({
            "acc": MulticlassAccuracy(num_classes=num_classes),
            "f1_macro": MulticlassF1Score(num_classes=num_classes, average="macro"),
            "f1_micro": MulticlassF1Score(num_classes=num_classes, average="micro"),
            "precision": MulticlassPrecision(num_classes=num_classes, average="macro"),
            "recall": MulticlassRecall(num_classes=num_classes, average="macro"),
        })
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")
        
        # Confusion matrix for test
        self.test_confusion = MulticlassConfusionMatrix(num_classes=num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] input tensor
        Returns:
            [B, num_classes] logits
        """
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits
    
    def get_channel_cam(self, x: torch.Tensor, target_class: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-channel Class Activation Map using Grad-CAM.
        
        Args:
            x: [B, C, T] input tensor
            target_class: Target class for CAM (None = predicted class)
            
        Returns:
            cam: [B, in_channels, T'] per-channel CAM
            channel_weights: [B, in_channels] channel importance
            preds: [B] predicted classes
        """
        B, C, T = x.shape
        x.requires_grad_(True)
        
        # Forward pass
        if self.channel_wise:
            pooled, features, channel_weights = self.encoder(x, return_cam_features=True)
            # We need gradients for features to compute Grad-CAM
            features.retain_grad()
        else:
            # For non-channel-wise, compute gradient-based CAM
            features = self.encoder.input_proj(x)
            for layer in self.encoder.layers:
                features = layer(features)
            # Need gradients for features
            features.retain_grad()
            
            pooled = features.mean(dim=-1)
            channel_weights = torch.ones(B, self.in_channels, device=x.device) / self.in_channels
        
        logits = self.classifier(pooled)
        preds = logits.argmax(dim=-1)
        
        # Use predicted class if target not specified
        if target_class is None:
            target = preds
        else:
            target = torch.full((B,), target_class, device=x.device, dtype=torch.long)
        
        # Compute gradients w.r.t. target class
        one_hot = F.one_hot(target, self.num_classes).float()
        score = (logits * one_hot).sum()
        
        # Backward to get gradients
        self.zero_grad()
        score.backward(retain_graph=True)
        
        # Compute per-channel CAM
        T_out = features.shape[-1]
        
        if self.channel_wise:
            # Grouped Grad-CAM
            # features: [B, hidden_dim, T']
            # grads: [B, hidden_dim, T']
            grads = features.grad
            
            # Reshape to [B, InChannels, PerChDim, T']
            per_ch_dim = self.encoder.per_channel_dim
            features_per_ch = features.view(B, self.in_channels, per_ch_dim, T_out)
            grads_per_ch = grads.view(B, self.in_channels, per_ch_dim, T_out)
            
            # Compute weights: GAP over time [B, InChannels, PerChDim, 1]
            weights = grads_per_ch.mean(dim=3, keepdim=True)
            
            # Weighted combination: Sum over sub-features (PerChDim)
            # [B, InChannels, T']
            cam = (weights * features_per_ch).sum(dim=2)
            
            # Optionally modulate by SE attention weights (Channel Attention)
            # This combines Gradient importance (Class-specific) with SE importance (Data-specific)
            cam = cam * channel_weights.unsqueeze(-1)
            
        else:
            # Standard Grad-CAM (Global)
            grads = features.grad
            weights = grads.mean(dim=2, keepdim=True) # [B, HiddenDim, 1]
            
            # Global CAM: [B, T']
            global_cam = (weights * features).sum(dim=1)
            
            # For visualization, we simply expand to all channels 
            # (since we don't have channel-specific mapping in standard mode)
            cam = global_cam.unsqueeze(1).expand(-1, self.in_channels, -1) # [B, C, T']
            
            # Use Input Gradients for channel importance
            if x.grad is not None:
                input_importance = x.grad.abs().mean(dim=-1)
                input_importance = input_importance / (input_importance.sum(dim=-1, keepdim=True) + 1e-8)
                cam = cam * input_importance.unsqueeze(-1)
                channel_weights = input_importance
        
        # Normalize CAM
        cam = F.relu(cam)
        cam_max = cam.amax(dim=-1, keepdim=True).clamp(min=1e-8)
        cam = cam / cam_max
        
        return cam.detach(), channel_weights.detach(), preds.detach()
    
    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss with optional class weights and label smoothing."""
        return F.cross_entropy(
            logits, labels,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
    
    def _mixup_data(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply Mixup augmentation.
        
        Returns:
            mixed_x: Mixed input
            y_a, y_b: Original labels for the two mixed samples
            lam: Mixing coefficient
        """
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def _mixup_criterion(self, logits: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
        """Compute mixup loss as weighted combination of two CE losses."""
        loss_a = F.cross_entropy(logits, y_a, weight=self.class_weights, label_smoothing=self.label_smoothing)
        loss_b = F.cross_entropy(logits, y_b, weight=self.class_weights, label_smoothing=self.label_smoothing)
        return lam * loss_a + (1 - lam) * loss_b
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch["data"]
        labels = batch["label"]
        
        # Apply Mixup if enabled
        if self.mixup_alpha > 0 and self.training:
            x, labels_a, labels_b, lam = self._mixup_data(x, labels)
            logits = self(x)
            loss = self._mixup_criterion(logits, labels_a, labels_b, lam)
            # For metrics, use original labels (before mixing)
            preds = logits.argmax(dim=-1)
            # Use the dominant label for metrics
            effective_labels = labels_a if lam > 0.5 else labels_b
            self.train_metrics.update(preds, effective_labels)
        else:
            logits = self(x)
            loss = self._compute_loss(logits, labels)
            preds = logits.argmax(dim=-1)
            self.train_metrics.update(preds, labels)
        
        self.log("train/loss", loss, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        x = batch["data"]
        labels = batch["label"]
        
        logits = self(x)
        loss = self._compute_loss(logits, labels)
        
        preds = logits.argmax(dim=-1)
        self.val_metrics.update(preds, labels)
        
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
    
    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        self.log_dict(metrics)
        self.val_metrics.reset()
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        x = batch["data"]
        labels = batch["label"]
        
        logits = self(x)
        loss = self._compute_loss(logits, labels)
        
        preds = logits.argmax(dim=-1)
        self.test_metrics.update(preds, labels)
        self.test_confusion.update(preds, labels)
        
        self.log("test/loss", loss, sync_dist=True)
    
    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        self.log_dict(metrics)
        self.test_metrics.reset()
        
        # Log confusion matrix
        confusion = self.test_confusion.compute()
        self.test_confusion.reset()
        
        # # Print confusion matrix
        # print("\n" + "=" * 50)
        # print("Confusion Matrix:")
        # print("=" * 50)
        # print(f"Classes: {self.class_names}")
        # print(confusion.cpu().numpy())
        # print("=" * 50)
    
    def configure_optimizers(self):
        from enose_uci_dataset.pretrain.optimizers.soap import SOAP
        
        optimizer = SOAP(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 100,
            eta_min=1e-6,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable 1D convolution for efficiency."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class LightweightCNN1D(nn.Module):
    """Lightweight 1D CNN using depthwise separable convolutions."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 32,
        num_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_proj = nn.Conv1d(in_channels, hidden_dim, 1)
        
        layers = []
        for _ in range(num_layers):
            layers.extend([
                DepthwiseSeparableConv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.layers = nn.Sequential(*layers)
        
        self.output_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.layers(x)
        x = x.mean(dim=-1)
        return x


class DualViewClassifier(L.LightningModule):
    """Dual-view classifier fusing time-domain and frequency-domain features.
    
    Two parallel encoders process time-domain (with optional lag) and 
    frequency-domain (FFT) views, then fuse features for classification.
    """
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.1,
        learning_rate: float = 3e-3,
        weight_decay: float = 1e-4,
        class_weights: Optional[List[float]] = None,
        label_smoothing: float = 0.0,
        class_names: Optional[List[str]] = None,
        encoder_type: str = "tcn",
        mixup_alpha: float = 0.0,
        fusion: str = "concat",  # "concat", "add", "attention"
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['class_names'])
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.encoder_type = encoder_type
        self.mixup_alpha = mixup_alpha
        self.fusion = fusion
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
        # Time-domain encoder
        if encoder_type == "tcn":
            self.time_encoder = TCNEncoder(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                kernel_size=kernel_size,
                dropout=dropout,
            )
        else:
            self.time_encoder = CNN1DEncoder(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                kernel_size=kernel_size,
                dropout=dropout,
            )
        
        # Frequency-domain encoder (same architecture)
        if encoder_type == "tcn":
            self.freq_encoder = TCNEncoder(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                kernel_size=kernel_size,
                dropout=dropout,
            )
        else:
            self.freq_encoder = CNN1DEncoder(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                kernel_size=kernel_size,
                dropout=dropout,
            )
        
        # Fusion and classifier
        if fusion == "concat":
            classifier_input_dim = hidden_dim * 2
        elif fusion == "attention":
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 2),
                nn.Softmax(dim=-1),
            )
            classifier_input_dim = hidden_dim
        else:  # add
            classifier_input_dim = hidden_dim
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, num_classes),
        )
        
        # Loss
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights))
        else:
            self.class_weights = None
        self.label_smoothing = label_smoothing
        
        # Metrics
        metrics = MetricCollection({
            "acc": MulticlassAccuracy(num_classes=num_classes),
            "f1_macro": MulticlassF1Score(num_classes=num_classes, average="macro"),
            "f1_micro": MulticlassF1Score(num_classes=num_classes, average="micro"),
            "precision": MulticlassPrecision(num_classes=num_classes, average="macro"),
            "recall": MulticlassRecall(num_classes=num_classes, average="macro"),
        })
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")
        self.test_confusion = MulticlassConfusionMatrix(num_classes=num_classes)
    
    def forward(self, x_time: torch.Tensor, x_freq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_time: [B, C, T] time-domain input (with lag features)
            x_freq: [B, C, F] frequency-domain input (FFT magnitude)
        Returns:
            [B, num_classes] logits
        """
        # Encode both views
        feat_time = self.time_encoder(x_time)  # [B, hidden_dim]
        feat_freq = self.freq_encoder(x_freq)  # [B, hidden_dim]
        
        # Fuse features
        if self.fusion == "concat":
            features = torch.cat([feat_time, feat_freq], dim=-1)
        elif self.fusion == "attention":
            combined = torch.cat([feat_time, feat_freq], dim=-1)
            weights = self.attention(combined)  # [B, 2]
            features = weights[:, 0:1] * feat_time + weights[:, 1:2] * feat_freq
        else:  # add
            features = feat_time + feat_freq
        
        logits = self.classifier(features)
        return logits
    
    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits, labels,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
    
    def _mixup_data(self, x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor):
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = x1.size(0)
        index = torch.randperm(batch_size, device=x1.device)
        
        mixed_x1 = lam * x1 + (1 - lam) * x1[index]
        mixed_x2 = lam * x2 + (1 - lam) * x2[index]
        y_a, y_b = y, y[index]
        return mixed_x1, mixed_x2, y_a, y_b, lam
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x_time = batch["data_time"]
        x_freq = batch["data_freq"]
        labels = batch["label"]
        
        if self.mixup_alpha > 0 and self.training:
            x_time, x_freq, labels_a, labels_b, lam = self._mixup_data(x_time, x_freq, labels)
            logits = self(x_time, x_freq)
            loss_a = F.cross_entropy(logits, labels_a, weight=self.class_weights, label_smoothing=self.label_smoothing)
            loss_b = F.cross_entropy(logits, labels_b, weight=self.class_weights, label_smoothing=self.label_smoothing)
            loss = lam * loss_a + (1 - lam) * loss_b
            preds = logits.argmax(dim=-1)
            effective_labels = labels_a if lam > 0.5 else labels_b
            self.train_metrics.update(preds, effective_labels)
        else:
            logits = self(x_time, x_freq)
            loss = self._compute_loss(logits, labels)
            preds = logits.argmax(dim=-1)
            self.train_metrics.update(preds, labels)
        
        self.log("train/loss", loss, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        x_time = batch["data_time"]
        x_freq = batch["data_freq"]
        labels = batch["label"]
        
        logits = self(x_time, x_freq)
        loss = self._compute_loss(logits, labels)
        
        preds = logits.argmax(dim=-1)
        self.val_metrics.update(preds, labels)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_metrics["acc"], prog_bar=True)
    
    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        x_time = batch["data_time"]
        x_freq = batch["data_freq"]
        labels = batch["label"]
        
        logits = self(x_time, x_freq)
        loss = self._compute_loss(logits, labels)
        
        preds = logits.argmax(dim=-1)
        self.test_metrics.update(preds, labels)
        self.test_confusion.update(preds, labels)
        self.log("test/loss", loss)
    
    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()
    
    def configure_optimizers(self):
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
