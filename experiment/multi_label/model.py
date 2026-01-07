"""Multi-label classification model for gas mixture identification.

Architecture options:
1. TCN (Temporal Convolutional Network) - good for time series
2. MLP - simple baseline
3. Transformer - attention-based
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelAUROC,
)


class TCNBlock(nn.Module):
    """Temporal Convolutional Block with residual connection."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.gelu(x)
        x = self.dropout(x)
        
        return x + residual


class TCNEncoder(nn.Module):
    """TCN-based encoder for time series."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_proj = nn.Conv1d(in_channels, hidden_dim, 1)
        
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(TCNBlock(
                hidden_dim, hidden_dim,
                kernel_size=kernel_size,
                dilation=dilation,
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
        # Global average pooling over time
        x = x.mean(dim=-1)
        return x


class MLPEncoder(nn.Module):
    """Simple MLP encoder (flattens input)."""
    
    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        input_dim = in_channels * seq_len
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        self.layers = nn.Sequential(*layers)
        self.output_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] input tensor
        Returns:
            [B, hidden_dim] representation
        """
        x = x.flatten(1)  # [B, C*T]
        return self.layers(x)


class MultiLabelClassifier(L.LightningModule):
    """Multi-label classifier for gas mixture identification.
    
    Trained on pure gases (single-label), tested on mixtures (multi-label).
    Uses BCE loss with label smoothing for better generalization.
    """
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 6,
        seq_len: int = 512,
        hidden_dim: int = 128,
        num_layers: int = 4,
        encoder_type: str = "tcn",
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.1,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Encoder
        if encoder_type == "tcn":
            self.encoder = TCNEncoder(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif encoder_type == "mlp":
            self.encoder = MLPEncoder(
                in_channels=in_channels,
                seq_len=seq_len,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
        # Loss
        self.label_smoothing = label_smoothing
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor(pos_weight))
        else:
            self.pos_weight = None
        
        # Metrics
        metrics = MetricCollection({
            "accuracy": MultilabelAccuracy(num_labels=num_classes, average="micro"),
            "f1_micro": MultilabelF1Score(num_labels=num_classes, average="micro"),
            "f1_macro": MultilabelF1Score(num_labels=num_classes, average="macro"),
            "precision": MultilabelPrecision(num_labels=num_classes, average="micro"),
            "recall": MultilabelRecall(num_labels=num_classes, average="micro"),
        })
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")
        
        # Additional metrics for test (mixtures)
        self.test_auroc = MultilabelAUROC(num_labels=num_classes, average="macro")
    
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
    
    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute BCE loss with optional label smoothing."""
        if self.label_smoothing > 0:
            # Apply label smoothing
            labels = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        if self.pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=self.pos_weight
            )
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        return loss
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch["data"]
        labels = batch["label"]
        
        logits = self(x)
        loss = self._compute_loss(logits, labels)
        
        # Metrics
        preds = torch.sigmoid(logits)
        self.train_metrics.update(preds, labels.int())
        
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
        
        preds = torch.sigmoid(logits)
        self.val_metrics.update(preds, labels.int())
        
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
    
    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        self.log_dict(metrics)
        self.val_metrics.reset()
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        x = batch["data"]
        labels = batch["label"]
        is_mixture = batch["is_mixture"]
        
        logits = self(x)
        loss = self._compute_loss(logits, labels)
        
        preds = torch.sigmoid(logits)
        self.test_metrics.update(preds, labels.int())
        self.test_auroc.update(preds, labels.int())
        
        self.log("test/loss", loss, sync_dist=True)
        
        # Log metrics separately for pure vs mixture samples
        if is_mixture.any():
            mixture_mask = is_mixture
            mixture_preds = preds[mixture_mask]
            mixture_labels = labels[mixture_mask]
            
            # Calculate mixture-specific accuracy
            mixture_correct = ((mixture_preds > 0.5) == mixture_labels.bool()).float().mean()
            self.log("test/mixture_accuracy", mixture_correct, sync_dist=True)
    
    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        metrics["test/auroc"] = self.test_auroc.compute()
        self.log_dict(metrics)
        self.test_metrics.reset()
        self.test_auroc.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50,
            T_mult=2,
            eta_min=1e-6,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
