"""Base classes for downstream tasks and models.

Design principles:
1. Downstream models receive RAW sensor data (not normalized features)
2. Models handle their own normalization (BatchNorm, LayerNorm, etc.)
3. Tasks define the data loading, loss, and metrics
4. Evaluator compares performance with original vs reconstructed inputs
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import lightning as L
from torch.utils.data import DataLoader, Dataset


class BaseDownstreamModel(L.LightningModule, ABC):
    """Base class for downstream task models.
    
    Downstream models:
    - Accept raw sensor data [B, C, T] as input
    - Handle their own normalization (BN, LN, etc.)
    - Output task-specific predictions
    """
    
    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        learning_rate: float = 1e-3,
        optimizer: str = "adamw",  # "adamw" or "soap"
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Raw sensor data [B, C, T]
            
        Returns:
            Task-specific output (e.g., concentration prediction)
        """
        pass
    
    @abstractmethod
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute task-specific loss."""
        pass
    
    @abstractmethod
    def compute_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute task-specific metrics."""
        pass
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, target = batch["data"], batch["target"]
        pred = self(x)
        loss = self.compute_loss(pred, target)
        
        self.log("train/loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        x, target = batch["data"], batch["target"]
        pred = self(x)
        loss = self.compute_loss(pred, target)
        metrics = self.compute_metrics(pred, target)
        
        self.log("val/loss", loss, prog_bar=True)
        for k, v in metrics.items():
            self.log(f"val/{k}", v, prog_bar=True)
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        x, target = batch["data"], batch["target"]
        pred = self(x)
        loss = self.compute_loss(pred, target)
        metrics = self.compute_metrics(pred, target)
        
        self.log("test/loss", loss)
        for k, v in metrics.items():
            self.log(f"test/{k}", v)
    
    def configure_optimizers(self):
        if self.optimizer_type == "soap":
            from ..optimizers.soap import SOAP
            optimizer = SOAP(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class BaseDownstreamTask(ABC):
    """Base class for downstream tasks.
    
    Tasks define:
    - Dataset loading and preprocessing
    - Train/val/test splits
    - Target extraction
    """
    
    name: str = "base_task"
    
    def __init__(
        self,
        data_root: Union[str, Path] = ".cache",
        seq_len: int = 1000,
        batch_size: int = 32,
        num_workers: int = 4,
        **kwargs,
    ):
        self.data_root = Path(data_root)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None
    
    @abstractmethod
    def setup(self) -> None:
        """Setup datasets."""
        pass
    
    @abstractmethod
    def get_num_channels(self) -> int:
        """Return number of input channels."""
        pass
    
    @abstractmethod
    def get_target_dim(self) -> int:
        """Return target dimension."""
        pass
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
