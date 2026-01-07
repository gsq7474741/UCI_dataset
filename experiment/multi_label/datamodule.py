"""Lightning DataModule for multi-label gas classification."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader, random_split
import lightning as L

from .dataset import MultiLabelGasDataset, collate_fn


class MultiLabelDataModule(L.LightningDataModule):
    """DataModule for multi-label gas classification.
    
    Training: Pure gas samples (single-label)
    Validation: Split from pure gas samples
    Testing: Mixture samples (multi-label)
    """
    
    def __init__(
        self,
        root: Union[str, Path] = ".cache",
        train_sources: Optional[List[str]] = None,
        test_sources: Optional[List[str]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        max_length: int = 512,
        num_channels: int = 6,
        val_split: float = 0.15,
        download: bool = False,
        seed: int = 42,
        smellnet_root: Optional[str] = None,
    ):
        """
        Args:
            root: Root directory for datasets
            train_sources: Sources for training (pure gases)
            test_sources: Sources for testing (mixtures)
            batch_size: Batch size
            num_workers: DataLoader workers
            max_length: Maximum sequence length
            num_channels: Number of sensor channels
            val_split: Fraction for validation
            download: Download missing datasets
            seed: Random seed
            smellnet_root: Optional path to local SmellNet data
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.root = Path(root)
        self.train_sources = train_sources or ["twin_gas_pure"]
        self.test_sources = test_sources or ["gas_sensor_turbulent", "gas_sensor_dynamic"]
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def prepare_data(self) -> None:
        """Download datasets if needed."""
        if self.hparams.download:
            # Just instantiate to trigger download
            _ = MultiLabelGasDataset(
                root=self.root,
                mode="train",
                sources=self.train_sources,
                max_length=self.hparams.max_length,
                num_channels=self.hparams.num_channels,
                download=True,
                smellnet_root=self.hparams.smellnet_root,
            )
            _ = MultiLabelGasDataset(
                root=self.root,
                mode="test",
                sources=self.test_sources,
                max_length=self.hparams.max_length,
                num_channels=self.hparams.num_channels,
                download=True,
            )
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Set up train/val/test datasets."""
        if stage == "fit" or stage is None:
            full_train = MultiLabelGasDataset(
                root=self.root,
                mode="train",
                sources=self.train_sources,
                max_length=self.hparams.max_length,
                num_channels=self.hparams.num_channels,
                download=False,
                smellnet_root=self.hparams.smellnet_root,
            )
            
            # Split into train/val
            n_samples = len(full_train)
            n_val = int(n_samples * self.hparams.val_split)
            n_train = n_samples - n_val
            
            generator = torch.Generator().manual_seed(self.hparams.seed)
            self.train_dataset, self.val_dataset = random_split(
                full_train, [n_train, n_val], generator=generator
            )
            
            print(f"Training: {n_train} samples, Validation: {n_val} samples")
            print(f"Number of classes: {full_train.num_classes}")
        
        if stage == "test" or stage is None:
            self.test_dataset = MultiLabelGasDataset(
                root=self.root,
                mode="test",
                sources=self.test_sources,
                max_length=self.hparams.max_length,
                num_channels=self.hparams.num_channels,
                download=False,
            )
            print(f"Test: {len(self.test_dataset)} samples")
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    
    @property
    def num_classes(self) -> int:
        """Get number of classes from train dataset."""
        if self.train_dataset is not None:
            # Handle Subset wrapper
            ds = self.train_dataset
            while hasattr(ds, "dataset"):
                ds = ds.dataset
            return ds.num_classes
        # Default
        from .dataset import GasLabelEncoder
        return GasLabelEncoder().num_classes
