"""Lightning DataModule for single-dataset classification."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import lightning as L
from torch.utils.data import DataLoader

from .dataset import SingleDatasetClassification, collate_fn, collate_fn_dual_view
from enose_uci_dataset.datasets.smellnet import (
    SMELLNET_PURE_MEAN,
    SMELLNET_PURE_STD,
    SMELLNET_MIXTURE_MEAN,
    SMELLNET_MIXTURE_STD,
)


class ClassificationDataModule(L.LightningDataModule):
    """DataModule for single-dataset gas classification.
    
    Args:
        root: Root directory for datasets
        dataset_name: Name of the dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        max_length: Maximum sequence length
        num_channels: Number of sensor channels
        download: Whether to download missing dataset
        train_ratio: Training data ratio
        val_ratio: Validation data ratio
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        root: Union[str, Path] = ".cache",
        dataset_name: str = "twin_gas_sensor_arrays",
        batch_size: int = 32,
        num_workers: int = 4,
        max_length: int = 512,
        num_channels: int = 8,
        download: bool = False,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
        trim_start: int = 0,
        trim_end: int = 0,
        frequency_domain: bool = False,
        fft_cutoff_hz: float = 0,
        subset: Optional[str] = None,
        window_size: int = 0,
        window_stride: int = 0,
        standardize: bool = True,  # Apply StandardScaler (fit on train)
        lag: int = 0,  # Lag for difference features (0=disabled)
        dual_view: bool = False,  # Dual-view fusion (time + freq)
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.num_channels = num_channels
        self.download = download
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.trim_start = trim_start
        self.trim_end = trim_end
        self.frequency_domain = frequency_domain
        self.fft_cutoff_hz = fft_cutoff_hz
        self.subset = subset
        self.window_size = window_size
        self.window_stride = window_stride
        self.standardize = standardize
        self.lag = lag
        self.dual_view = dual_view
        
        self.scaler_mean = None
        self.scaler_std = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage."""
        if stage == "fit" or stage is None:
            # Create train dataset first (without scaler)
            self.train_dataset = SingleDatasetClassification(
                root=self.root,
                dataset_name=self.dataset_name,
                split="train",
                max_length=self.max_length,
                num_channels=self.num_channels,
                download=self.download,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                seed=self.seed,
                trim_start=self.trim_start,
                trim_end=self.trim_end,
                frequency_domain=self.frequency_domain,
                fft_cutoff_hz=self.fft_cutoff_hz,
                subset=self.subset,
                window_size=self.window_size,
                window_stride=self.window_stride,
                lag=self.lag,
                dual_view=self.dual_view,
            )
            
            # Apply standardization using fixed constants (like ImageNet mean/std)
            if self.standardize:
                if self.dataset_name == "smellnet":
                    # Use pre-computed global constants from training set
                    if self.subset == "mixture" and SMELLNET_MIXTURE_MEAN is not None:
                        self.scaler_mean = np.array(SMELLNET_MIXTURE_MEAN, dtype=np.float32)
                        self.scaler_std = np.array(SMELLNET_MIXTURE_STD, dtype=np.float32)
                    else:  # pure (default)
                        self.scaler_mean = np.array(SMELLNET_PURE_MEAN, dtype=np.float32)
                        self.scaler_std = np.array(SMELLNET_PURE_STD, dtype=np.float32)
                    print(f"Using fixed SmellNet normalization constants (like ImageNet)")
                else:
                    # For other datasets, compute from training set
                    print("Computing StandardScaler stats from training data...")
                    self.scaler_mean, self.scaler_std = self.train_dataset.compute_scaler_stats()
                # Apply scaler to train dataset
                self.train_dataset.scaler_mean = self.scaler_mean
                self.train_dataset.scaler_std = self.scaler_std
            
            self.val_dataset = SingleDatasetClassification(
                root=self.root,
                dataset_name=self.dataset_name,
                split="val",
                max_length=self.max_length,
                num_channels=self.num_channels,
                download=False,  # Already downloaded in train
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                seed=self.seed,
                trim_start=self.trim_start,
                trim_end=self.trim_end,
                frequency_domain=self.frequency_domain,
                fft_cutoff_hz=self.fft_cutoff_hz,
                subset=self.subset,
                window_size=self.window_size,
                window_stride=self.window_stride,
                scaler_mean=self.scaler_mean,
                scaler_std=self.scaler_std,
                lag=self.lag,
                dual_view=self.dual_view,
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = SingleDatasetClassification(
                root=self.root,
                dataset_name=self.dataset_name,
                split="test",
                max_length=self.max_length,
                num_channels=self.num_channels,
                download=False,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                seed=self.seed,
                trim_start=self.trim_start,
                trim_end=self.trim_end,
                frequency_domain=self.frequency_domain,
                fft_cutoff_hz=self.fft_cutoff_hz,
                subset=self.subset,
                window_size=self.window_size,
                window_stride=self.window_stride,
                scaler_mean=self.scaler_mean,
                scaler_std=self.scaler_std,
                lag=self.lag,
                dual_view=self.dual_view,
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn_dual_view if self.dual_view else collate_fn,
            pin_memory=True,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn_dual_view if self.dual_view else collate_fn,
            pin_memory=True,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn_dual_view if self.dual_view else collate_fn,
            pin_memory=True,
        )
    
    @property
    def sample_rate(self) -> float:
        """Sample rate of the dataset in Hz."""
        if self.train_dataset is not None:
            return self.train_dataset.sample_rate
        return 100.0  # fallback
    
    @property
    def num_classes(self) -> int:
        """Number of classes in the dataset."""
        if self.train_dataset is not None:
            return self.train_dataset.num_classes
        # Create temporary dataset to get num_classes
        ds = SingleDatasetClassification(
            root=self.root,
            dataset_name=self.dataset_name,
            split="train",
            max_length=self.max_length,
            num_channels=self.num_channels,
            download=self.download,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            seed=self.seed,
        )
        return ds.num_classes
    
    def get_class_names(self) -> List[str]:
        """Get class names from the dataset."""
        if self.train_dataset is not None:
            return self.train_dataset.get_class_names()
        ds = SingleDatasetClassification(
            root=self.root,
            dataset_name=self.dataset_name,
            split="train",
            max_length=self.max_length,
            num_channels=self.num_channels,
            download=self.download,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            seed=self.seed,
        )
        return ds.get_class_names()
    
    def get_class_weights(self):
        """Get class weights for handling imbalanced data."""
        if self.train_dataset is not None:
            return self.train_dataset.get_class_weights()
        return None
