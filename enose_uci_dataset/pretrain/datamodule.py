"""Lightning DataModule for e-nose pretraining."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import lightning as L

from ..datasets import CombinedEnoseDataset, list_datasets
from ..datasets._info import get_dataset_info
from .model import SensorEmbedding


class PretrainingCollator:
    """Collator for pretraining batches.
    
    Handles variable length sequences by padding/truncating to fixed length.
    """
    
    def __init__(
        self,
        max_length: int = 4096,
        pad_value: float = 0.0,
        max_channels: int = 20,
    ):
        self.max_length = max_length
        self.pad_value = pad_value
        self.max_channels = max_channels
        self.sensor_embed = SensorEmbedding(1)  # Just for index lookup
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_data = []
        batch_sensor_indices = []
        batch_channel_masks = []
        batch_sample_rates = []
        
        for item in batch:
            data = item["data"]  # [C, T] numpy array
            meta = item["meta"]
            
            # Replace NaN/Inf with 0 for numerical stability in FP16
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            C, T = data.shape
            
            # Truncate or pad time dimension
            if T > self.max_length:
                # Random crop
                start = np.random.randint(0, T - self.max_length)
                data = data[:, start:start + self.max_length]
            elif T < self.max_length:
                # Pad
                pad_width = ((0, 0), (0, self.max_length - T))
                data = np.pad(data, pad_width, mode='constant', constant_values=self.pad_value)
            
            # Get sensor indices from metadata
            channel_models = meta.get("channel_models", [])
            
            # Handle channel count mismatch
            if C < self.max_channels:
                # Pad channels
                pad_width = ((0, self.max_channels - C), (0, 0))
                data = np.pad(data, pad_width, mode='constant', constant_values=self.pad_value)
                channel_mask = np.zeros(self.max_channels, dtype=bool)
                channel_mask[:C] = True
                # Pad sensor indices
                sensor_indices = np.zeros(self.max_channels, dtype=np.int64)
                for i, model in enumerate(channel_models):
                    sensor_indices[i] = self.sensor_embed.get_sensor_idx(model)
            elif C > self.max_channels:
                # Randomly select max_channels from available channels
                selected_idx = np.random.choice(C, self.max_channels, replace=False)
                selected_idx = np.sort(selected_idx)  # Keep relative order
                data = data[selected_idx]
                channel_mask = np.ones(self.max_channels, dtype=bool)
                # Select corresponding sensor models
                sensor_indices = np.zeros(self.max_channels, dtype=np.int64)
                for i, orig_idx in enumerate(selected_idx):
                    if orig_idx < len(channel_models):
                        sensor_indices[i] = self.sensor_embed.get_sensor_idx(channel_models[orig_idx])
            else:
                # Exact match
                channel_mask = np.ones(self.max_channels, dtype=bool)
                sensor_indices = np.zeros(self.max_channels, dtype=np.int64)
                for i, model in enumerate(channel_models[:self.max_channels]):
                    sensor_indices[i] = self.sensor_embed.get_sensor_idx(model)
            
            # Get sample rate (default to 1 Hz if not specified)
            sample_rate = meta.get("sample_rate_hz", 1) or 1
            
            batch_data.append(data)
            batch_sensor_indices.append(sensor_indices)
            batch_channel_masks.append(channel_mask)
            batch_sample_rates.append(sample_rate)
        
        return {
            "data": torch.tensor(np.stack(batch_data), dtype=torch.float32),
            "sensor_indices": torch.tensor(np.stack(batch_sensor_indices), dtype=torch.long),
            "channel_mask": torch.tensor(np.stack(batch_channel_masks), dtype=torch.bool),
            "sample_rate": torch.tensor(batch_sample_rates, dtype=torch.float32),
        }


class NormalizedDatasetWrapper(Dataset):
    """Wrapper that uses get_normalized_sample interface."""
    
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data, meta = self.dataset.get_normalized_sample(idx)
        return {"data": data, "meta": meta}


class EnosePretrainingDataModule(L.LightningDataModule):
    """Lightning DataModule for e-nose pretraining.
    
    Combines multiple UCI e-nose datasets for self-supervised pretraining.
    """
    
    def __init__(
        self,
        root: Union[str, Path] = ".cache",
        datasets: Optional[List[str]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        max_length: int = 4096,
        max_channels: int = 20,
        val_split: float = 0.1,
        download: bool = False,
        seed: int = 42,
    ):
        """
        Args:
            root: Root directory for datasets
            datasets: List of dataset names to use. If None, uses all available.
            batch_size: Batch size for training
            num_workers: Number of dataloader workers
            max_length: Maximum sequence length (pad/truncate)
            max_channels: Maximum number of channels
            val_split: Fraction of data for validation
            download: Whether to download missing datasets
            seed: Random seed for train/val split
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.root = Path(root)
        self.datasets_to_use = datasets
        self.combined_dataset = None
        self.train_dataset = None
        self.val_dataset = None
    
    def prepare_data(self) -> None:
        """Download datasets if needed."""
        if self.hparams.download:
            _ = CombinedEnoseDataset(
                root=str(self.root),
                datasets=self.datasets_to_use,
                download=True,
            )
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Set up train and validation datasets."""
        if self.combined_dataset is None:
            self.combined_dataset = CombinedEnoseDataset(
                root=str(self.root),
                datasets=self.datasets_to_use,
                download=False,
            )
            
            # Wrap with normalized interface
            wrapped = NormalizedDatasetWrapper(self.combined_dataset)
            
            # Split into train/val
            n_samples = len(wrapped)
            n_val = int(n_samples * self.hparams.val_split)
            n_train = n_samples - n_val
            
            generator = torch.Generator().manual_seed(self.hparams.seed)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                wrapped, [n_train, n_val], generator=generator
            )
            
            print(f"Pretraining data: {n_train} train, {n_val} val samples")
            print(f"Datasets: {list(self.combined_dataset.datasets.keys())}")
            print(f"Sensor models: {self.combined_dataset.get_all_sensor_models()}")
    
    def train_dataloader(self) -> DataLoader:
        collator = PretrainingCollator(
            max_length=self.hparams.max_length,
            max_channels=self.hparams.max_channels,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=collator,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )
    
    def val_dataloader(self) -> DataLoader:
        collator = PretrainingCollator(
            max_length=self.hparams.max_length,
            max_channels=self.hparams.max_channels,
        )
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collator,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )
