"""Downstream task implementations.

Currently implemented:
- ConcentrationRegressionTask: Predict gas concentration from sensor data
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .base import BaseDownstreamTask


class ConcentrationDataset(Dataset):
    """Dataset for concentration regression task.
    
    Returns raw sensor data and concentration target.
    No normalization - downstream model handles that.
    """
    
    def __init__(
        self,
        data: np.ndarray,  # [N, C, T]
        targets: np.ndarray,  # [N] or [N, D]
        seq_len: int = 1000,
    ):
        self.data = data
        self.targets = targets
        self.seq_len = seq_len
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = torch.from_numpy(self.data[idx]).float()  # [C, T]
        target = torch.tensor(self.targets[idx]).float()
        
        # Ensure sequence length
        C, T = x.shape
        if T > self.seq_len:
            x = x[:, :self.seq_len]
        elif T < self.seq_len:
            x = torch.nn.functional.pad(x, (0, self.seq_len - T), value=0)
        
        return {"data": x, "target": target}


class ConcentrationRegressionTask(BaseDownstreamTask):
    """Concentration regression task using Twin Gas dataset.
    
    Task: Predict gas concentration (ppm) from raw sensor readings.
    Input: Raw sensor data [B, 8, T] (no normalization)
    Output: Concentration value (scalar)
    """
    
    name = "concentration_regression"
    
    def __init__(
        self,
        data_root: Union[str, Path] = ".cache",
        seq_len: int = 1000,
        batch_size: int = 32,
        num_workers: int = 4,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        truncate_ratio: float = 0.25,
        truncate_start_ratio: float = 0.0666,
        normalize_target: bool = False,  # Use raw ppm values
        **kwargs,
    ):
        super().__init__(data_root, seq_len, batch_size, num_workers, **kwargs)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.truncate_ratio = truncate_ratio
        self.truncate_start_ratio = truncate_start_ratio
        self.normalize_target = normalize_target
        
        self._num_channels = 8
        self._target_mean = 0.0
        self._target_std = 1.0
    
    def setup(self) -> None:
        """Load Twin Gas dataset and create splits."""
        from ...datasets import TwinGasSensorArrays
        
        dataset = TwinGasSensorArrays(str(self.data_root), download=True)
        print(f"[ConcentrationTask] Loaded {len(dataset)} samples")
        
        # Extract data and targets
        all_data = []
        all_targets = []
        
        for idx in range(len(dataset)):
            rec = dataset._samples[idx]
            df, target = dataset._load_sample(rec)
            
            # Get raw sensor data [T, C]
            data_raw = df.values.astype(np.float32)
            orig_len = data_raw.shape[0]
            
            # Truncate to gas response segment
            start_idx = int(orig_len * self.truncate_start_ratio)
            end_idx = int(orig_len * self.truncate_ratio)
            start_idx = max(0, min(start_idx, orig_len - 1))
            end_idx = max(start_idx + 1, min(end_idx, orig_len))
            data_raw = data_raw[start_idx:end_idx]
            
            T, C = data_raw.shape
            
            # Downsample to seq_len
            if T > self.seq_len:
                idx_t = np.linspace(0, T - 1, self.seq_len, dtype=np.float32)
                idx_floor = np.floor(idx_t).astype(np.int32)
                idx_ceil = np.minimum(idx_floor + 1, T - 1)
                alpha = idx_t - idx_floor
                data_raw = data_raw[idx_floor] * (1 - alpha[:, None]) + data_raw[idx_ceil] * alpha[:, None]
            elif T < self.seq_len:
                pad = np.zeros((self.seq_len - T, C), dtype=np.float32)
                data_raw = np.concatenate([data_raw, pad], axis=0)
            
            # Transpose to [C, T]
            data_raw = data_raw.T  # [C, T]
            
            all_data.append(data_raw)
            all_targets.append(target["ppm"])
        
        all_data = np.stack(all_data, axis=0)  # [N, C, T]
        all_targets = np.array(all_targets, dtype=np.float32)  # [N]
        
        # Normalize targets for stable training
        if self.normalize_target:
            self._target_mean = all_targets.mean()
            self._target_std = all_targets.std()
            all_targets = (all_targets - self._target_mean) / (self._target_std + 1e-8)
        
        # Create train/val/test splits
        n_samples = len(all_data)
        indices = np.random.permutation(n_samples)
        
        n_train = int(n_samples * self.train_ratio)
        n_val = int(n_samples * self.val_ratio)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        self._train_dataset = ConcentrationDataset(
            all_data[train_idx], all_targets[train_idx], self.seq_len
        )
        self._val_dataset = ConcentrationDataset(
            all_data[val_idx], all_targets[val_idx], self.seq_len
        )
        self._test_dataset = ConcentrationDataset(
            all_data[test_idx], all_targets[test_idx], self.seq_len
        )
        
        print(f"[ConcentrationTask] Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        print(f"[ConcentrationTask] Target stats: mean={self._target_mean:.2f}, std={self._target_std:.2f}")
    
    def get_num_channels(self) -> int:
        return self._num_channels
    
    def get_target_dim(self) -> int:
        return 1
    
    def denormalize_target(self, pred: torch.Tensor) -> torch.Tensor:
        """Convert normalized prediction back to original scale."""
        if self.normalize_target:
            return pred * self._target_std + self._target_mean
        return pred
    
    @property
    def target_mean(self) -> float:
        return self._target_mean
    
    @property
    def target_std(self) -> float:
        return self._target_std
