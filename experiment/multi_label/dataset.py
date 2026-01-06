"""Dataset classes for multi-label gas classification.

Core idea:
- Train on pure gases (single-label)
- Test on mixtures (multi-label)
- See if model can identify component gases in mixtures
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from enose_uci_dataset.datasets import (
    TwinGasSensorArrays,
    GasSensorTurbulent,
    GasSensorDynamic,
)


class GasLabelEncoder:
    """Unified gas label encoder across datasets.
    
    Maps gas names to indices for multi-label classification.
    Handles different naming conventions across datasets.
    """
    
    # Canonical gas names from mixture datasets
    # GasSensorTurbulent: Ethylene + Methane or Ethylene + CO
    # GasSensorDynamic: Ethylene + CO or Ethylene + Methane
    GASES = [
        "Ethylene",
        "Methane", 
        "CO",
    ]
    
    # Name aliases for normalization
    ALIASES = {
        "ethylene": "Ethylene",
        "methane": "Methane",
        "co": "CO",
        "carbon_monoxide": "CO",
        "carbon monoxide": "CO",
    }
    
    def __init__(self):
        self.gas_to_idx = {gas: idx for idx, gas in enumerate(self.GASES)}
        self.idx_to_gas = {idx: gas for gas, idx in self.gas_to_idx.items()}
        self.num_classes = len(self.GASES)
    
    def normalize_name(self, name: str) -> str:
        """Normalize gas name to canonical form."""
        name_lower = name.lower().strip()
        if name_lower in self.ALIASES:
            return self.ALIASES[name_lower]
        # Check if it's already a canonical name (case-insensitive)
        for gas in self.GASES:
            if gas.lower() == name_lower:
                return gas
        return name  # Return as-is if not found
    
    def encode(self, gases: Union[str, List[str]]) -> np.ndarray:
        """Encode gas name(s) to multi-hot vector."""
        if isinstance(gases, str):
            gases = [gases]
        
        label = np.zeros(self.num_classes, dtype=np.float32)
        for gas in gases:
            normalized = self.normalize_name(gas)
            if normalized in self.gas_to_idx:
                label[self.gas_to_idx[normalized]] = 1.0
        return label
    
    def decode(self, label: np.ndarray, threshold: float = 0.5) -> List[str]:
        """Decode multi-hot vector to gas names."""
        gases = []
        indices = np.where(label > threshold)[0]
        for idx in indices:
            if idx in self.idx_to_gas:
                gases.append(self.idx_to_gas[idx])
        return gases


class MultiLabelGasDataset(Dataset):
    """Dataset for multi-label gas classification.
    
    Combines multiple source datasets and provides unified interface.
    
    Args:
        root: Root directory for datasets
        mode: 'train' (pure gases only) or 'test' (mixtures)
        sources: List of source datasets to use
        max_length: Maximum sequence length
        num_channels: Number of sensor channels to use
        download: Whether to download missing datasets
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        mode: str = "train",
        sources: Optional[List[str]] = None,
        max_length: int = 512,
        num_channels: int = 6,
        download: bool = False,
        smellnet_root: Optional[str] = None,
    ):
        self.root = Path(root)
        self.mode = mode
        self.max_length = max_length
        self.num_channels = num_channels
        self.label_encoder = GasLabelEncoder()
        
        # Default sources based on mode
        if sources is None:
            if mode == "train":
                sources = ["twin_gas_pure"]  # Pure gases: Ethylene, Methane, CO
            else:  # test
                sources = ["gas_sensor_turbulent", "gas_sensor_dynamic"]  # Mixtures
        
        self.sources = sources
        self.samples: List[Dict[str, Any]] = []
        
        # Load samples from each source
        for source in sources:
            self._load_source(source, download, smellnet_root)
        
        print(f"[{mode}] Loaded {len(self.samples)} samples from {sources}")
    
    def _load_source(self, source: str, download: bool, smellnet_root: Optional[str]) -> None:
        """Load samples from a specific source dataset."""
        
        if source == "twin_gas_pure":
            self._load_twin_gas_pure(download)
        elif source == "gas_sensor_turbulent":
            self._load_turbulent(download)
        elif source == "gas_sensor_dynamic":
            self._load_dynamic(download)
        else:
            print(f"Warning: Unknown source {source}")
    
    def _load_twin_gas_pure(self, download: bool) -> None:
        """Load TwinGasSensorArrays pure gas samples."""
        try:
            ds = TwinGasSensorArrays(
                root=str(self.root),
                download=download,
            )
            
            # Gas index mapping from TwinGasSensorArrays:
            # gas_to_idx = {"Ea": 0, "CO": 1, "Ey": 2, "Me": 3}
            gas_idx_to_name = {
                0: "Ethanol",   # Ea - NOT in mixture datasets, will be filtered
                1: "CO",        # CO
                2: "Ethylene",  # Ey
                3: "Methane",   # Me
            }
            
            # Only include gases that are in our target set (Ethylene, Methane, CO)
            target_gases = set(self.label_encoder.GASES)
            
            for idx in range(len(ds)):
                record = ds._samples[idx]
                gas_idx = record.target.get("gas", -1)
                gas_name = gas_idx_to_name.get(gas_idx, "unknown")
                
                # Skip Ethanol since it's not in the mixture datasets
                if gas_name not in target_gases:
                    continue
                
                self.samples.append({
                    "source": "twin_gas_pure",
                    "dataset": ds,
                    "index": idx,
                    "gases": [gas_name],
                    "is_mixture": False,
                })
        except Exception as e:
            print(f"Warning: Failed to load TwinGasSensorArrays: {e}")
    
    def _load_turbulent(self, download: bool) -> None:
        """Load GasSensorTurbulent mixture samples."""
        try:
            ds = GasSensorTurbulent(
                root=str(self.root),
                download=download,
            )
            
            for idx in range(len(ds)):
                record = ds._samples[idx]
                meta = record.meta
                
                # Always has Ethylene + (Methane or CO)
                gases = ["Ethylene"]
                second_gas = meta.get("second_gas_name", "Methane")
                gases.append(second_gas)
                
                # Filter out zero concentration components
                eth_level = record.target.get("ethylene_level", 0)
                second_level = record.target.get("second_level", 0)
                
                active_gases = []
                if eth_level > 0:
                    active_gases.append("Ethylene")
                if second_level > 0:
                    active_gases.append(second_gas)
                
                # If both are zero, skip this sample
                if len(active_gases) == 0:
                    continue
                
                self.samples.append({
                    "source": "gas_sensor_turbulent",
                    "dataset": ds,
                    "index": idx,
                    "gases": active_gases,
                    "is_mixture": len(active_gases) > 1,
                })
        except Exception as e:
            print(f"Warning: Failed to load GasSensorTurbulent: {e}")
    
    def _load_dynamic(self, download: bool) -> None:
        """Load GasSensorDynamic mixture samples."""
        try:
            ds = GasSensorDynamic(
                root=str(self.root),
                mixture="both",
                download=download,
            )
            
            for idx in range(len(ds)):
                record = ds._samples[idx]
                meta = record.meta
                
                # Ethylene + (CO or Methane)
                gas2_name = meta.get("gas2_name", "CO")
                eth_ppm = record.target.get("ethylene_ppm", 0)
                gas2_ppm = record.target.get("gas2_ppm", 0)
                
                active_gases = []
                if eth_ppm is not None and eth_ppm > 0:
                    active_gases.append("Ethylene")
                if gas2_ppm is not None and gas2_ppm > 0:
                    active_gases.append(gas2_name)
                
                if len(active_gases) == 0:
                    continue
                
                self.samples.append({
                    "source": "gas_sensor_dynamic",
                    "dataset": ds,
                    "index": idx,
                    "gases": active_gases,
                    "is_mixture": len(active_gases) > 1,
                })
        except Exception as e:
            print(f"Warning: Failed to load GasSensorDynamic: {e}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        ds = sample["dataset"]
        ds_idx = sample["index"]
        
        # Load data from source dataset
        record = ds._samples[ds_idx]
        data, target = ds._load_sample(record)
        
        # Convert to numpy if DataFrame
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        # Ensure [T, C] format
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Handle [C, T] vs [T, C] format
        # Most datasets return [T, C], but SmellNet returns [T, C] after CSV read
        T, C = data.shape
        if C > T:  # Likely [C, T] format
            data = data.T
            T, C = data.shape
        
        # Truncate/pad time dimension
        if T > self.max_length:
            start = np.random.randint(0, T - self.max_length)
            data = data[start:start + self.max_length]
        elif T < self.max_length:
            pad_width = ((0, self.max_length - T), (0, 0))
            data = np.pad(data, pad_width, mode='constant', constant_values=0)
        
        # Truncate/pad channel dimension
        if C > self.num_channels:
            data = data[:, :self.num_channels]
        elif C < self.num_channels:
            pad_width = ((0, 0), (0, self.num_channels - C))
            data = np.pad(data, pad_width, mode='constant', constant_values=0)
        
        # Normalize per sample (z-score)
        data = data.astype(np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        mean = data.mean()
        std = data.std() + 1e-8
        data = (data - mean) / std
        
        # Transpose to [C, T] for model input
        data = data.T
        
        # Encode labels
        label = self.label_encoder.encode(sample["gases"])
        
        return {
            "data": torch.from_numpy(data),
            "label": torch.from_numpy(label),
            "is_mixture": torch.tensor(sample["is_mixture"], dtype=torch.bool),
            "gases": sample["gases"],
            "source": sample["source"],
        }
    
    @property
    def num_classes(self) -> int:
        return self.label_encoder.num_classes


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for DataLoader."""
    return {
        "data": torch.stack([b["data"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "is_mixture": torch.stack([b["is_mixture"] for b in batch]),
        "gases": [b["gases"] for b in batch],
        "source": [b["source"] for b in batch],
    }
