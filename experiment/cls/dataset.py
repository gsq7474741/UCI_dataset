"""Single-dataset classification with global label space.

Each dataset is trained independently, but labels are mapped to a unified
global label space (GasLabel enum) for consistent evaluation across datasets.
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
    DATASETS,
    GasLabel,
    GAS_LABEL_MAPPINGS,
    get_global_label,
)
from enose_uci_dataset.datasets.external.smellnet import (
    SMELLNET_PURE_MEAN,
    SMELLNET_PURE_STD,
    SMELLNET_MIXTURE_MEAN,
    SMELLNET_MIXTURE_STD,
)
from enose_uci_dataset.datasets.external.g919_55 import (
    G919_IDX_TO_ODOR,
    G919_NUM_CLASSES,
)


# Mapping from dataset name to local label extraction function
# Returns (local_label_idx, label_name) from target dict
def _extract_twin_gas_label(target: Dict) -> Tuple[int, str]:
    """TwinGasSensorArrays: gas index 0-3 (Ea, CO, Ey, Me)"""
    gas_idx = target.get("gas", 0)
    names = {0: "Ethanol", 1: "CO", 2: "Ethylene", 3: "Methane"}
    return gas_idx, names.get(gas_idx, "Unknown")


def _extract_turbulent_label(target: Dict) -> Tuple[int, str]:
    """GasSensorTurbulent: second_gas 0=Methane, 1=CO (mixed with Ethylene)"""
    second_gas = target.get("second_gas", 0)
    # For classification, use mixture type as label
    names = {0: "Ethylene+Methane", 1: "Ethylene+CO"}
    return second_gas, names.get(second_gas, "Unknown")


def _extract_dynamic_label(target: Dict) -> Tuple[int, str]:
    """GasSensorDynamic: mixture 0=ethylene_co, 1=ethylene_methane"""
    mixture = target.get("mixture", 0)
    names = {0: "Ethylene+CO", 1: "Ethylene+Methane"}
    return mixture, names.get(mixture, "Unknown")


def _extract_low_conc_label(target: Dict) -> Tuple[int, str]:
    """GasSensorLowConcentration: gas_idx 0-5"""
    gas_idx = target.get("gas_idx", 0)
    names = {
        0: "Ethanol", 1: "Acetone", 2: "Toluene",
        3: "Ethyl Acetate", 4: "Isopropanol", 5: "n-Hexane"
    }
    return gas_idx, names.get(gas_idx, "Unknown")


def _extract_flow_mod_label(target: Dict) -> Tuple[int, str]:
    """GasSensorFlowModulation: gas_class 0-3"""
    gas_class = target.get("gas_class", 0)
    names = {0: "Air", 1: "Acetone", 2: "Ethanol", 3: "Acetone+Ethanol"}
    return gas_class, names.get(gas_class, "Unknown")


def _extract_qcm_label(target: Dict) -> Tuple[int, str]:
    """AlcoholQCMSensor: class_idx 0-4"""
    class_idx = target.get("class_idx", 0)
    names = {
        0: "1-Octanol", 1: "1-Propanol", 2: "2-Butanol",
        3: "2-Propanol", 4: "1-Isobutanol"
    }
    return class_idx, names.get(class_idx, "Unknown")


def _extract_drift_label(target: Dict) -> Tuple[int, str]:
    """GasSensorArrayDrift: gas 0-5"""
    gas = target.get("gas", 0)
    names = {
        0: "Ethanol", 1: "Ethylene", 2: "Ammonia",
        3: "Acetaldehyde", 4: "Acetone", 5: "Toluene"
    }
    return gas, names.get(gas, "Unknown")


def _extract_temp_mod_label(target: Dict) -> Tuple[int, str]:
    """GasSensorTemperatureModulation: gas concentration level"""
    # This dataset doesn't have discrete gas classes
    # Use concentration level as pseudo-label
    conc = target.get("concentration", 0)
    if conc == 0:
        return 0, "Background"
    elif conc <= 5:
        return 1, "Low"
    elif conc <= 10:
        return 2, "Medium"
    else:
        return 3, "High"


def _extract_home_label(target: Dict) -> Tuple[int, str]:
    """GasSensorsForHomeActivityMonitoring: activity label"""
    # This is activity recognition, not gas classification
    label = target.get("label", 0)
    return label, f"Activity_{label}"


def _extract_smellnet_label(target: Dict) -> Tuple[int, str]:
    """SmellNet: ingredient classification (50 classes for pure, 43 for mixture)"""
    label = target.get("label", 0)
    ingredient = target.get("ingredient", f"Class_{label}")
    return label, ingredient


def _extract_g919_label(target: Dict) -> Tuple[int, str]:
    """G919: 55 odor classification"""
    odor_idx = target.get("odor_idx", 0)
    odor_name = target.get("odor_name", G919_IDX_TO_ODOR.get(odor_idx, f"Odor_{odor_idx}"))
    return odor_idx, odor_name


LABEL_EXTRACTORS = {
    "twin_gas_sensor_arrays": _extract_twin_gas_label,
    "gas_sensor_array_exposed_to_turbulent_gas_mixtures": _extract_turbulent_label,
    "gas_sensor_array_under_dynamic_gas_mixtures": _extract_dynamic_label,
    "gas_sensor_array_low_concentration": _extract_low_conc_label,
    "gas_sensor_array_under_flow_modulation": _extract_flow_mod_label,
    "alcohol_qcm_sensor_dataset": _extract_qcm_label,
    "gas_sensor_array_drift_dataset_at_different_concentrations": _extract_drift_label,
    "gas_sensor_array_temperature_modulation": _extract_temp_mod_label,
    "gas_sensors_for_home_activity_monitoring": _extract_home_label,
    "smellnet": _extract_smellnet_label,
    "g919_55": _extract_g919_label,
}


class GlobalLabelEncoder:
    """Encoder that maps dataset-local labels to global GasLabel space.
    
    Uses GAS_LABEL_MAPPINGS from _global.py for consistent mapping.
    """
    
    def __init__(self):
        # Build reverse mapping: GasLabel -> index in our classification
        # We only use non-UNKNOWN labels that appear in datasets
        self.used_labels = set()
        for dataset_mapping in GAS_LABEL_MAPPINGS.values():
            for gas_label in dataset_mapping.values():
                if gas_label != GasLabel.UNKNOWN:
                    self.used_labels.add(gas_label)
        
        # Sort by enum value for consistency
        self.used_labels = sorted(self.used_labels, key=lambda x: x.value)
        
        # Create mappings
        self.label_to_idx = {label: idx for idx, label in enumerate(self.used_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.used_labels)
        
        # Create name mapping
        self.idx_to_name = {idx: label.name for idx, label in self.idx_to_label.items()}
        self.name_to_idx = {name: idx for idx, name in self.idx_to_name.items()}
    
    def encode(self, dataset_name: str, local_label: int) -> int:
        """Encode dataset-local label to global index."""
        global_label = get_global_label(dataset_name, local_label)
        gas_label = GasLabel(global_label)
        
        if gas_label in self.label_to_idx:
            return self.label_to_idx[gas_label]
        return -1  # Unknown
    
    def decode(self, idx: int) -> str:
        """Decode global index to label name."""
        return self.idx_to_name.get(idx, "UNKNOWN")
    
    def get_class_names(self) -> List[str]:
        """Get list of all class names in order."""
        return [self.idx_to_name[i] for i in range(self.num_classes)]


class SingleDatasetClassification(Dataset):
    """Classification dataset for a single e-nose dataset.
    
    Maps local labels to global label space for unified evaluation.
    
    Args:
        root: Root directory for datasets
        dataset_name: Name of the dataset to use
        split: 'train', 'val', or 'test'
        max_length: Maximum sequence length
        num_channels: Number of sensor channels
        download: Whether to download missing dataset
        train_ratio: Ratio of data for training (rest split between val/test)
        seed: Random seed for split
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        dataset_name: str,
        split: str = "train",
        max_length: int = 512,
        num_channels: int = 8,
        download: bool = False,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
        trim_start: int = 0,
        trim_end: int = 0,
        frequency_domain: bool = False,  # Use FFT frequency domain
        fft_cutoff_hz: float = 0,  # FFT cutoff frequency in Hz (0=all)
        subset: Optional[str] = None,  # For SmellNet: 'pure' or 'mixture'
        window_size: int = 0,  # Sliding window size (0=full sequence)
        window_stride: int = 0,  # Sliding window stride (0=window_size//2)
        scaler_mean: Optional[np.ndarray] = None,  # Pre-computed mean for standardization
        scaler_std: Optional[np.ndarray] = None,   # Pre-computed std for standardization
        lag: int = 0,  # Lag for difference features (0=disabled, typical: 25)
        dual_view: bool = False,  # Return both time and freq domain for fusion
    ):
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.split = split
        self.max_length = max_length
        self.num_channels = num_channels
        self.trim_start = trim_start  # Start from this index (absolute)
        self.trim_end = trim_end      # End at this index (absolute, 0=no limit)
        self.frequency_domain = frequency_domain
        self.fft_cutoff_hz = fft_cutoff_hz
        self.subset = subset
        self.window_size = window_size
        self.window_stride = window_stride if window_stride > 0 else (window_size // 2 if window_size > 0 else 0)
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        self.lag = lag
        self.dual_view = dual_view
        
        # Get dataset class
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")
        
        dataset_cls = DATASETS[dataset_name]
        
        # Load dataset with subset if applicable (e.g., SmellNet)
        if dataset_name == "smellnet":
            # SmellNet uses smellnet_root for local data path
            # Load ALL data (split=None), then split internally by train_ratio/val_ratio
            self.dataset = dataset_cls(
                root=str(self.root), 
                download=download, 
                subset=subset or "pure",
                split=None,  # Load all splits, we do our own splitting
                smellnet_root=str(self.root) if (self.root / "data").exists() else None,
            )
        else:
            self.dataset = dataset_cls(root=str(self.root), download=download)
        
        # Get sample rate from dataset metadata
        self.sample_rate = self.dataset.sample_rate_hz or 100.0  # fallback to 100Hz
        
        # Get label extractor
        self.label_extractor = LABEL_EXTRACTORS.get(dataset_name)
        if self.label_extractor is None:
            raise ValueError(f"No label extractor for dataset: {dataset_name}")
        
        # Build local label encoder (dataset-specific classes)
        self._build_local_labels()
        
        # Split dataset
        self._split_dataset(train_ratio, val_ratio, seed)
        
        # Build sliding window indices if window_size > 0
        self._build_window_indices()
        
        window_info = f", window={self.window_size}x{self.window_stride}" if self.window_size > 0 else ""
        print(f"[{dataset_name}][{split}] Loaded {len(self)} samples, "
              f"{self.num_classes} classes{window_info}")
    
    def _build_local_labels(self):
        """Build local label space from dataset samples."""
        label_counts = {}
        
        for idx in range(len(self.dataset)):
            record = self.dataset._samples[idx]
            local_label, label_name = self.label_extractor(record.target)
            
            if local_label not in label_counts:
                label_counts[local_label] = {"name": label_name, "count": 0}
            label_counts[local_label]["count"] += 1
        
        # Sort by label index
        sorted_labels = sorted(label_counts.keys())
        
        self.local_to_idx = {label: idx for idx, label in enumerate(sorted_labels)}
        self.idx_to_local = {idx: label for label, idx in self.local_to_idx.items()}
        self.idx_to_name = {idx: label_counts[self.idx_to_local[idx]]["name"] 
                           for idx in range(len(sorted_labels))}
        self.num_classes = len(sorted_labels)
        
        # Store label distribution
        self.label_counts = {self.local_to_idx[k]: v["count"] for k, v in label_counts.items()}
    
    def _split_dataset(self, train_ratio: float, val_ratio: float, seed: int):
        """Split dataset into train/val/test.
        
        For datasets with original split info (e.g., SmellNet), respects original splits:
        - Original 'train' samples → our train + val (stratified)
        - Original 'test'/'test_seen' samples → our test
        
        For other datasets, uses stratified random splitting.
        """
        np.random.seed(seed)
        
        # Check if dataset has original split info ('split' or 'is_train')
        has_split_info = (hasattr(self.dataset, '_samples') and 
                          len(self.dataset._samples) > 0 and
                          ('split' in self.dataset._samples[0].meta or 
                           'is_train' in self.dataset._samples[0].meta))
        
        if has_split_info:
            # Respect original train/test splits
            orig_train_indices = []
            orig_test_indices = []
            
            for idx in range(len(self.dataset)):
                record = self.dataset._samples[idx]
                # Check both 'split' and 'is_train' meta fields
                if 'is_train' in record.meta:
                    is_train = record.meta.get('is_train', True)
                else:
                    orig_split = record.meta.get('split', 'train')
                    is_train = (orig_split == 'train')
                
                if is_train:
                    orig_train_indices.append(idx)
                else:  # test, test_seen, test_unseen
                    orig_test_indices.append(idx)
            
            # Split original train into our train + val (stratified by label)
            label_to_train_indices = {i: [] for i in range(self.num_classes)}
            for idx in orig_train_indices:
                record = self.dataset._samples[idx]
                local_label, _ = self.label_extractor(record.target)
                label_idx = self.local_to_idx[local_label]
                label_to_train_indices[label_idx].append(idx)
            
            train_indices, val_indices = [], []
            # Use val_ratio to split original train into train+val
            # e.g., if val_ratio=0.15, then 85% train, 15% val from original train
            val_from_train_ratio = val_ratio / (train_ratio + val_ratio)
            
            for label_idx, indices in label_to_train_indices.items():
                np.random.shuffle(indices)
                n = len(indices)
                n_val = max(1, int(n * val_from_train_ratio))  # At least 1 for val
                val_indices.extend(indices[:n_val])
                train_indices.extend(indices[n_val:])
            
            test_indices = orig_test_indices
            
            np.random.shuffle(train_indices)
            np.random.shuffle(val_indices)
            np.random.shuffle(test_indices)
        else:
            # Original behavior: stratified random split
            label_indices = {i: [] for i in range(self.num_classes)}
            
            for idx in range(len(self.dataset)):
                record = self.dataset._samples[idx]
                local_label, _ = self.label_extractor(record.target)
                label_idx = self.local_to_idx[local_label]
                label_indices[label_idx].append(idx)
            
            train_indices, val_indices, test_indices = [], [], []
            
            for label_idx, indices in label_indices.items():
                np.random.shuffle(indices)
                n = len(indices)
                n_train = int(n * train_ratio)
                n_val = int(n * val_ratio)
                
                train_indices.extend(indices[:n_train])
                val_indices.extend(indices[n_train:n_train + n_val])
                test_indices.extend(indices[n_train + n_val:])
            
            np.random.shuffle(train_indices)
            np.random.shuffle(val_indices)
            np.random.shuffle(test_indices)
        
        if self.split == "train":
            self.indices = train_indices
        elif self.split == "val":
            self.indices = val_indices
        else:  # test
            self.indices = test_indices
    
    def _build_window_indices(self):
        """Build sliding window indices: [(sample_idx, window_start), ...]"""
        if self.window_size <= 0:
            # No windowing - each sample is one item
            self.window_indices = [(idx, 0) for idx in self.indices]
            return
        
        self.window_indices = []
        for ds_idx in self.indices:
            record = self.dataset._samples[ds_idx]
            # Get sample length
            data, _ = self.dataset._load_sample(record)
            if isinstance(data, pd.DataFrame):
                T = len(data)
            else:
                T = len(data) if hasattr(data, '__len__') else data.shape[0]
            
            # Apply trim bounds to get effective length
            start = self.trim_start
            end = self.trim_end if self.trim_end > 0 else T
            effective_T = end - start
            
            # Generate windows
            if effective_T >= self.window_size:
                for win_start in range(0, effective_T - self.window_size + 1, self.window_stride):
                    self.window_indices.append((ds_idx, start + win_start))
            else:
                # Sample too short - use entire sample
                self.window_indices.append((ds_idx, start))
    
    def __len__(self) -> int:
        if self.window_size > 0:
            return len(self.window_indices)
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get sample and window info
        if self.window_size > 0:
            ds_idx, win_start = self.window_indices[idx]
        else:
            ds_idx = self.indices[idx]
            win_start = 0
        
        record = self.dataset._samples[ds_idx]
        
        # Load data
        data, target = self.dataset._load_sample(record)
        
        # Convert to numpy
        if isinstance(data, pd.DataFrame):
            data = data.select_dtypes(include=[np.number]).values
        data = np.array(data, dtype=np.float32)
        
        # Ensure [T, C] format
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        T, C = data.shape
        if C > T:  # Likely [C, T] format
            data = data.T
            T, C = data.shape
        
        # Apply windowing or trimming
        if self.window_size > 0:
            # Use sliding window
            win_end = min(win_start + self.window_size, T)
            data = data[win_start:win_end]
            T = data.shape[0]
        elif self.trim_start > 0 or self.trim_end > 0:
            # Apply time trimming (absolute indices)
            start_idx = self.trim_start
            end_idx = self.trim_end if self.trim_end > 0 else T  # 0 means no limit
            if start_idx < end_idx and end_idx <= T:
                data = data[start_idx:end_idx]
                T = data.shape[0]
        
        # Pad/truncate channel dimension only
        if C > self.num_channels:
            data = data[:, :self.num_channels]
        elif C < self.num_channels:
            pad_width = ((0, 0), (0, self.num_channels - C))
            data = np.pad(data, pad_width, mode='constant', constant_values=0)
        
        # Truncate/pad to max_length if specified
        if self.max_length > 0:
            if T > self.max_length:
                # Truncate to max_length
                data = data[:self.max_length, :]
                T = self.max_length
            elif T < self.max_length:
                # Pad with zeros to max_length
                pad_width = ((0, self.max_length - T), (0, 0))
                data = np.pad(data, pad_width, mode='constant', constant_values=0)
                T = self.max_length
        
        # Handle NaN/inf
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Transpose to [C, T] for Conv1d
        data = data.T  # [C, T]
        
        # Apply StandardScaler if provided (per-channel standardization)
        if self.scaler_mean is not None and self.scaler_std is not None:
            # data: [C, T], scaler_mean/std: [C]
            data = (data - self.scaler_mean[:, None]) / (self.scaler_std[:, None] + 1e-8)
        
        # Apply lag difference features: x[t] - x[t-lag]
        if self.lag > 0:
            # data: [C, T]
            # Compute difference: current - lagged
            # Pad beginning with zeros (first lag values have no reference)
            diff = data[:, self.lag:] - data[:, :-self.lag]  # [C, T-lag]
            # Pad to maintain length (or just use shorter sequence)
            padding = np.zeros((data.shape[0], self.lag), dtype=data.dtype)
            data = np.concatenate([padding, diff], axis=1)  # [C, T]
        
        # Dual-view mode: return both time-domain (with lag) and frequency-domain
        if self.dual_view:
            # Time-domain data (already has lag applied if lag > 0)
            data_time = data.copy()
            
            # Frequency-domain data
            signal_length = data.shape[1]
            fft_data = np.fft.rfft(data, axis=1)
            data_freq = np.log1p(np.abs(fft_data)).astype(np.float32)
            
            # Get label
            local_label, label_name = self.label_extractor(record.target)
            label_idx = self.local_to_idx[local_label]
            
            return {
                "data_time": torch.from_numpy(data_time),
                "data_freq": torch.from_numpy(data_freq),
                "label": torch.tensor(label_idx, dtype=torch.long),
                "length": T,
                "local_label": local_label,
                "label_name": label_name,
            }
        
        # Apply FFT if frequency domain mode
        if self.frequency_domain:
            # FFT per channel, keep magnitude (power spectrum)
            signal_length = data.shape[1]  # T after trim
            fft_data = np.fft.rfft(data, axis=1)  # [C, T//2+1]
            # Use log magnitude for better scale
            data = np.log1p(np.abs(fft_data)).astype(np.float32)
            # Apply frequency cutoff if specified (convert Hz to bin index)
            if self.fft_cutoff_hz > 0:
                # freq_resolution = sample_rate / signal_length
                # cutoff_bin = cutoff_hz / freq_resolution = cutoff_hz * signal_length / sample_rate
                cutoff_bin = int(self.fft_cutoff_hz * signal_length / self.sample_rate)
                cutoff_bin = min(cutoff_bin, data.shape[1])  # Don't exceed available bins
                data = data[:, :cutoff_bin]
        
        # Get label
        local_label, label_name = self.label_extractor(record.target)
        label_idx = self.local_to_idx[local_label]
        
        return {
            "data": torch.from_numpy(data),
            "label": torch.tensor(label_idx, dtype=torch.long),
            "length": T,  # Store original length for padding
            "local_label": local_label,
            "label_name": label_name,
        }
    
    def get_class_names(self) -> List[str]:
        """Get list of class names in order."""
        return [self.idx_to_name[i] for i in range(self.num_classes)]
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced data (inverse frequency)."""
        total = sum(self.label_counts.values())
        weights = []
        for i in range(self.num_classes):
            count = self.label_counts.get(i, 1)
            weights.append(total / (self.num_classes * count))
        return torch.tensor(weights, dtype=torch.float32)
    
    def compute_scaler_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and std across all samples for StandardScaler.
        
        Returns:
            Tuple of (mean, std) arrays with shape [C]
        """
        # Collect all data to compute statistics
        all_data = []
        for idx in range(len(self)):
            sample = self[idx]
            data = sample["data"].numpy()  # [C, T]
            all_data.append(data)
        
        # Stack and compute per-channel statistics
        # all_data: list of [C, T] arrays with varying T
        channel_values = [[] for _ in range(self.num_channels)]
        for data in all_data:
            for c in range(data.shape[0]):
                channel_values[c].extend(data[c].tolist())
        
        mean = np.array([np.mean(v) for v in channel_values], dtype=np.float32)
        std = np.array([np.std(v) for v in channel_values], dtype=np.float32)
        
        return mean, std


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader with dynamic padding."""
    # Find max length in batch
    max_len = max(b["data"].shape[1] for b in batch)
    num_channels = batch[0]["data"].shape[0]
    
    # Pad all samples to max length
    padded_data = []
    for b in batch:
        data = b["data"]  # [C, T]
        T = data.shape[1]
        if T < max_len:
            # Pad with zeros
            pad = torch.zeros(num_channels, max_len - T, dtype=data.dtype)
            data = torch.cat([data, pad], dim=1)
        padded_data.append(data)
    
    return {
        "data": torch.stack(padded_data),
        "label": torch.stack([b["label"] for b in batch]),
        "length": torch.tensor([b["length"] for b in batch]),
        "local_label": [b["local_label"] for b in batch],
        "label_name": [b["label_name"] for b in batch],
    }


def collate_fn_dual_view(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate function for dual-view DataLoader with dynamic padding."""
    # Find max lengths
    max_len_time = max(b["data_time"].shape[1] for b in batch)
    max_len_freq = max(b["data_freq"].shape[1] for b in batch)
    num_channels = batch[0]["data_time"].shape[0]
    
    # Pad time-domain data
    padded_time = []
    for b in batch:
        data = b["data_time"]
        T = data.shape[1]
        if T < max_len_time:
            pad = torch.zeros(num_channels, max_len_time - T, dtype=data.dtype)
            data = torch.cat([data, pad], dim=1)
        padded_time.append(data)
    
    # Pad frequency-domain data
    padded_freq = []
    for b in batch:
        data = b["data_freq"]
        F = data.shape[1]
        if F < max_len_freq:
            pad = torch.zeros(num_channels, max_len_freq - F, dtype=data.dtype)
            data = torch.cat([data, pad], dim=1)
        padded_freq.append(data)
    
    return {
        "data_time": torch.stack(padded_time),
        "data_freq": torch.stack(padded_freq),
        "label": torch.stack([b["label"] for b in batch]),
        "length": torch.tensor([b["length"] for b in batch]),
        "local_label": [b["local_label"] for b in batch],
        "label_name": [b["label_name"] for b in batch],
    }


# Dataset metadata for classification experiments
CLASSIFICATION_DATASETS = {
    "twin_gas_sensor_arrays": {
        "num_classes": 4,
        "task": "gas_classification",
        "description": "4 pure gases: Ethanol, CO, Ethylene, Methane",
    },
    "gas_sensor_array_drift_dataset_at_different_concentrations": {
        "num_classes": 6,
        "task": "gas_classification",
        "description": "6 gases with drift: Ethanol, Ethylene, Ammonia, Acetaldehyde, Acetone, Toluene",
    },
    "gas_sensor_array_low_concentration": {
        "num_classes": 6,
        "task": "gas_classification",
        "description": "6 VOCs at low concentration",
    },
    "gas_sensor_array_under_flow_modulation": {
        "num_classes": 4,
        "task": "gas_classification",
        "description": "Air, Acetone, Ethanol, Mixture",
    },
    "alcohol_qcm_sensor_dataset": {
        "num_classes": 5,
        "task": "alcohol_classification",
        "description": "5 alcohols via QCM sensor",
    },
    "gas_sensor_array_exposed_to_turbulent_gas_mixtures": {
        "num_classes": 2,
        "task": "mixture_classification",
        "description": "Ethylene+Methane vs Ethylene+CO",
    },
    "gas_sensor_array_under_dynamic_gas_mixtures": {
        "num_classes": 2,
        "task": "mixture_classification",
        "description": "Ethylene+CO vs Ethylene+Methane",
    },
    "smellnet": {
        "num_classes": 50,  # 50 for pure, 43 for mixture
        "task": "ingredient_classification",
        "description": "SmellNet: 50 pure (1Hz) or 43 mixture (10Hz) classes",
    },
    "g919_55": {
        "num_classes": 55,
        "task": "odor_classification",
        "description": "SJTU-G919: 55 odors across 8 categories (perfume, condiment, fruit, milk, spice, vegetable, wine, drink)",
    },
}


def list_classification_datasets() -> List[str]:
    """List available datasets for classification."""
    return list(CLASSIFICATION_DATASETS.keys())
