"""Combined E-nose Dataset for cross-dataset pretraining.

This module provides utilities for combining multiple UCI e-nose datasets
into a unified dataset for pretraining foundation models.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from torch.utils.data import Dataset as _TorchDataset
except Exception:
    class _TorchDataset:
        pass

from ._base import BaseEnoseDataset
from ._info import ChannelConfig, get_dataset_info, list_datasets


def _get_datasets_registry():
    """Lazy import to avoid circular dependency."""
    from .alcohol_qcm_sensor import AlcoholQCMSensor
    from .gas_sensor_array_drift import GasSensorArrayDrift
    from .gas_sensor_dynamic import GasSensorDynamic
    from .gas_sensor_flow_modulation import GasSensorFlowModulation
    from .gas_sensor_low_concentration import GasSensorLowConcentration
    from .gas_sensor_temperature_modulation import GasSensorTemperatureModulation
    from .gas_sensor_turbulent import GasSensorTurbulent
    from .gas_sensors_for_home_activity_monitoring import GasSensorsForHomeActivityMonitoring
    from .twin_gas_sensor_arrays import TwinGasSensorArrays
    
    return {
        "alcohol_qcm_sensor_dataset": AlcoholQCMSensor,
        "gas_sensor_array_drift_dataset_at_different_concentrations": GasSensorArrayDrift,
        "gas_sensor_array_exposed_to_turbulent_gas_mixtures": GasSensorTurbulent,
        "gas_sensor_array_low_concentration": GasSensorLowConcentration,
        "gas_sensor_array_temperature_modulation": GasSensorTemperatureModulation,
        "gas_sensor_array_under_dynamic_gas_mixtures": GasSensorDynamic,
        "gas_sensor_array_under_flow_modulation": GasSensorFlowModulation,
        "gas_sensors_for_home_activity_monitoring": GasSensorsForHomeActivityMonitoring,
        "twin_gas_sensor_arrays": TwinGasSensorArrays,
    }


class CombinedEnoseDataset(_TorchDataset):
    """Combined dataset for cross-dataset pretraining.
    
    This class combines multiple UCI e-nose datasets into a single dataset,
    providing unified access for pretraining foundation models.
    
    Features:
    - Weighted sampling across datasets
    - Unified [C, T] data format via get_normalized_sample
    - Cross-dataset sensor model lookup
    - Channel masking support
    
    Example:
        >>> combined = CombinedEnoseDataset(
        ...     root='.cache',
        ...     datasets=['twin_gas_sensor_arrays', 'gas_sensors_for_home_activity_monitoring'],
        ...     download=True
        ... )
        >>> data, meta = combined.get_normalized_sample(0)
        >>> print(f"Dataset: {meta['dataset']}, Shape: {data.shape}")
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        datasets: Optional[List[str]] = None,
        *,
        download: bool = False,
        weights: Optional[List[float]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """Initialize combined dataset.
        
        Args:
            root: Root directory for all datasets
            datasets: List of dataset names to include. If None, includes all available.
            download: Whether to download missing datasets
            weights: Sampling weights per dataset (for weighted sampling)
            transform: Transform to apply to data
            target_transform: Transform to apply to targets
        """
        self.root = Path(root).expanduser()
        self.transform = transform
        self.target_transform = target_transform
        
        # Determine which datasets to load
        if datasets is None:
            datasets = list_datasets()
        self.dataset_names = datasets
        
        # Load all specified datasets
        self._datasets: Dict[str, BaseEnoseDataset] = {}
        self._sample_map: List[Tuple[str, int]] = []  # (dataset_name, local_index)
        
        for name in datasets:
            try:
                datasets_registry = _get_datasets_registry()
                cls = datasets_registry.get(name)
                if cls is None:
                    print(f"Warning: Dataset '{name}' not found in registry, skipping")
                    continue
                ds = cls(str(self.root), download=download)
                self._datasets[name] = ds
                
                # Build sample map
                for i in range(len(ds)):
                    self._sample_map.append((name, i))
                    
            except Exception as e:
                print(f"Warning: Failed to load dataset '{name}': {e}")
        
        if not self._datasets:
            raise RuntimeError("No datasets were successfully loaded")
        
        # Set up weights for weighted sampling
        if weights is not None:
            if len(weights) != len(self._datasets):
                raise ValueError(f"Weights length ({len(weights)}) must match loaded datasets ({len(self._datasets)})")
            self._weights = np.array(weights)
        else:
            # Default: weight by dataset size
            self._weights = np.array([len(ds) for ds in self._datasets.values()])
        self._weights = self._weights / self._weights.sum()
        
        # Build sensor model index for cross-dataset lookup
        self._sensor_model_index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        for ds_name, ds in self._datasets.items():
            for ch in ds.channels:
                self._sensor_model_index[ch.sensor_model].append((ds_name, ch.index))
    
    def __len__(self) -> int:
        return len(self._sample_map)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        ds_name, local_idx = self._sample_map[index]
        ds = self._datasets[ds_name]
        data, target = ds[local_idx]
        
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return data, target
    
    def get_normalized_sample(self, index: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Get sample in unified format from the combined dataset.
        
        Args:
            index: Global sample index
            
        Returns:
            Tuple of (data, metadata) in standardized format
        """
        ds_name, local_idx = self._sample_map[index]
        ds = self._datasets[ds_name]
        return ds.get_normalized_sample(local_idx)
    
    def get_sample(self, index: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Get raw (unnormalized) sample from the combined dataset.
        
        Used when backbone models handle normalization internally via BatchNorm.
        
        Args:
            index: Global sample index
            
        Returns:
            Tuple of (data, metadata) where data is raw/unnormalized
        """
        ds_name, local_idx = self._sample_map[index]
        ds = self._datasets[ds_name]
        return ds.get_sample(local_idx)
    
    def get_sample_with_mask(
        self,
        index: int,
        mask_channels: Optional[List[int]] = None,
        mask_ratio: float = 0.0,
        mask_value: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Get sample with channel masking from the combined dataset.
        
        Args:
            index: Global sample index
            mask_channels: List of channel indices to mask
            mask_ratio: Random mask ratio if mask_channels not specified
            mask_value: Value to use for masked channels
            
        Returns:
            Tuple of (masked_data, mask, metadata)
        """
        ds_name, local_idx = self._sample_map[index]
        ds = self._datasets[ds_name]
        return ds.get_sample_with_mask(local_idx, mask_channels, mask_ratio, mask_value)
    
    @property
    def datasets(self) -> Dict[str, BaseEnoseDataset]:
        """Get dictionary of loaded datasets."""
        return self._datasets
    
    @property
    def weights(self) -> np.ndarray:
        """Get sampling weights for each dataset."""
        return self._weights
    
    def get_dataset(self, name: str) -> BaseEnoseDataset:
        """Get a specific dataset by name."""
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' not loaded. Available: {list(self._datasets.keys())}")
        return self._datasets[name]
    
    def get_samples_from_dataset(self, name: str) -> List[int]:
        """Get global indices for all samples from a specific dataset.
        
        Args:
            name: Dataset name
            
        Returns:
            List of global indices
        """
        return [i for i, (ds_name, _) in enumerate(self._sample_map) if ds_name == name]
    
    def get_all_sensor_models(self) -> List[str]:
        """Get list of all unique sensor models across all datasets."""
        return list(self._sensor_model_index.keys())
    
    def get_channels_by_model_cross_dataset(self, model: str) -> List[Tuple[str, int]]:
        """Get all channels with a specific sensor model across all datasets.
        
        Args:
            model: Sensor model name (e.g., 'TGS2602')
            
        Returns:
            List of (dataset_name, channel_index) tuples
        """
        return self._sensor_model_index.get(model, [])
    
    def get_samples_by_sensor_model(self, model: str) -> List[int]:
        """Get global indices of samples from datasets that have a specific sensor model.
        
        Args:
            model: Sensor model name
            
        Returns:
            List of global sample indices
        """
        relevant_datasets = set(ds_name for ds_name, _ in self._sensor_model_index.get(model, []))
        return [i for i, (ds_name, _) in enumerate(self._sample_map) if ds_name in relevant_datasets]
    
    def summary(self) -> str:
        """Get a summary of the combined dataset."""
        lines = [
            f"CombinedEnoseDataset with {len(self._datasets)} datasets, {len(self)} total samples",
            "",
            "Datasets:",
        ]
        for name, ds in self._datasets.items():
            weight = self._weights[list(self._datasets.keys()).index(name)]
            lines.append(f"  - {name}: {len(ds)} samples ({weight:.1%})")
        
        lines.extend([
            "",
            f"Unique sensor models: {len(self._sensor_model_index)}",
        ])
        for model, locations in sorted(self._sensor_model_index.items()):
            ds_count = len(set(ds for ds, _ in locations))
            lines.append(f"  - {model}: {len(locations)} channels in {ds_count} dataset(s)")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"CombinedEnoseDataset(datasets={list(self._datasets.keys())}, samples={len(self)})"


class PretrainingDataset(_TorchDataset):
    """Dataset wrapper for pretraining with channel masking.
    
    This class wraps CombinedEnoseDataset or any BaseEnoseDataset and provides
    automatic channel masking for self-supervised pretraining.
    
    Example:
        >>> combined = CombinedEnoseDataset(root='.cache', datasets=['twin_gas_sensor_arrays'])
        >>> pretrain_ds = PretrainingDataset(combined, mask_ratio=0.25)
        >>> masked_data, mask, meta = pretrain_ds[0]
    """
    
    def __init__(
        self,
        dataset: Union[CombinedEnoseDataset, BaseEnoseDataset],
        mask_ratio: float = 0.25,
        mask_value: float = 0.0,
        fixed_mask_channels: Optional[List[int]] = None,
    ):
        """Initialize pretraining dataset.
        
        Args:
            dataset: Base dataset to wrap
            mask_ratio: Ratio of channels to mask (0-1)
            mask_value: Value to use for masked channels
            fixed_mask_channels: If set, always mask these channels instead of random
        """
        self.dataset = dataset
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.fixed_mask_channels = fixed_mask_channels
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Get masked sample for pretraining.
        
        Returns:
            Tuple of (masked_data, mask, metadata)
        """
        if hasattr(self.dataset, 'get_sample_with_mask'):
            return self.dataset.get_sample_with_mask(
                index,
                mask_channels=self.fixed_mask_channels,
                mask_ratio=self.mask_ratio,
                mask_value=self.mask_value,
            )
        else:
            raise TypeError("Dataset must support get_sample_with_mask method")
