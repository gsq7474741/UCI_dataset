from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import signal

try:
    from torch.utils.data import Dataset as _TorchDataset  # type: ignore
except Exception:  # pragma: no cover
    class _TorchDataset:  # type: ignore
        pass


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    path: Path
    target: Any
    meta: Dict[str, Any]


class DatasetWithTransforms(_TorchDataset):
    def __init__(
        self,
        root: Union[str, Path],
        *,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        root_path = Path(root).expanduser()
        self.root = root_path

        has_transforms = transforms is not None
        has_separate = transform is not None or target_transform is not None
        if has_transforms and has_separate:
            raise ValueError("Only transforms or transform/target_transform can be passed, not both")

        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform

    def __repr__(self) -> str:
        head = f"{self.__class__.__name__}({self.root})"
        extra = self.extra_repr()
        if not extra:
            return head
        lines = [head]
        for line in extra.split("\n"):
            lines.append(f"  {line}")
        return "\n".join(lines)

    def extra_repr(self) -> str:
        return ""


class BaseEnoseDataset(DatasetWithTransforms):
    name: str = ""

    def __init__(
        self,
        root: Union[str, Path],
        *,
        split: Optional[str] = None,
        download: bool = False,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)

        if not self.name:
            raise ValueError("Dataset must define class attribute 'name'")

        self.split = split
        from .schema._info import get_dataset_info  # local import to avoid cycles

        self.info = get_dataset_info(self.name)
        self._samples: List[SampleRecord] = []

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                f"Dataset not found or corrupted: {self.name}. "
                f"You can use download=True to download it."
            )

        self._samples = self._make_dataset()

    @property
    def dataset_dir(self) -> Path:
        return self.root / self.name

    @property
    def uci_id(self) -> Optional[int]:
        return self.info.uci_id

    @property
    def tasks(self) -> List[str]:
        return list(self.info.tasks)

    @property
    def sensor_type(self) -> str:
        return self.info.sensors.type

    @property
    def num_sensors(self) -> int:
        return self.info.sensors.count

    @property
    def sample_rate_hz(self) -> Optional[int]:
        if self.info.time_series is None:
            return None
        return self.info.time_series.sample_rate_hz

    @property
    def raw_dir(self) -> Path:
        return self.dataset_dir / "raw"

    @property
    def channels(self) -> Tuple:
        """Get all channel configurations for this dataset."""
        return self.info.sensors.channels

    @property
    def channel_models(self) -> List[str]:
        """Get list of sensor model names for all channels."""
        return [ch.sensor_model for ch in self.channels]

    @property
    def manufacturer(self) -> str:
        """Get sensor manufacturer."""
        return self.info.sensors.manufacturer

    def get_channel(self, index: int):
        """Get configuration for a specific channel.
        
        Args:
            index: Channel index (0-based)
            
        Returns:
            ChannelConfig for the specified channel
        """
        if not self.channels:
            raise ValueError(f"Dataset {self.name} has no channel configurations")
        if index < 0 or index >= len(self.channels):
            raise IndexError(f"Channel index {index} out of range [0, {len(self.channels)})")
        return self.channels[index]

    def get_channels_by_model(self, model: str) -> List[int]:
        """Get channel indices for a specific sensor model.
        
        Args:
            model: Sensor model name (e.g., 'TGS2602')
            
        Returns:
            List of channel indices with matching sensor model
        """
        return [ch.index for ch in self.channels if ch.sensor_model == model]

    def get_channels_by_target_gas(self, gas: str) -> List[int]:
        """Get channel indices that respond to a specific gas.
        
        Args:
            gas: Target gas name (e.g., 'Methane', 'CO')
            
        Returns:
            List of channel indices that respond to the specified gas
        """
        return [ch.index for ch in self.channels if gas in ch.target_gases]

    def get_channel_metadata_dict(self, index: int) -> Dict[str, Any]:
        """Get channel metadata as a dictionary.
        
        Args:
            index: Channel index
            
        Returns:
            Dictionary with channel metadata
        """
        ch = self.get_channel(index)
        return {
            "index": ch.index,
            "sensor_model": ch.sensor_model,
            "target_gases": list(ch.target_gases),
            "unit": ch.unit,
            "heater_voltage": ch.heater_voltage,
            "description": ch.description,
        }

    def get_all_channel_metadata(self) -> List[Dict[str, Any]]:
        """Get metadata for all channels as list of dictionaries."""
        return [self.get_channel_metadata_dict(i) for i in range(len(self.channels))]

    def get_normalized_sample(self, index: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Get a sample in standardized format for cross-dataset pretraining.
        
        This method provides a unified interface across all datasets, returning
        data in [C, T] format with comprehensive metadata.
        
        Args:
            index: Sample index
            
        Returns:
            Tuple of (data, metadata) where:
            - data: np.ndarray of shape [num_channels, time_steps]
            - metadata: Dict containing:
                - dataset: Dataset name
                - sample_id: Sample identifier
                - channel_models: List of sensor model names
                - channel_targets: List of target gas tuples per channel
                - sample_rate_hz: Sampling rate
                - manufacturer: Sensor manufacturer
                - target: Original target/label
                - record_meta: Original sample metadata
        """
        rec = self._samples[index]
        data, target = self._load_sample(rec)
        
        # Convert to numpy array in [C, T] format with float32 dtype
        if hasattr(data, 'values'):  # DataFrame
            # Select only numeric columns
            if hasattr(data, 'select_dtypes'):
                numeric_data = data.select_dtypes(include=[np.number])
                data_array = numeric_data.values.astype(np.float32)
            else:
                data_array = data.values
            if data_array.ndim == 2:
                # Assume [T, C] format from DataFrame, transpose to [C, T]
                data_array = data_array.T
        elif isinstance(data, np.ndarray):
            data_array = data
            # Ensure [C, T] format
            if data_array.ndim == 1:
                data_array = data_array.reshape(1, -1)
            elif data_array.ndim == 2:
                # Heuristic: if rows > cols, assume [T, C] and transpose to [C, T]
                # (typically num_channels << num_timesteps)
                if data_array.shape[0] > data_array.shape[1]:
                    data_array = data_array.T
        else:
            data_array = np.array(data)
            if data_array.ndim == 1:
                data_array = data_array.reshape(1, -1)
        
        # Ensure float32 dtype (handle object arrays from mixed types)
        if data_array.dtype == np.object_ or not np.issubdtype(data_array.dtype, np.floating):
            data_array = data_array.astype(np.float32)
        
        # Per-channel z-score normalization for cross-dataset compatibility
        # Each channel is normalized independently to align with downstream task processing
        # This preserves relative relationships between channels while handling different scales
        mean = np.nanmean(data_array, axis=1, keepdims=True)  # [C, 1]
        std = np.nanstd(data_array, axis=1, keepdims=True)    # [C, 1]
        std = np.where(std > 1e-6, std, 1.0)  # Avoid division by zero
        # Handle channels with all NaN (mean would be NaN)
        mean = np.nan_to_num(mean, nan=0.0)
        data_array = (data_array - mean) / std
        
        # Final cleanup for FP16 compatibility
        data_array = np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        metadata = {
            "dataset": self.name,
            "sample_id": rec.sample_id,
            "channel_models": self.channel_models,
            "channel_targets": [list(ch.target_gases) for ch in self.channels],
            "sample_rate_hz": self.sample_rate_hz,
            "manufacturer": self.manufacturer,
            "target": target,
            "record_meta": dict(rec.meta),
        }
        
        return data_array, metadata
    
    def get_sample(self, index: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Get a raw (unnormalized) sample for models that handle normalization internally.
        
        Same as get_normalized_sample but WITHOUT per-channel z-score normalization.
        Used when backbone models handle normalization internally via BatchNorm.
        
        Args:
            index: Sample index
            
        Returns:
            Tuple of (data, metadata) where:
            - data: np.ndarray of shape [num_channels, time_steps] (raw, unnormalized)
            - metadata: Dict with sample metadata
        """
        rec = self._samples[index]
        data, target = self._load_sample(rec)
        
        # Convert to numpy array in [C, T] format with float32 dtype
        if hasattr(data, 'values'):  # DataFrame
            if hasattr(data, 'select_dtypes'):
                numeric_data = data.select_dtypes(include=[np.number])
                data_array = numeric_data.values.astype(np.float32)
            else:
                data_array = data.values
            if data_array.ndim == 2:
                data_array = data_array.T  # [T, C] -> [C, T]
        elif isinstance(data, np.ndarray):
            data_array = data
            if data_array.ndim == 1:
                data_array = data_array.reshape(1, -1)
            elif data_array.ndim == 2:
                if data_array.shape[0] > data_array.shape[1]:
                    data_array = data_array.T
        else:
            data_array = np.array(data)
            if data_array.ndim == 1:
                data_array = data_array.reshape(1, -1)
        
        # Ensure float32 dtype
        if data_array.dtype == np.object_ or not np.issubdtype(data_array.dtype, np.floating):
            data_array = data_array.astype(np.float32)
        
        # NO normalization - backbone handles it internally via BatchNorm
        # Just cleanup for FP16 compatibility
        data_array = np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        metadata = {
            "dataset": self.name,
            "sample_id": rec.sample_id,
            "channel_models": self.channel_models,
            "channel_targets": [list(ch.target_gases) for ch in self.channels],
            "sample_rate_hz": self.sample_rate_hz,
            "manufacturer": self.manufacturer,
            "target": target,
            "record_meta": dict(rec.meta),
        }
        
        return data_array, metadata

    def get_sample_with_mask(
        self,
        index: int,
        mask_channels: Optional[List[int]] = None,
        mask_ratio: float = 0.0,
        mask_value: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Get a sample with optional channel masking for pretraining.
        
        Args:
            index: Sample index
            mask_channels: List of channel indices to mask (overrides mask_ratio)
            mask_ratio: Random mask ratio if mask_channels not specified
            mask_value: Value to use for masked channels
            
        Returns:
            Tuple of (data, mask, metadata) where:
            - data: np.ndarray [C, T] with masked channels set to mask_value
            - mask: np.ndarray [C] boolean mask (True = available, False = masked)
            - metadata: Dict with sample metadata
        """
        data_array, metadata = self.get_normalized_sample(index)
        num_channels = data_array.shape[0]
        
        # Create mask
        mask = np.ones(num_channels, dtype=bool)
        
        if mask_channels is not None:
            for ch in mask_channels:
                if 0 <= ch < num_channels:
                    mask[ch] = False
        elif mask_ratio > 0:
            num_mask = int(num_channels * mask_ratio)
            if num_mask > 0:
                mask_indices = np.random.choice(num_channels, num_mask, replace=False)
                mask[mask_indices] = False
        
        # Apply mask
        masked_data = data_array.copy()
        masked_data[~mask] = mask_value
        
        metadata["mask"] = mask.tolist()
        metadata["masked_channels"] = np.where(~mask)[0].tolist()
        
        return masked_data, mask, metadata

    @staticmethod
    def resample_temporal(
        data: np.ndarray,
        src_rate: int,
        dst_rate: int,
        method: str = "polyphase",
    ) -> np.ndarray:
        """Resample time series data from source to destination sampling rate.
        
        Implements temporal resampling as part of standardization transformation F:
            T̂_i = floor(T_i × (f_std / f_i))
        
        Args:
            data: Input array of shape [C, T] (channels × time steps)
            src_rate: Source sampling rate in Hz
            dst_rate: Destination sampling rate in Hz
            method: Resampling method ('polyphase', 'fft', 'linear')
                - 'polyphase': scipy.signal.resample_poly (best quality, default)
                - 'fft': scipy.signal.resample (frequency domain)
                - 'linear': numpy linear interpolation (fastest)
        
        Returns:
            Resampled array of shape [C, T̂] where T̂ = floor(T × dst_rate / src_rate)
        """
        if src_rate == dst_rate:
            return data
        
        if src_rate <= 0 or dst_rate <= 0:
            return data
        
        C, T = data.shape
        
        # Calculate target length
        T_new = int(T * dst_rate / src_rate)
        if T_new <= 0:
            T_new = 1
        
        if method == "polyphase":
            # Polyphase resampling (best quality for integer ratios)
            from math import gcd
            g = gcd(src_rate, dst_rate)
            up = dst_rate // g
            down = src_rate // g
            # resample_poly applies along axis=-1 by default
            resampled = signal.resample_poly(data, up, down, axis=1)
        elif method == "fft":
            # FFT-based resampling
            resampled = signal.resample(data, T_new, axis=1)
        elif method == "linear":
            # Linear interpolation (fastest)
            x_old = np.linspace(0, 1, T)
            x_new = np.linspace(0, 1, T_new)
            resampled = np.zeros((C, T_new), dtype=data.dtype)
            for c in range(C):
                resampled[c] = np.interp(x_new, x_old, data[c])
        else:
            raise ValueError(f"Unknown resampling method: {method}")
        
        return resampled.astype(data.dtype)

    def get_standardized_sample(
        self,
        index: int,
        target_rate: Optional[int] = None,
        resample_method: str = "polyphase",
        normalize: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Get a sample standardized to global reference space Ω_global.
        
        This method performs the standardization transformation F(d_i | Ω_global):
        1. Temporal resampling: X_i → X̂_i at unified rate f_std
        2. Per-channel z-score normalization (optional)
        3. Global sensor index mapping in metadata
        
        Args:
            index: Sample index
            target_rate: Target sampling rate in Hz. If None, uses F_STD from _global.py
            resample_method: Resampling algorithm ('polyphase', 'fft', 'linear')
            normalize: Whether to apply per-channel z-score normalization
        
        Returns:
            Tuple of (data, metadata) where:
            - data: np.ndarray [C, T̂] resampled to target_rate
            - metadata: Dict with standardization info including:
                - original_rate: Original sampling rate
                - target_rate: Target sampling rate after resampling
                - original_length: Original time steps T_i
                - resampled_length: New time steps T̂_i
                - global_sensor_ids: Global sensor indices from S_all
        """
        from ._global import F_STD, get_sensor_id, get_global_channel_mapping
        
        if target_rate is None:
            target_rate = F_STD
        
        # Get raw sample
        rec = self._samples[index]
        data, target = self._load_sample(rec)
        
        # Convert to [C, T] format
        if hasattr(data, 'values'):  # DataFrame
            if hasattr(data, 'select_dtypes'):
                numeric_data = data.select_dtypes(include=[np.number])
                data_array = numeric_data.values.astype(np.float32)
            else:
                data_array = data.values.astype(np.float32)
            if data_array.ndim == 2:
                data_array = data_array.T  # [T, C] -> [C, T]
        elif isinstance(data, np.ndarray):
            data_array = data.astype(np.float32)
            if data_array.ndim == 1:
                data_array = data_array.reshape(1, -1)
            elif data_array.ndim == 2 and data_array.shape[0] > data_array.shape[1]:
                data_array = data_array.T
        else:
            data_array = np.array(data, dtype=np.float32)
            if data_array.ndim == 1:
                data_array = data_array.reshape(1, -1)
        
        original_length = data_array.shape[1]
        original_rate = self.sample_rate_hz
        
        # Step 1: Temporal resampling
        if original_rate is not None and original_rate != target_rate:
            data_array = self.resample_temporal(
                data_array, original_rate, target_rate, method=resample_method
            )
        
        resampled_length = data_array.shape[1]
        
        # Step 2: Per-channel normalization (optional)
        if normalize:
            mean = np.nanmean(data_array, axis=1, keepdims=True)
            std = np.nanstd(data_array, axis=1, keepdims=True)
            std = np.where(std > 1e-6, std, 1.0)
            mean = np.nan_to_num(mean, nan=0.0)
            data_array = (data_array - mean) / std
        
        # Cleanup for numerical stability
        data_array = np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Step 3: Build metadata with global mappings
        # Get global sensor indices
        global_mapping = get_global_channel_mapping(self.name)
        if not global_mapping:
            # Fallback: map each channel model to global ID
            global_mapping = [get_sensor_id(m) for m in self.channel_models]
        
        metadata = {
            "dataset": self.name,
            "sample_id": rec.sample_id,
            # Original info
            "channel_models": self.channel_models,
            "channel_targets": [list(ch.target_gases) for ch in self.channels],
            "manufacturer": self.manufacturer,
            "target": target,
            "record_meta": dict(rec.meta),
            # Standardization info
            "original_rate": original_rate,
            "target_rate": target_rate,
            "original_length": original_length,
            "resampled_length": resampled_length,
            "global_sensor_ids": global_mapping,
            "normalized": normalize,
        }
        
        return data_array, metadata

    def download(self) -> None:
        raise NotImplementedError

    def _check_exists(self) -> bool:
        return self.dataset_dir.exists()

    def _make_dataset(self) -> List[SampleRecord]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        rec = self._samples[index]
        data, target = self._load_sample(rec)

        if self.transforms is not None:
            data, target = self.transforms(data, target)
        else:
            if self.transform is not None:
                data = self.transform(data)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return data, target

    def _load_sample(self, record: SampleRecord) -> Tuple[Any, Any]:
        raise NotImplementedError

    def get_record(self, index: int) -> SampleRecord:
        return self._samples[index]

    def extra_repr(self) -> str:
        parts = []
        if self.split is not None:
            parts.append(f"split={self.split}")
        parts.append(f"samples={len(self)}")
        return "\n".join(parts)
