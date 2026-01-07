"""Unified E-nose Dataset for multi-dataset loading and multi-task learning.

This module provides a unified interface for loading one or more e-nose datasets
with configurable task types, data splits, and preprocessing options.

Example:
    >>> from enose_uci_dataset.datasets import UnifiedEnoseDataset, TaskType, SplitType
    >>> 
    >>> # 单数据集
    >>> ds = UnifiedEnoseDataset(
    ...     root="./data",
    ...     datasets="twin_gas_sensor_arrays",
    ...     task=TaskType.GAS_CLASSIFICATION,
    ...     split=SplitType.TRAIN,
    ... )
    >>> 
    >>> # 多数据集
    >>> ds = UnifiedEnoseDataset(
    ...     root="./data",
    ...     datasets=["twin_gas_sensor_arrays", "g919_55"],
    ...     task=TaskType.SELF_SUPERVISED,
    ...     split=SplitType.TRAIN,
    ... )
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from torch.utils.data import Dataset as _TorchDataset
except Exception:
    class _TorchDataset:
        pass


# =============================================================================
# Type Definitions and Enums
# =============================================================================

class TaskType(str, Enum):
    """Task type enumeration - determines what labels are returned."""
    
    # Classification tasks
    GAS_CLASSIFICATION = "gas_classification"
    """Gas type classification (CO, Ethanol, Methane, etc.)"""
    
    ODOR_CLASSIFICATION = "odor_classification"
    """Odor/ingredient classification (for SmellNet, G919)"""
    
    ACTIVITY_RECOGNITION = "activity_recognition"
    """Activity recognition (for home monitoring dataset)"""
    
    # Regression tasks
    CONCENTRATION_REGRESSION = "concentration_regression"
    """Gas concentration regression (ppm)"""
    
    # Domain adaptation tasks
    DRIFT_COMPENSATION = "drift_compensation"
    """Sensor drift compensation (returns data + domain_id)"""
    
    # Unsupervised tasks
    ANOMALY_DETECTION = "anomaly_detection"
    """Anomaly detection (normal=0, anomaly=1)"""
    
    SELF_SUPERVISED = "self_supervised"
    """Self-supervised pretraining (no labels, returns data only)"""
    
    # Multi-task
    MULTI_TASK = "multi_task"
    """Multi-task learning (returns Dict[task_name, label])"""
    
    # Raw - returns original dataset labels unchanged
    RAW = "raw"
    """Return original dataset labels without conversion"""


class SplitType(str, Enum):
    """Data split type enumeration."""
    
    TRAIN = "train"
    """Training set"""
    
    VAL = "val"
    """Validation set"""
    
    TEST = "test"
    """Test set"""
    
    ALL = "all"
    """All data (no split)"""


class NormalizeType(str, Enum):
    """Normalization method enumeration."""
    
    ZSCORE = "zscore"
    """Per-sample z-score normalization: (x - mean) / std"""
    
    MINMAX = "minmax"
    """Per-sample min-max normalization: (x - min) / (max - min)"""
    
    GLOBAL_ZSCORE = "global_zscore"
    """Global z-score using dataset statistics"""
    
    NONE = "none"
    """No normalization"""


class ChannelAlignMode(str, Enum):
    """Channel alignment mode for multi-dataset loading."""
    
    NONE = "none"
    """No alignment - keep original channels"""
    
    PAD = "pad"
    """Pad to max_channels with zeros"""
    
    GLOBAL = "global"
    """Align to global sensor space (M_TOTAL dimensions)"""


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class SplitConfig:
    """Configuration for data splitting.
    
    Attributes:
        train_ratio: Ratio for training set (default 0.8)
        val_ratio: Ratio for validation set (default 0.1)
        test_ratio: Ratio for test set (default 0.1)
        seed: Random seed for reproducible splits
        stratify: Whether to stratify by label
    """
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    stratify: bool = True
    
    def __post_init__(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


@dataclass
class WindowConfig:
    """Configuration for sliding window segmentation.
    
    Attributes:
        size: Window size in time steps
        stride: Window stride in time steps (default = size, no overlap)
        min_length: Minimum sample length to include
    """
    size: int
    stride: Optional[int] = None
    min_length: Optional[int] = None
    
    def __post_init__(self):
        if self.stride is None:
            self.stride = self.size
        if self.min_length is None:
            self.min_length = self.size


@dataclass
class UnifiedDatasetConfig:
    """Complete configuration for UnifiedEnoseDataset.
    
    This dataclass defines all configurable options for the unified dataset.
    """
    # Dataset selection
    datasets: Union[str, List[str]] = field(default_factory=lambda: "all")
    """Dataset name(s) to load. Can be a single name, list, or "all"."""
    
    root: str = ".cache"
    """Root directory for dataset storage."""
    
    download: bool = False
    """Whether to download missing datasets."""
    
    # Task configuration
    task: TaskType = TaskType.RAW
    """Task type determining what labels are returned."""
    
    use_global_labels: bool = True
    """Whether to convert labels to global label space."""
    
    # Split configuration
    split: SplitType = SplitType.ALL
    """Which split to load."""
    
    split_config: SplitConfig = field(default_factory=SplitConfig)
    """Configuration for automatic split generation."""
    
    # Temporal processing
    target_sample_rate: Optional[float] = None
    """Target sample rate for resampling (Hz). None = no resampling."""
    
    window: Optional[WindowConfig] = None
    """Sliding window configuration. None = no windowing."""
    
    # Normalization
    normalize: NormalizeType = NormalizeType.ZSCORE
    """Normalization method."""
    
    # Channel handling
    channel_align: ChannelAlignMode = ChannelAlignMode.NONE
    """How to align channels across datasets."""
    
    max_channels: int = 16
    """Maximum number of channels (for PAD mode)."""
    
    # Sampling
    balance_datasets: bool = False
    """Whether to balance samples across datasets."""
    
    balance_classes: bool = False
    """Whether to balance samples across classes."""
    
    # Transforms
    transform: Optional[Callable] = None
    """Transform to apply to data."""
    
    target_transform: Optional[Callable] = None
    """Transform to apply to labels."""


# =============================================================================
# Dataset Registry
# =============================================================================

def _get_dataset_registry() -> Dict[str, type]:
    """Lazy import dataset classes to avoid circular imports."""
    from .gas_sensor_dynamic import GasSensorDynamic
    from .gas_sensor_flow_modulation import GasSensorFlowModulation
    from .gas_sensor_low_concentration import GasSensorLowConcentration
    from .gas_sensor_temperature_modulation import GasSensorTemperatureModulation
    from .gas_sensor_turbulent import GasSensorTurbulent
    from .gas_sensors_for_home_activity_monitoring import GasSensorsForHomeActivityMonitoring
    from .twin_gas_sensor_arrays import TwinGasSensorArrays
    from .smellnet import SmellNet
    from .g919_55 import G919SensorDataset
    
    return {
        "gas_sensor_array_exposed_to_turbulent_gas_mixtures": GasSensorTurbulent,
        "gas_sensor_array_low_concentration": GasSensorLowConcentration,
        "gas_sensor_array_temperature_modulation": GasSensorTemperatureModulation,
        "gas_sensor_array_under_dynamic_gas_mixtures": GasSensorDynamic,
        "gas_sensor_array_under_flow_modulation": GasSensorFlowModulation,
        "gas_sensors_for_home_activity_monitoring": GasSensorsForHomeActivityMonitoring,
        "twin_gas_sensor_arrays": TwinGasSensorArrays,
        "smellnet_pure": SmellNet,
        "smellnet_mixture": SmellNet,
        "g919_55": G919SensorDataset,
    }


# Task compatibility matrix
TASK_DATASET_COMPATIBILITY: Dict[TaskType, List[str]] = {
    TaskType.GAS_CLASSIFICATION: [
        "twin_gas_sensor_arrays",
        "gas_sensor_array_exposed_to_turbulent_gas_mixtures",
        "gas_sensor_array_low_concentration",
        "gas_sensor_array_under_flow_modulation",
    ],
    TaskType.ODOR_CLASSIFICATION: [
        "g919_55",
        "smellnet_pure",
        "smellnet_mixture",
    ],
    TaskType.ACTIVITY_RECOGNITION: [
        "gas_sensors_for_home_activity_monitoring",
    ],
    TaskType.CONCENTRATION_REGRESSION: [
        "twin_gas_sensor_arrays",
        "gas_sensor_array_under_dynamic_gas_mixtures",
        "gas_sensor_array_exposed_to_turbulent_gas_mixtures",
        "gas_sensor_array_temperature_modulation",
    ],
    TaskType.DRIFT_COMPENSATION: [],  # No time-series datasets for drift compensation
    TaskType.ANOMALY_DETECTION: [],  # All datasets supported
    TaskType.SELF_SUPERVISED: [],    # All datasets supported
    TaskType.MULTI_TASK: [],         # All datasets supported
    TaskType.RAW: [],                # All datasets supported
}


# =============================================================================
# Main Dataset Class
# =============================================================================

class UnifiedEnoseDataset(_TorchDataset):
    """Unified E-nose Dataset supporting multi-dataset loading and multi-task learning.
    
    This class provides a single interface for loading one or more e-nose datasets
    with configurable task types, automatic data splitting, and preprocessing.
    
    Features:
        - Load single or multiple datasets
        - Automatic train/val/test splitting (8:1:1 by default)
        - Multiple task types (classification, regression, self-supervised, etc.)
        - Global label space conversion
        - Configurable normalization
        - Sliding window segmentation
        - Channel alignment across datasets
    
    Example:
        >>> # Single dataset, gas classification
        >>> ds = UnifiedEnoseDataset(
        ...     root="./data",
        ...     datasets="twin_gas_sensor_arrays",
        ...     task=TaskType.GAS_CLASSIFICATION,
        ...     split=SplitType.TRAIN,
        ...     download=True,
        ... )
        >>> data, label = ds[0]
        
        >>> # Multiple datasets, self-supervised pretraining
        >>> ds = UnifiedEnoseDataset(
        ...     root="./data",
        ...     datasets=["twin_gas_sensor_arrays", "g919_55"],
        ...     task=TaskType.SELF_SUPERVISED,
        ...     split=SplitType.TRAIN,
        ...     normalize=NormalizeType.ZSCORE,
        ... )
        >>> data, _ = ds[0]  # label is None for self-supervised
    
    Attributes:
        config: UnifiedDatasetConfig with all settings
        datasets: Dict of loaded dataset instances
        samples: List of (dataset_name, local_idx, split) tuples
    """
    
    def __init__(
        self,
        root: Union[str, Path] = ".cache",
        datasets: Union[str, List[str], None] = None,
        *,
        task: Union[TaskType, str] = TaskType.RAW,
        split: Union[SplitType, str] = SplitType.ALL,
        download: bool = False,
        # Split config
        split_config: Optional[SplitConfig] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        split_seed: int = 42,
        # Preprocessing
        normalize: Union[NormalizeType, str] = NormalizeType.ZSCORE,
        target_sample_rate: Optional[float] = None,
        window_size: Optional[int] = None,
        window_stride: Optional[int] = None,
        # Labels
        use_global_labels: bool = True,
        # Channel handling
        channel_align: Union[ChannelAlignMode, str] = ChannelAlignMode.NONE,
        max_channels: int = 16,
        # Sampling
        balance_datasets: bool = False,
        balance_classes: bool = False,
        # Transforms
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        # Special dataset args
        local_paths: Optional[Dict[str, str]] = None,
    ):
        """Initialize unified dataset.
        
        Args:
            root: Root directory for datasets
            datasets: Dataset name(s). Can be:
                - str: Single dataset name
                - List[str]: Multiple dataset names
                - None or "all": All available datasets
            task: Task type (determines label format)
            split: Data split to load
            download: Whether to download missing datasets
            split_config: Custom split configuration
            train_ratio: Training set ratio (if split_config not provided)
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            split_seed: Random seed for splits
            normalize: Normalization method
            target_sample_rate: Target sample rate for resampling
            window_size: Sliding window size (None = no windowing)
            window_stride: Sliding window stride
            use_global_labels: Convert to global label space
            channel_align: Channel alignment mode
            max_channels: Max channels for PAD mode
            balance_datasets: Balance samples across datasets
            balance_classes: Balance samples across classes
            transform: Data transform
            target_transform: Label transform
            local_paths: Dict of dataset_name -> local_path for local-only datasets
        """
        self.root = Path(root).expanduser()
        self.local_paths = local_paths or {}
        
        # Convert string enums
        if isinstance(task, str):
            task = TaskType(task)
        if isinstance(split, str):
            split = SplitType(split)
        if isinstance(normalize, str):
            normalize = NormalizeType(normalize)
        if isinstance(channel_align, str):
            channel_align = ChannelAlignMode(channel_align)
        
        self.task = task
        self.split = split
        self.normalize = normalize
        self.channel_align = channel_align
        self.max_channels = max_channels
        self.use_global_labels = use_global_labels
        self.balance_datasets = balance_datasets
        self.balance_classes = balance_classes
        self.transform = transform
        self.target_transform = target_transform
        self.target_sample_rate = target_sample_rate
        
        # Split config
        if split_config is not None:
            self.split_config = split_config
        else:
            self.split_config = SplitConfig(
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=split_seed,
            )
        
        # Window config
        if window_size is not None:
            self.window_config = WindowConfig(size=window_size, stride=window_stride)
        else:
            self.window_config = None
        
        # Resolve dataset names
        self._dataset_names = self._resolve_dataset_names(datasets)
        
        # Validate task compatibility
        self._validate_task_compatibility()
        
        # Load datasets
        self._datasets: Dict[str, Any] = {}
        self._load_datasets(download)
        
        # Build sample index with splits
        self._samples: List[Tuple[str, int]] = []
        self._sample_labels: List[Any] = []
        self._build_sample_index()
        
        # Apply balancing if requested
        if balance_datasets or balance_classes:
            self._apply_balancing()
    
    def _resolve_dataset_names(self, datasets: Union[str, List[str], None]) -> List[str]:
        """Resolve dataset specification to list of names."""
        registry = _get_dataset_registry()
        all_names = list(registry.keys())
        
        if datasets is None or datasets == "all":
            return all_names
        elif isinstance(datasets, str):
            return [datasets]
        else:
            return list(datasets)
    
    def _validate_task_compatibility(self):
        """Validate that datasets are compatible with the task."""
        if self.task in (TaskType.ANOMALY_DETECTION, TaskType.SELF_SUPERVISED, 
                         TaskType.MULTI_TASK, TaskType.RAW):
            return  # These tasks work with any dataset
        
        compatible = TASK_DATASET_COMPATIBILITY.get(self.task, [])
        if compatible:
            for name in self._dataset_names:
                if name not in compatible:
                    raise ValueError(
                        f"Dataset '{name}' is not compatible with task {self.task}. "
                        f"Compatible datasets: {compatible}"
                    )
    
    def _load_datasets(self, download: bool):
        """Load all specified datasets."""
        registry = _get_dataset_registry()
        
        for name in self._dataset_names:
            if name not in registry:
                raise ValueError(f"Unknown dataset: {name}")
            
            cls = registry[name]
            
            try:
                # Handle special datasets
                if name == "g919_55":
                    local_path = self.local_paths.get(name)
                    if local_path is None:
                        raise ValueError(
                            f"Dataset 'g919_55' requires local_path. "
                            f"Pass local_paths={{'g919_55': '/path/to/G919-55'}}"
                        )
                    ds = cls(
                        root=str(self.root),
                        local_path=local_path,
                        split=None,  # We handle splits ourselves
                    )
                elif name in ("smellnet_pure", "smellnet_mixture"):
                    ds = cls(
                        root=str(self.root),
                        subset="pure" if name == "smellnet_pure" else "mixture",
                    )
                else:
                    ds = cls(
                        root=str(self.root),
                        download=download,
                        split=None,
                    )
                
                self._datasets[name] = ds
                
            except Exception as e:
                raise RuntimeError(f"Failed to load dataset '{name}': {e}") from e
    
    def _build_sample_index(self):
        """Build sample index with train/val/test splits."""
        all_samples: List[Tuple[str, int, Any]] = []  # (ds_name, local_idx, label)
        
        for ds_name, ds in self._datasets.items():
            for local_idx in range(len(ds)):
                label = self._get_label(ds_name, ds, local_idx)
                all_samples.append((ds_name, local_idx, label))
        
        # Generate or use existing splits
        split_indices = self._get_split_indices(all_samples)
        
        # Filter by requested split
        if self.split == SplitType.ALL:
            indices = list(range(len(all_samples)))
        else:
            indices = split_indices[self.split.value]
        
        # Build final sample list
        self._samples = [(all_samples[i][0], all_samples[i][1]) for i in indices]
        self._sample_labels = [all_samples[i][2] for i in indices]
    
    def _get_split_indices(
        self, 
        all_samples: List[Tuple[str, int, Any]]
    ) -> Dict[str, List[int]]:
        """Generate train/val/test split indices.
        
        Logic:
        1. If dataset has native splits, use them
        2. Otherwise, generate 8:1:1 split
        3. If only train/test exist, split val from train
        """
        n = len(all_samples)
        if n == 0:
            return {"train": [], "val": [], "test": []}
        
        # Check if any dataset has native splits
        has_native_splits = any(
            hasattr(ds, 'get_split_indices') or hasattr(ds, 'split')
            for ds in self._datasets.values()
        )
        
        # For now, generate splits based on config
        rng = np.random.RandomState(self.split_config.seed)
        indices = np.arange(n)
        
        # Check if stratification is possible
        first_label = all_samples[0][2] if all_samples else None
        if self.split_config.stratify and first_label is not None:
            # Stratified split
            labels = [self._get_stratify_key(lbl) for _, _, lbl in all_samples]
            indices = self._stratified_split(indices, labels, rng)
        else:
            rng.shuffle(indices)
        
        # Calculate split sizes
        n_train = int(n * self.split_config.train_ratio)
        n_val = int(n * self.split_config.val_ratio)
        
        train_idx = indices[:n_train].tolist()
        val_idx = indices[n_train:n_train + n_val].tolist()
        test_idx = indices[n_train + n_val:].tolist()
        
        return {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }
    
    def _stratified_split(
        self, 
        indices: np.ndarray, 
        labels: List[Any],
        rng: np.random.RandomState
    ) -> np.ndarray:
        """Perform stratified shuffle."""
        from collections import defaultdict
        
        # Group by label
        label_indices = defaultdict(list)
        for idx in indices:
            label_indices[labels[idx]].append(idx)
        
        # Shuffle within each group and interleave
        result = []
        for lbl_indices in label_indices.values():
            rng.shuffle(lbl_indices)
            result.extend(lbl_indices)
        
        return np.array(result)
    
    def _get_stratify_key(self, label: Any) -> Any:
        """Get key for stratification from label."""
        if isinstance(label, dict):
            # Multi-task: use first task's label
            return list(label.values())[0] if label else 0
        elif isinstance(label, (list, tuple)):
            return label[0] if label else 0
        else:
            return label
    
    def _get_label(self, ds_name: str, ds: Any, local_idx: int) -> Any:
        """Get label for a sample based on task type."""
        # Get raw sample
        if hasattr(ds, 'get_normalized_sample'):
            _, meta = ds.get_normalized_sample(local_idx)
            raw_target = meta.get('target', {})
        else:
            _, raw_target = ds[local_idx]
        
        # Convert based on task
        if self.task == TaskType.RAW:
            return raw_target
        
        elif self.task == TaskType.SELF_SUPERVISED:
            return None
        
        elif self.task == TaskType.GAS_CLASSIFICATION:
            return self._convert_gas_label(ds_name, raw_target)
        
        elif self.task == TaskType.ODOR_CLASSIFICATION:
            return self._convert_odor_label(ds_name, raw_target)
        
        elif self.task == TaskType.CONCENTRATION_REGRESSION:
            return self._convert_concentration(raw_target)
        
        elif self.task == TaskType.ACTIVITY_RECOGNITION:
            return self._convert_activity_label(raw_target)
        
        elif self.task == TaskType.DRIFT_COMPENSATION:
            return self._convert_drift_label(raw_target)
        
        elif self.task == TaskType.ANOMALY_DETECTION:
            return 0  # Default: normal (anomaly labels need to be set externally)
        
        elif self.task == TaskType.MULTI_TASK:
            return {
                "gas": self._convert_gas_label(ds_name, raw_target),
                "concentration": self._convert_concentration(raw_target),
            }
        
        return raw_target
    
    def _convert_gas_label(self, ds_name: str, target: Any) -> int:
        """Convert raw target to gas classification label."""
        if isinstance(target, dict):
            # Different datasets use different key names
            # turbulent: 'second_gas' (0=Methane, 1=CO)
            # twin: 'gas'
            # drift: 'gas'
            local_label = target.get('gas', target.get('second_gas', 0))
        else:
            local_label = target
        
        if self.use_global_labels:
            from ._global import GAS_LABEL_MAPPINGS, get_global_label
            return get_global_label(ds_name, local_label, 'gas')
        else:
            return local_label
    
    def _convert_odor_label(self, ds_name: str, target: Any) -> int:
        """Convert raw target to odor classification label."""
        if isinstance(target, dict):
            # G919 uses 'odor_idx', SmellNet uses 'label'
            return target.get('odor_idx', target.get('odor', target.get('label', 0)))
        return target
    
    def _convert_concentration(self, target: Any) -> float:
        """Convert raw target to concentration value."""
        if isinstance(target, dict):
            return float(target.get('ppm', target.get('concentration', 0.0)))
        return float(target) if target is not None else 0.0
    
    def _convert_activity_label(self, target: Any) -> int:
        """Convert raw target to activity label."""
        if isinstance(target, dict):
            return target.get('activity', 0)
        return target
    
    def _convert_drift_label(self, target: Any) -> Tuple[int, int]:
        """Convert raw target to (gas_label, batch_id) for drift compensation."""
        if isinstance(target, dict):
            return (target.get('gas', 0), target.get('batch', 0))
        return (target, 0)
    
    def _apply_balancing(self):
        """Apply dataset or class balancing."""
        if not self._samples:
            return
        
        if self.balance_datasets:
            # Upsample smaller datasets
            from collections import Counter
            ds_counts = Counter(s[0] for s in self._samples)
            max_count = max(ds_counts.values())
            
            new_samples = []
            new_labels = []
            rng = np.random.RandomState(self.split_config.seed)
            
            for ds_name in ds_counts:
                ds_indices = [i for i, s in enumerate(self._samples) if s[0] == ds_name]
                n_repeat = max_count // len(ds_indices)
                n_extra = max_count % len(ds_indices)
                
                for _ in range(n_repeat):
                    for idx in ds_indices:
                        new_samples.append(self._samples[idx])
                        new_labels.append(self._sample_labels[idx])
                
                extra_indices = rng.choice(ds_indices, n_extra, replace=False)
                for idx in extra_indices:
                    new_samples.append(self._samples[idx])
                    new_labels.append(self._sample_labels[idx])
            
            self._samples = new_samples
            self._sample_labels = new_labels
    
    def __len__(self) -> int:
        return len(self._samples)
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, Any]:
        """Get a sample.
        
        Returns:
            Tuple of (data, label) where:
            - data: np.ndarray of shape [C, T] (or [C, T, W] if windowed)
            - label: Depends on task type
        """
        ds_name, local_idx = self._samples[index]
        label = self._sample_labels[index]
        
        # Load data
        ds = self._datasets[ds_name]
        if hasattr(ds, 'get_normalized_sample'):
            data, meta = ds.get_normalized_sample(local_idx)
        else:
            data, _ = ds[local_idx]
            if hasattr(data, 'values'):
                data = data.values.T.astype(np.float32)
        
        # Apply normalization
        data = self._apply_normalization(data)
        
        # Apply resampling if needed
        if self.target_sample_rate is not None:
            orig_rate = ds.sample_rate_hz or 1.0
            if orig_rate != self.target_sample_rate:
                data = self._resample(data, orig_rate, self.target_sample_rate)
        
        # Apply windowing if configured
        if self.window_config is not None:
            data = self._apply_windowing(data)
        
        # Apply channel alignment
        data = self._apply_channel_alignment(data, ds_name)
        
        # Apply transforms
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None and label is not None:
            label = self.target_transform(label)
        
        return data, label
    
    def _apply_normalization(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization to data."""
        if self.normalize == NormalizeType.NONE:
            return data
        
        elif self.normalize == NormalizeType.ZSCORE:
            mean = data.mean()
            std = data.std()
            if std > 1e-8:
                return (data - mean) / std
            return data - mean
        
        elif self.normalize == NormalizeType.MINMAX:
            min_val = data.min()
            max_val = data.max()
            if max_val - min_val > 1e-8:
                return (data - min_val) / (max_val - min_val)
            return data - min_val
        
        return data
    
    def _resample(
        self, 
        data: np.ndarray, 
        orig_rate: float, 
        target_rate: float
    ) -> np.ndarray:
        """Resample time series data."""
        from scipy import signal
        
        if abs(orig_rate - target_rate) < 1e-6:
            return data
        
        # Calculate resampling ratio
        ratio = target_rate / orig_rate
        new_length = int(data.shape[1] * ratio)
        
        resampled = signal.resample(data, new_length, axis=1)
        return resampled.astype(np.float32)
    
    def _apply_windowing(self, data: np.ndarray) -> np.ndarray:
        """Apply sliding window segmentation."""
        if self.window_config is None:
            return data
        
        T = data.shape[1]
        size = self.window_config.size
        stride = self.window_config.stride
        
        if T < size:
            # Pad if too short
            pad_width = size - T
            data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')
            return data
        
        # Extract windows
        n_windows = (T - size) // stride + 1
        windows = []
        for i in range(n_windows):
            start = i * stride
            windows.append(data[:, start:start + size])
        
        # Return first window (or stack all windows)
        return windows[0]  # For now, return first window
    
    def _apply_channel_alignment(self, data: np.ndarray, ds_name: str) -> np.ndarray:
        """Apply channel alignment based on mode."""
        if self.channel_align == ChannelAlignMode.NONE:
            return data
        
        elif self.channel_align == ChannelAlignMode.PAD:
            C, T = data.shape
            if C < self.max_channels:
                pad = np.zeros((self.max_channels - C, T), dtype=data.dtype)
                data = np.concatenate([data, pad], axis=0)
            return data[:self.max_channels]
        
        elif self.channel_align == ChannelAlignMode.GLOBAL:
            from ._global import M_TOTAL, get_global_channel_mapping
            
            C, T = data.shape
            aligned = np.zeros((M_TOTAL, T), dtype=data.dtype)
            
            channel_mapping = get_global_channel_mapping(ds_name)
            for local_ch, global_ch in enumerate(channel_mapping):
                if local_ch < C and global_ch < M_TOTAL:
                    aligned[global_ch] = data[local_ch]
            
            return aligned
        
        return data
    
    def get_sample_with_meta(self, index: int) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
        """Get sample with full metadata.
        
        Returns:
            Tuple of (data, label, metadata)
        """
        ds_name, local_idx = self._samples[index]
        label = self._sample_labels[index]
        
        ds = self._datasets[ds_name]
        data, meta = ds.get_normalized_sample(local_idx)
        
        # Apply preprocessing
        data = self._apply_normalization(data)
        
        if self.target_sample_rate is not None:
            orig_rate = ds.sample_rate_hz or 1.0
            if orig_rate != self.target_sample_rate:
                data = self._resample(data, orig_rate, self.target_sample_rate)
        
        if self.window_config is not None:
            data = self._apply_windowing(data)
        
        data = self._apply_channel_alignment(data, ds_name)
        
        # Add unified metadata
        meta['unified'] = {
            'dataset': ds_name,
            'local_index': local_idx,
            'task': self.task.value,
            'split': self.split.value,
        }
        
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None and label is not None:
            label = self.target_transform(label)
        
        return data, label, meta
    
    @property
    def num_classes(self) -> Optional[int]:
        """Get number of classes for classification tasks."""
        if self.task == TaskType.GAS_CLASSIFICATION:
            if self.use_global_labels:
                from ._global import GasLabel
                return len(GasLabel)
            else:
                # Max local classes across datasets
                return max(
                    getattr(ds, 'num_classes', 10) 
                    for ds in self._datasets.values()
                )
        
        elif self.task == TaskType.ODOR_CLASSIFICATION:
            return max(
                getattr(ds, 'num_classes', 50)
                for ds in self._datasets.values()
            )
        
        elif self.task == TaskType.ACTIVITY_RECOGNITION:
            from ._global import ActivityLabel
            return len(ActivityLabel)
        
        return None
    
    @property
    def num_channels(self) -> int:
        """Get number of channels (max across datasets or aligned)."""
        if self.channel_align == ChannelAlignMode.GLOBAL:
            from ._global import M_TOTAL
            return M_TOTAL
        elif self.channel_align == ChannelAlignMode.PAD:
            return self.max_channels
        else:
            return max(ds.num_sensors for ds in self._datasets.values())
    
    @property
    def dataset_names(self) -> List[str]:
        """Get list of loaded dataset names."""
        return list(self._datasets.keys())
    
    def summary(self) -> str:
        """Get summary of the dataset configuration."""
        lines = [
            f"UnifiedEnoseDataset",
            f"  Task: {self.task.value}",
            f"  Split: {self.split.value}",
            f"  Samples: {len(self)}",
            f"  Datasets: {len(self._datasets)}",
        ]
        
        for name, ds in self._datasets.items():
            lines.append(f"    - {name}: {len(ds)} samples, {ds.num_sensors} channels")
        
        lines.extend([
            f"  Normalize: {self.normalize.value}",
            f"  Channel align: {self.channel_align.value}",
        ])
        
        if self.num_classes is not None:
            lines.append(f"  Num classes: {self.num_classes}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return (
            f"UnifiedEnoseDataset(task={self.task.value}, split={self.split.value}, "
            f"datasets={self.dataset_names}, samples={len(self)})"
        )
