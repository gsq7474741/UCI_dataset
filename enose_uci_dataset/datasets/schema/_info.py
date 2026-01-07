"""Dataset information module - simplified version using YAML-generated data.

This module provides DatasetInfo dataclasses and accessor functions.
All actual data is now sourced from schemas/datasets.yaml via _generated.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ExtractConfig:
    """Configuration for dataset extraction."""
    type: str = "standard"
    subdir: Optional[str] = None
    nested_zip: Optional[str] = None


@dataclass(frozen=True)
class ChannelConfig:
    """Configuration for a single sensor channel."""
    index: int
    sensor_model: str
    target_gases: Tuple[str, ...] = ()
    unit: str = "raw"
    heater_voltage: Optional[float] = None
    description: str = ""


@dataclass(frozen=True)
class SensorConfig:
    """Configuration for the sensor array."""
    type: str = "MOX"
    count: int = 0
    channels: Tuple[ChannelConfig, ...] = ()
    manufacturer: str = ""


@dataclass(frozen=True)
class TimeSeriesConfig:
    """Time series configuration."""
    continuous: bool = False
    sample_rate_hz: Optional[float] = None


@dataclass(frozen=True)
class DatasetInfo:
    """Dataset metadata container."""
    name: str
    uci_id: Optional[int]
    file_name: str
    sha1: str
    url: str
    description: str = ""
    tasks: List[str] = field(default_factory=list)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    time_series: Optional[TimeSeriesConfig] = None
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    paper_link: Optional[str] = None


def _build_dataset_info(info_dict: Dict) -> DatasetInfo:
    """Build DatasetInfo from generated dictionary."""
    # Build channel configs
    channel_sensors = info_dict.get("channel_sensors", [])
    channels = tuple(
        ChannelConfig(index=i, sensor_model=sensor)
        for i, sensor in enumerate(channel_sensors)
    )
    
    # Build sensor config
    sensors = SensorConfig(
        type=info_dict.get("sensor_type", "MOX"),
        count=info_dict.get("num_channels", len(channel_sensors)),
        channels=channels,
        manufacturer=info_dict.get("manufacturer", ""),
    )
    
    # Build time series config
    sample_rate = info_dict.get("sample_rate_hz")
    collection_type = info_dict.get("collection_type", "discrete")
    time_series = TimeSeriesConfig(
        continuous=(collection_type == "continuous"),
        sample_rate_hz=sample_rate,
    )
    
    # Build extract config
    extract = ExtractConfig(
        type=info_dict.get("extract_type", "standard"),
        subdir=info_dict.get("extract_subdir") or None,
        nested_zip=info_dict.get("nested_zip") or None,
    )
    
    return DatasetInfo(
        name=info_dict["name"],
        uci_id=info_dict.get("uci_id"),
        file_name=info_dict.get("file_name", ""),
        sha1=info_dict.get("sha1", ""),
        url=info_dict.get("url", ""),
        description=info_dict.get("description", ""),
        tasks=info_dict.get("tasks", []),
        sensors=sensors,
        time_series=time_series,
        extract=extract,
        paper_link=info_dict.get("paper_link"),
    )


# Cache for DatasetInfo instances
_DATASET_INFO_CACHE: Dict[str, DatasetInfo] = {}


def get_dataset_info(name: str) -> DatasetInfo:
    """Get DatasetInfo by dataset name.
    
    Data is sourced from schemas/datasets.yaml via _generated.py.
    """
    normalized = name.lower().replace("-", "_")
    if normalized == "smellnet":
        normalized = "smellnet_pure"
    
    if normalized not in _DATASET_INFO_CACHE:
        # Import here to avoid circular imports
        from ._generated import get_dataset_info_dict
        info_dict = get_dataset_info_dict(normalized)
        _DATASET_INFO_CACHE[normalized] = _build_dataset_info(info_dict)
    
    return _DATASET_INFO_CACHE[normalized]


def list_datasets() -> List[str]:
    """List all available dataset names."""
    from ._generated import list_all_datasets
    return list_all_datasets()
