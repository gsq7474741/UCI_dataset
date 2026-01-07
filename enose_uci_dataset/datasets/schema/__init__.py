"""Schema module - metadata definitions and auto-generation."""
from ._info import (
    DatasetInfo,
    SensorConfig,
    ChannelConfig,
    TimeSeriesConfig,
    ExtractConfig,
    get_dataset_info,
)

__all__ = [
    "DatasetInfo",
    "SensorConfig",
    "ChannelConfig",
    "TimeSeriesConfig",
    "ExtractConfig",
    "get_dataset_info",
]
