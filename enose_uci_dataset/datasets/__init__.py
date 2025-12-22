"""enose_uci_dataset.datasets

目标是提供和 torchvision.datasets 尽量一致的使用体验：

    from enose_uci_dataset.datasets import GasSensorsForHomeActivityMonitoring
    ds = GasSensorsForHomeActivityMonitoring(root="/path/to/data", download=True)
"""

from ._base import BaseEnoseDataset, DatasetWithTransforms, SampleRecord
from ._info import DatasetInfo, get_dataset_info, list_datasets
from .gas_sensors_for_home_activity_monitoring import GasSensorsForHomeActivityMonitoring
from .twin_gas_sensor_arrays import TwinGasSensorArrays


DATASETS = {
    "gas_sensors_for_home_activity_monitoring": GasSensorsForHomeActivityMonitoring,
    "twin_gas_sensor_arrays": TwinGasSensorArrays,
}


def get_dataset_class(name: str):
    normalized = name.lower().replace("-", "_")
    if normalized not in DATASETS:
        available = ", ".join(sorted(DATASETS.keys()))
        raise KeyError(f"Unknown dataset: {name}. Available: {available}")
    return DATASETS[normalized]


__all__ = [
    # base
    "BaseEnoseDataset",
    "DatasetWithTransforms",
    "SampleRecord",
    # metadata
    "DatasetInfo",
    "get_dataset_info",
    "list_datasets",
    # datasets
    "GasSensorsForHomeActivityMonitoring",
    "TwinGasSensorArrays",
    # registry
    "DATASETS",
    "get_dataset_class",
]
