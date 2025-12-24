"""enose_uci_dataset.datasets

目标是提供和 torchvision.datasets 尽量一致的使用体验：

    from enose_uci_dataset.datasets import GasSensorsForHomeActivityMonitoring
    ds = GasSensorsForHomeActivityMonitoring(root="/path/to/data", download=True)
"""

from ._base import BaseEnoseDataset, DatasetWithTransforms, SampleRecord
from ._info import ChannelConfig, DatasetInfo, SensorConfig, get_dataset_info, list_datasets
from .combined import CombinedEnoseDataset, PretrainingDataset
from .alcohol_qcm_sensor import AlcoholQCMSensor
from .gas_sensor_array_drift import GasSensorArrayDrift
from .gas_sensor_dynamic import GasSensorDynamic
from .gas_sensor_flow_modulation import GasSensorFlowModulation
from .gas_sensor_low_concentration import GasSensorLowConcentration
from .gas_sensor_temperature_modulation import GasSensorTemperatureModulation
from .gas_sensor_turbulent import GasSensorTurbulent
from .gas_sensors_for_home_activity_monitoring import GasSensorsForHomeActivityMonitoring
from .twin_gas_sensor_arrays import TwinGasSensorArrays


DATASETS = {
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
    "ChannelConfig",
    "DatasetInfo",
    "SensorConfig",
    "get_dataset_info",
    "list_datasets",
    # combined datasets
    "CombinedEnoseDataset",
    "PretrainingDataset",
    # datasets
    "AlcoholQCMSensor",
    "GasSensorArrayDrift",
    "GasSensorDynamic",
    "GasSensorFlowModulation",
    "GasSensorLowConcentration",
    "GasSensorTemperatureModulation",
    "GasSensorTurbulent",
    "GasSensorsForHomeActivityMonitoring",
    "TwinGasSensorArrays",
    # registry
    "DATASETS",
    "get_dataset_class",
]
