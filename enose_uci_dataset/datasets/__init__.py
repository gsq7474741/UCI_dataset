"""enose_uci_dataset.datasets

目标是提供和 torchvision.datasets 尽量一致的使用体验：

    from enose_uci_dataset.datasets import GasSensorsForHomeActivityMonitoring
    ds = GasSensorsForHomeActivityMonitoring(root="/path/to/data", download=True)
"""

from ._base import BaseEnoseDataset, DatasetWithTransforms, SampleRecord
from ._info import ChannelConfig, DatasetInfo, SensorConfig, get_dataset_info, list_datasets
from ._global import (
    # Global space
    OMEGA_GLOBAL, F_STD, M_TOTAL,
    # Sensor space
    SensorModel, SENSOR_MODELS, SENSOR_NAME_TO_ID, get_sensor_id, get_sensor_model,
    TargetGas, GAS_NAME_TO_ENUM,
    # Task space - types and definitions
    TaskType, TaskDefinition, TASKS,
    # Task space - classification labels
    GasLabel, ActivityLabel, IngredientLabel,
    GAS_LABEL_MAPPINGS, ACTIVITY_LABEL_MAPPINGS, get_global_label,
    # Task space - regression configs
    ConcentrationConfig, CONCENTRATION_CONFIGS, normalize_concentration,
    # Task space - drift/transfer configs
    DriftDomainConfig, DRIFT_CONFIGS,
    TransferConfig, TRANSFER_CONFIGS,
    # Dataset mappings
    DATASET_SAMPLE_RATES, DATASET_CHANNEL_TO_GLOBAL, get_global_channel_mapping,
)
from .combined import CombinedEnoseDataset, PretrainingDataset
from .alcohol_qcm_sensor import AlcoholQCMSensor
from .gas_sensor_array_drift_at_different_concentrations import GasSensorArrayDrift
from .gas_sensor_dynamic import GasSensorDynamic
from .gas_sensor_flow_modulation import GasSensorFlowModulation
from .gas_sensor_low_concentration import GasSensorLowConcentration
from .gas_sensor_temperature_modulation import GasSensorTemperatureModulation
from .gas_sensor_turbulent import GasSensorTurbulent
from .gas_sensors_for_home_activity_monitoring import GasSensorsForHomeActivityMonitoring
from .twin_gas_sensor_arrays import TwinGasSensorArrays
from .smellnet import SmellNet, SMELLNET_INGREDIENTS, SMELLNET_6_CHANNELS, SMELLNET_ALL_CHANNELS, HF_REPO_ID


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
    "smellnet": SmellNet,
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
    "SmellNet",
    "SMELLNET_INGREDIENTS",
    "SMELLNET_6_CHANNELS",
    "SMELLNET_ALL_CHANNELS",
    "HF_REPO_ID",
    # registry
    "DATASETS",
    "get_dataset_class",
    # global space (Ω_global)
    "OMEGA_GLOBAL",
    "F_STD",
    "M_TOTAL",
    # sensor space
    "SensorModel",
    "SENSOR_MODELS",
    "SENSOR_NAME_TO_ID",
    "get_sensor_id",
    "get_sensor_model",
    "TargetGas",
    "GAS_NAME_TO_ENUM",
    # task space - types
    "TaskType",
    "TaskDefinition",
    "TASKS",
    # task space - classification labels
    "GasLabel",
    "ActivityLabel",
    "IngredientLabel",
    "GAS_LABEL_MAPPINGS",
    "ACTIVITY_LABEL_MAPPINGS",
    "get_global_label",
    # task space - regression
    "ConcentrationConfig",
    "CONCENTRATION_CONFIGS",
    "normalize_concentration",
    # task space - drift/transfer
    "DriftDomainConfig",
    "DRIFT_CONFIGS",
    "TransferConfig",
    "TRANSFER_CONFIGS",
    # dataset mappings
    "DATASET_SAMPLE_RATES",
    "DATASET_CHANNEL_TO_GLOBAL",
    "get_global_channel_mapping",
]
