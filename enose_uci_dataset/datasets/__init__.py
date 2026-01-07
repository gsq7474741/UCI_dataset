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
from .unified import (
    UnifiedEnoseDataset,
    TaskType,
    SplitType,
    NormalizeType,
    ChannelAlignMode,
    SplitConfig,
    WindowConfig,
    UnifiedDatasetConfig,
    TASK_DATASET_COMPATIBILITY,
)
from .gas_sensor_dynamic import GasSensorDynamic
from .gas_sensor_flow_modulation import GasSensorFlowModulation
from .gas_sensor_low_concentration import GasSensorLowConcentration
from .gas_sensor_temperature_modulation import GasSensorTemperatureModulation
from .gas_sensor_turbulent import GasSensorTurbulent
from .gas_sensors_for_home_activity_monitoring import GasSensorsForHomeActivityMonitoring
from .twin_gas_sensor_arrays import TwinGasSensorArrays
from .smellnet import SmellNet, SMELLNET_INGREDIENTS, SMELLNET_6_CHANNELS, SMELLNET_ALL_CHANNELS, HF_REPO_ID
from .g919_55 import (
    G919SensorDataset,
    G919_CATEGORIES,
    G919_ODOR_TO_IDX,
    G919_IDX_TO_ODOR,
    G919_NUM_CLASSES,
    G919_CATEGORY_TO_IDX,
)


DATASETS = {
    "gas_sensor_array_exposed_to_turbulent_gas_mixtures": GasSensorTurbulent,
    "gas_sensor_array_low_concentration": GasSensorLowConcentration,
    "gas_sensor_array_temperature_modulation": GasSensorTemperatureModulation,
    "gas_sensor_array_under_dynamic_gas_mixtures": GasSensorDynamic,
    "gas_sensor_array_under_flow_modulation": GasSensorFlowModulation,
    "gas_sensors_for_home_activity_monitoring": GasSensorsForHomeActivityMonitoring,
    "twin_gas_sensor_arrays": TwinGasSensorArrays,
    "smellnet": SmellNet,
    "g919_55": G919SensorDataset,
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
    # unified dataset
    "UnifiedEnoseDataset",
    "TaskType",
    "SplitType",
    "NormalizeType",
    "ChannelAlignMode",
    "SplitConfig",
    "WindowConfig",
    "UnifiedDatasetConfig",
    "TASK_DATASET_COMPATIBILITY",
    # datasets
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
    # G919-55
    "G919SensorDataset",
    "G919_CATEGORIES",
    "G919_ODOR_TO_IDX",
    "G919_IDX_TO_ODOR",
    "G919_NUM_CLASSES",
    "G919_CATEGORY_TO_IDX",
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
