"""
AUTO-GENERATED FILE - DO NOT EDIT MANUALLY
============================================
Generated from YAML schemas in schemas/ directory.
Timestamp: 2026-01-08 02:01:42

To modify sensor/dataset definitions, edit the YAML files:
    - schemas/sensors.yaml
    - schemas/datasets.yaml
    - schemas/labels.yaml

This file is auto-regenerated on import if schemas change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple


# =============================================================================
# SENSOR MODELS (from schemas/sensors.yaml)
# =============================================================================

@dataclass(frozen=True)
class SensorModel:
    """Global sensor model definition with target gas mapping."""
    id: int
    name: str
    target_gases: FrozenSet[str]
    sensor_type: str
    manufacturer: str
    description: str = ""


SENSOR_MODELS: Tuple[SensorModel, ...] = (
    SensorModel(0, "UNKNOWN", frozenset({}), "UNKNOWN", "", "Unknown sensor type"),
    SensorModel(1, "TGS2600", frozenset({"Hydrogen", "CO"}), "MOX", "Figaro", "Air contaminants"),
    SensorModel(2, "TGS2602", frozenset({"Ammonia", "H2S", "VOC"}), "MOX", "Figaro", "Air quality"),
    SensorModel(3, "TGS2603", frozenset({"Ammonia", "H2S"}), "MOX", "Figaro", "Odor detection"),
    SensorModel(4, "TGS2610", frozenset({"Propane"}), "MOX", "Figaro", "LP gas"),
    SensorModel(5, "TGS2611", frozenset({"Methane"}), "MOX", "Figaro", "Methane detection"),
    SensorModel(6, "TGS2612", frozenset({"Methane", "Propane", "Butane"}), "MOX", "Figaro", "Combustible gas"),
    SensorModel(7, "TGS2620", frozenset({"Alcohol", "VOC"}), "MOX", "Figaro", "Alcohol/solvent vapors"),
    SensorModel(8, "TGS2630", frozenset({"Freon"}), "MOX", "Figaro", "Refrigerant gas"),
    SensorModel(9, "TGS3870-A04", frozenset({"CO", "Methane"}), "MOX", "Figaro", "CO/Methane combo"),
    SensorModel(10, "TGS813", frozenset({"Combustible"}), "MOX", "Figaro", "Combustible gas"),
    SensorModel(11, "TGS822", frozenset({"Alcohol", "VOC"}), "MOX", "Figaro", "Organic solvent"),
    SensorModel(12, "TGS2609", frozenset({"VOC", "Odor"}), "MOX", "Figaro", "VOC/Odor detection"),
    SensorModel(13, "SB-500-12", frozenset({"CO"}), "MOX", "FIS", "CO detection"),
    SensorModel(14, "MQ2", frozenset({"Combustible", "Smoke"}), "MOX", "Winsen", "Combustible gas/Smoke"),
    SensorModel(15, "MQ3", frozenset({"Alcohol"}), "MOX", "Winsen", "Alcohol sensor"),
    SensorModel(16, "MQ5", frozenset({"LPG"}), "MOX", "Winsen", "LPG/natural gas"),
    SensorModel(17, "MQ9", frozenset({"CO"}), "MOX", "Winsen", "CO sensor"),
    SensorModel(18, "MQ135", frozenset({"NH3", "NOx", "Benzene"}), "MOX", "Winsen", "Air quality"),
    SensorModel(19, "MQ137", frozenset({"Ammonia"}), "MOX", "Winsen", "Ammonia sensor"),
    SensorModel(20, "MQ138", frozenset({"Alcohol", "Benzene", "VOC"}), "MOX", "Winsen", "Organic gas"),
    SensorModel(21, "GM102B", frozenset({"NO2"}), "MOX", "Winsen", "NO2 sensor (Grove V2)"),
    SensorModel(22, "GM202B", frozenset({"Formaldehyde", "VOC"}), "MOX", "Winsen", "Formaldehyde/VOC (Grove V2)"),
    SensorModel(23, "GM302B", frozenset({"Ethanol"}), "MOX", "Winsen", "Ethanol sensor (Grove V2)"),
    SensorModel(24, "GM502B", frozenset({"VOC"}), "MOX", "Winsen", "VOC sensor (Grove V2)"),
    SensorModel(25, "GM702B", frozenset({"CO"}), "MOX", "Winsen", "CO sensor (Grove V2)"),
    SensorModel(26, "SMD1005", frozenset({"VOC"}), "MEMS", "Huiwen", "VOC MEMS sensor"),
    SensorModel(27, "SMD1013B", frozenset({"Combustible"}), "MEMS", "Huiwen", "Combustible gas MEMS"),
    SensorModel(28, "MP503", frozenset({"NO2"}), "MOX", "Winsen", "NO2 sensor"),
    SensorModel(29, "WSP2110", frozenset({"Ethanol", "VOC"}), "MOX", "Winsen", "VOC/Ethanol sensor"),
    SensorModel(30, "2M012", frozenset({"Formaldehyde"}), "EC", "Generic", "Formaldehyde"),
    SensorModel(31, "2SH12", frozenset({"H2S"}), "EC", "Generic", "H2S sensor"),
    SensorModel(32, "VOCS-P", frozenset({"VOC"}), "PID", "Generic", "VOC PID sensor"),
    SensorModel(33, "MOX", frozenset({}), "MOX", "Generic", "Generic MOX sensor"),
    SensorModel(34, "TEMPERATURE", frozenset({}), "ENV", "Virtual", "Virtual temperature sensor (°C)"),
    SensorModel(35, "HUMIDITY", frozenset({}), "ENV", "Virtual", "Virtual humidity sensor (%RH)"),
)

# Quick lookup: sensor name -> global index
SENSOR_NAME_TO_ID: Dict[str, int] = {s.name: s.id for s in SENSOR_MODELS}

# Total dimension of universal sensor space
M_TOTAL: int = len(SENSOR_MODELS)


def get_sensor_id(name: str) -> int:
    """Get global sensor index by name. Returns 0 (UNKNOWN) if not found."""
    return SENSOR_NAME_TO_ID.get(name, 0)


def get_sensor_model(name: str) -> SensorModel:
    """Get sensor model by name. Returns UNKNOWN if not found."""
    idx = get_sensor_id(name)
    return SENSOR_MODELS[idx]

# =============================================================================
# DATASET METADATA (from schemas/datasets.yaml)
# =============================================================================

DATASET_SAMPLE_RATES: Dict[str, Optional[float]] = {
    "twin_gas_sensor_arrays": 100,
    "gas_sensor_array_under_dynamic_gas_mixtures": 100,
    "gas_sensor_array_exposed_to_turbulent_gas_mixtures": 50,
    "gas_sensor_array_low_concentration": 1,
    "gas_sensor_array_temperature_modulation": 3.5,
    "gas_sensor_array_under_flow_modulation": 25,
    "gas_sensors_for_home_activity_monitoring": 1,
    "smellnet_pure": 1,
    "smellnet_mixture": 10,
    "g919_55": 1,
}

DATASET_COLLECTION_TYPE: Dict[str, str] = {
    "twin_gas_sensor_arrays": "discrete",
    "gas_sensor_array_under_dynamic_gas_mixtures": "continuous",
    "gas_sensor_array_exposed_to_turbulent_gas_mixtures": "continuous",
    "gas_sensor_array_low_concentration": "continuous",
    "gas_sensor_array_temperature_modulation": "continuous",
    "gas_sensor_array_under_flow_modulation": "discrete",
    "gas_sensors_for_home_activity_monitoring": "continuous",
    "smellnet_pure": "discrete",
    "smellnet_mixture": "discrete",
    "g919_55": "discrete",
}

DATASET_RESPONSE_TYPE: Dict[str, str] = {
    "twin_gas_sensor_arrays": "resistance",
    "gas_sensor_array_under_dynamic_gas_mixtures": "conductance",
    "gas_sensor_array_exposed_to_turbulent_gas_mixtures": "resistance",
    "gas_sensor_array_low_concentration": "resistance",
    "gas_sensor_array_temperature_modulation": "resistance",
    "gas_sensor_array_under_flow_modulation": "resistance",
    "gas_sensors_for_home_activity_monitoring": "resistance",
    "smellnet_pure": "resistance_ratio",
    "smellnet_mixture": "resistance_ratio",
    "g919_55": "resistance_ratio",
}

# Channel to global sensor ID mapping
DATASET_CHANNEL_TO_GLOBAL: Dict[str, List[int]] = {
    "twin_gas_sensor_arrays": [
        get_sensor_id("TGS2611"),
        get_sensor_id("TGS2612"),
        get_sensor_id("TGS2610"),
        get_sensor_id("TGS2602"),
        get_sensor_id("TGS2611"),
        get_sensor_id("TGS2612"),
        get_sensor_id("TGS2610"),
        get_sensor_id("TGS2602"),
    ],
    "gas_sensor_array_under_dynamic_gas_mixtures": [
        get_sensor_id("TGS2602"),
        get_sensor_id("TGS2602"),
        get_sensor_id("TGS2600"),
        get_sensor_id("TGS2600"),
        get_sensor_id("TGS2610"),
        get_sensor_id("TGS2610"),
        get_sensor_id("TGS2620"),
        get_sensor_id("TGS2620"),
        get_sensor_id("TGS2602"),
        get_sensor_id("TGS2602"),
        get_sensor_id("TGS2600"),
        get_sensor_id("TGS2600"),
        get_sensor_id("TGS2610"),
        get_sensor_id("TGS2610"),
        get_sensor_id("TGS2620"),
        get_sensor_id("TGS2620"),
    ],
    "gas_sensor_array_exposed_to_turbulent_gas_mixtures": [
        get_sensor_id("TGS2611"),
        get_sensor_id("TGS2612"),
        get_sensor_id("TGS2610"),
        get_sensor_id("TGS2602"),
        get_sensor_id("TGS2611"),
        get_sensor_id("TGS2612"),
        get_sensor_id("TGS2610"),
        get_sensor_id("TGS2602"),
        get_sensor_id("TEMPERATURE"),
        get_sensor_id("HUMIDITY"),
    ],
    "gas_sensor_array_low_concentration": [
        get_sensor_id("TGS2611"),
        get_sensor_id("TGS2620"),
        get_sensor_id("TGS2602"),
        get_sensor_id("TGS826"),
        get_sensor_id("MQ136"),
        get_sensor_id("MQ137"),
        get_sensor_id("MQ138"),
        get_sensor_id("MQ3"),
        get_sensor_id("MQ9"),
        get_sensor_id("MQ135"),
    ],
    "gas_sensor_array_temperature_modulation": [
        get_sensor_id("TGS3870-A04"),
        get_sensor_id("TGS3870-A04"),
        get_sensor_id("TGS3870-A04"),
        get_sensor_id("TGS3870-A04"),
        get_sensor_id("TGS3870-A04"),
        get_sensor_id("TGS3870-A04"),
        get_sensor_id("TGS3870-A04"),
        get_sensor_id("TGS3870-A04"),
        get_sensor_id("SB-500-12"),
        get_sensor_id("SB-500-12"),
        get_sensor_id("SB-500-12"),
        get_sensor_id("SB-500-12"),
        get_sensor_id("SB-500-12"),
        get_sensor_id("SB-500-12"),
        get_sensor_id("TEMPERATURE"),
        get_sensor_id("HUMIDITY"),
    ],
    "gas_sensor_array_under_flow_modulation": [
        get_sensor_id("TGS2611"),
        get_sensor_id("TGS2612"),
        get_sensor_id("TGS2610"),
        get_sensor_id("TGS2602"),
        get_sensor_id("TGS2611"),
        get_sensor_id("TGS2612"),
        get_sensor_id("TGS2610"),
        get_sensor_id("TGS2602"),
        get_sensor_id("TGS2620"),
        get_sensor_id("TGS2620"),
        get_sensor_id("TGS2602"),
        get_sensor_id("TGS2611"),
        get_sensor_id("TGS2610"),
        get_sensor_id("TGS2610"),
        get_sensor_id("TGS2610"),
        get_sensor_id("TGS2600"),
    ],
    "gas_sensors_for_home_activity_monitoring": [
        get_sensor_id("TGS2602"),
        get_sensor_id("TGS2602"),
        get_sensor_id("TGS2600"),
        get_sensor_id("TGS2600"),
        get_sensor_id("TGS2610"),
        get_sensor_id("TGS2610"),
        get_sensor_id("TGS2620"),
        get_sensor_id("TGS2620"),
    ],
    "smellnet_pure": [
        get_sensor_id("GM102B"),
        get_sensor_id("GM302B"),
        get_sensor_id("GM502B"),
        get_sensor_id("GM702B"),
        get_sensor_id("MQ135"),
        get_sensor_id("MQ9"),
    ],
    "smellnet_mixture": [
        get_sensor_id("GM102B"),
        get_sensor_id("GM302B"),
        get_sensor_id("GM502B"),
        get_sensor_id("GM702B"),
        get_sensor_id("MQ135"),
        get_sensor_id("MQ9"),
    ],
    "g919_55": [
        get_sensor_id("GM502B"),
        get_sensor_id("GM202B"),
        get_sensor_id("SMD1005"),
        get_sensor_id("SMD1013B"),
        get_sensor_id("MQ138"),
        get_sensor_id("MQ2"),
        get_sensor_id("TGS2609"),
        get_sensor_id("TGS2620"),
    ],
}


def get_global_channel_mapping(dataset_name: str) -> List[int]:
    """Get global sensor indices for a dataset's channels."""
    return DATASET_CHANNEL_TO_GLOBAL.get(dataset_name, [])

# =============================================================================
# DATASET INFO (from schemas/datasets.yaml)
# =============================================================================

DATASET_INFO: Dict[str, Dict] = {
    "twin_gas_sensor_arrays": {
        "name": "twin_gas_sensor_arrays",
        "uci_id": 361,
        "file_name": "twin+gas+sensor+arrays.zip",
        "sha1": "a235ab3cf55685fc8346eafea3f18e080adb77a3",
        "url": "https://archive.ics.uci.edu/static/public/361/twin+gas+sensor+arrays.zip",
        "description": "Twin Gas Sensor Arrays",
        "tasks": ['classification', 'drift_compensation', 'transfer_learning'],
        "sensor_type": "MOX",
        "manufacturer": "Figaro",
        "sample_rate_hz": 100,
        "collection_type": "discrete",
        "response_type": "resistance",
        "extract_type": "standard",
        "extract_subdir": "data1",
        "nested_zip": "",
        "paper_link": "",
        "channel_sensors": ['TGS2611', 'TGS2612', 'TGS2610', 'TGS2602', 'TGS2611', 'TGS2612', 'TGS2610', 'TGS2602'],
        "num_channels": 8,
    },
    "gas_sensor_array_under_dynamic_gas_mixtures": {
        "name": "gas_sensor_array_under_dynamic_gas_mixtures",
        "uci_id": 322,
        "file_name": "gas+sensor+array+under+dynamic+gas+mixtures.zip",
        "sha1": "12dd9c43e621d56b53f2f8894dddd6e410283d47",
        "url": "https://archive.ics.uci.edu/static/public/322/gas+sensor+array+under+dynamic+gas+mixtures.zip",
        "description": "Gas Sensor Array under Dynamic Gas Mixtures",
        "tasks": ['classification', 'concentration_prediction'],
        "sensor_type": "MOX",
        "manufacturer": "Figaro",
        "sample_rate_hz": 100,
        "collection_type": "continuous",
        "response_type": "conductance",
        "extract_type": "standard",
        "extract_subdir": "",
        "nested_zip": "",
        "paper_link": "",
        "channel_sensors": ['TGS2602', 'TGS2602', 'TGS2600', 'TGS2600', 'TGS2610', 'TGS2610', 'TGS2620', 'TGS2620', 'TGS2602', 'TGS2602', 'TGS2600', 'TGS2600', 'TGS2610', 'TGS2610', 'TGS2620', 'TGS2620'],
        "num_channels": 16,
    },
    "gas_sensor_array_exposed_to_turbulent_gas_mixtures": {
        "name": "gas_sensor_array_exposed_to_turbulent_gas_mixtures",
        "uci_id": 309,
        "file_name": "gas+sensor+array+exposed+to+turbulent+gas+mixtures.zip",
        "sha1": "c812370aa04a1491781767b3cf606ce6fa0d221e",
        "url": "https://archive.ics.uci.edu/static/public/309/gas+sensor+array+exposed+to+turbulent+gas+mixtures.zip",
        "description": "Gas Sensor Array Exposed to Turbulent Gas Mixtures",
        "tasks": ['classification', 'concentration_prediction'],
        "sensor_type": "MOX",
        "manufacturer": "Figaro",
        "sample_rate_hz": 50,
        "collection_type": "continuous",
        "response_type": "resistance",
        "extract_type": "turbo",
        "extract_subdir": "",
        "nested_zip": "",
        "paper_link": "",
        "channel_sensors": ['TGS2611', 'TGS2612', 'TGS2610', 'TGS2602', 'TGS2611', 'TGS2612', 'TGS2610', 'TGS2602', 'TEMPERATURE', 'HUMIDITY'],
        "num_channels": 10,
    },
    "gas_sensor_array_low_concentration": {
        "name": "gas_sensor_array_low_concentration",
        "uci_id": 1081,
        "file_name": "gas+sensor+array+low-concentration.zip",
        "sha1": "41818a074550dd23a222856afdc19c7fbf32903e",
        "url": "https://archive.ics.uci.edu/static/public/1081/gas+sensor+array+low-concentration.zip",
        "description": "Gas Sensor Array Low-Concentration Dataset",
        "tasks": ['classification', 'concentration_prediction'],
        "sensor_type": "MOX",
        "manufacturer": "Mixed",
        "sample_rate_hz": 1,
        "collection_type": "continuous",
        "response_type": "resistance",
        "extract_type": "standard",
        "extract_subdir": "",
        "nested_zip": "",
        "paper_link": "https://doi.org/10.1109/TIM.2023.3251416",
        "channel_sensors": ['TGS2611', 'TGS2620', 'TGS2602', 'TGS826', 'MQ136', 'MQ137', 'MQ138', 'MQ3', 'MQ9', 'MQ135'],
        "num_channels": 10,
    },
    "gas_sensor_array_temperature_modulation": {
        "name": "gas_sensor_array_temperature_modulation",
        "uci_id": 487,
        "file_name": "gas+sensor+array+temperature+modulation.zip",
        "sha1": "e9caaded42fa57086da2a56f5a3dfb2f7a7708d8",
        "url": "https://archive.ics.uci.edu/static/public/487/gas+sensor+array+temperature+modulation.zip",
        "description": "Gas Sensor Array Temperature Modulation",
        "tasks": ['classification'],
        "sensor_type": "MOX",
        "manufacturer": "Figaro/FIS",
        "sample_rate_hz": 3.5,
        "collection_type": "continuous",
        "response_type": "resistance",
        "extract_type": "turbo",
        "extract_subdir": "",
        "nested_zip": "",
        "paper_link": "",
        "channel_sensors": ['TGS3870-A04', 'TGS3870-A04', 'TGS3870-A04', 'TGS3870-A04', 'TGS3870-A04', 'TGS3870-A04', 'TGS3870-A04', 'TGS3870-A04', 'SB-500-12', 'SB-500-12', 'SB-500-12', 'SB-500-12', 'SB-500-12', 'SB-500-12', 'TEMPERATURE', 'HUMIDITY'],
        "num_channels": 16,
    },
    "gas_sensor_array_under_flow_modulation": {
        "name": "gas_sensor_array_under_flow_modulation",
        "uci_id": 308,
        "file_name": "gas+sensor+array+under+flow+modulation.zip",
        "sha1": "d8c3de28ece518cf57ed842c1d5b8a23ee03398b",
        "url": "https://archive.ics.uci.edu/static/public/308/gas+sensor+array+under+flow+modulation.zip",
        "description": "Gas Sensor Array under Flow Modulation",
        "tasks": ['classification'],
        "sensor_type": "MOX",
        "manufacturer": "Figaro",
        "sample_rate_hz": 25,
        "collection_type": "discrete",
        "response_type": "resistance",
        "extract_type": "turbo",
        "extract_subdir": "",
        "nested_zip": "",
        "paper_link": "",
        "channel_sensors": ['TGS2611', 'TGS2612', 'TGS2610', 'TGS2602', 'TGS2611', 'TGS2612', 'TGS2610', 'TGS2602', 'TGS2620', 'TGS2620', 'TGS2602', 'TGS2611', 'TGS2610', 'TGS2610', 'TGS2610', 'TGS2600'],
        "num_channels": 16,
    },
    "gas_sensors_for_home_activity_monitoring": {
        "name": "gas_sensors_for_home_activity_monitoring",
        "uci_id": 362,
        "file_name": "gas+sensors+for+home+activity+monitoring.zip",
        "sha1": "34101ca24e556dc14a6ee1e2910111ed49c0e6ce",
        "url": "https://archive.ics.uci.edu/static/public/362/gas+sensors+for+home+activity+monitoring.zip",
        "description": "Gas Sensors for Home Activity Monitoring",
        "tasks": ['classification', 'activity_recognition'],
        "sensor_type": "MOX",
        "manufacturer": "Figaro",
        "sample_rate_hz": 1,
        "collection_type": "continuous",
        "response_type": "resistance",
        "extract_type": "nested",
        "extract_subdir": "",
        "nested_zip": "HT_Sensor_dataset.zip",
        "paper_link": "",
        "channel_sensors": ['TGS2602', 'TGS2602', 'TGS2600', 'TGS2600', 'TGS2610', 'TGS2610', 'TGS2620', 'TGS2620'],
        "num_channels": 8,
    },
    "smellnet_pure": {
        "name": "smellnet_pure",
        "uci_id": None,
        "file_name": "",
        "sha1": "",
        "url": "https://huggingface.co/datasets/DeweiFeng/smell-net",
        "description": "SmellNet Pure Substances: 50 classes @ 1Hz",
        "tasks": ['classification'],
        "sensor_type": "MOX",
        "manufacturer": "Seeed/Winsen",
        "sample_rate_hz": 1,
        "collection_type": "discrete",
        "response_type": "resistance_ratio",
        "extract_type": "huggingface",
        "extract_subdir": "",
        "nested_zip": "",
        "paper_link": "",
        "channel_sensors": ['GM102B', 'GM302B', 'GM502B', 'GM702B', 'MQ135', 'MQ9'],
        "num_channels": 6,
    },
    "smellnet_mixture": {
        "name": "smellnet_mixture",
        "uci_id": None,
        "file_name": "",
        "sha1": "",
        "url": "https://huggingface.co/datasets/DeweiFeng/smell-net",
        "description": "SmellNet Mixtures: 43 classes @ 10Hz",
        "tasks": ['classification'],
        "sensor_type": "MOX",
        "manufacturer": "Seeed/Winsen",
        "sample_rate_hz": 10,
        "collection_type": "discrete",
        "response_type": "resistance_ratio",
        "extract_type": "huggingface",
        "extract_subdir": "",
        "nested_zip": "",
        "paper_link": "",
        "channel_sensors": ['GM102B', 'GM302B', 'GM502B', 'GM702B', 'MQ135', 'MQ9'],
        "num_channels": 6,
    },
    "g919_55": {
        "name": "g919_55",
        "uci_id": None,
        "file_name": "",
        "sha1": "",
        "url": "",
        "description": "SJTU-G919: 55 odor classes, 8-channel MEMS+Ceramic array",
        "tasks": ['classification'],
        "sensor_type": "MOX/MEMS",
        "manufacturer": "Winsen/Huiwen/Figaro",
        "sample_rate_hz": 1,
        "collection_type": "discrete",
        "response_type": "resistance_ratio",
        "extract_type": "local",
        "extract_subdir": "",
        "nested_zip": "",
        "paper_link": "",
        "channel_sensors": ['GM502B', 'GM202B', 'SMD1005', 'SMD1013B', 'MQ138', 'MQ2', 'TGS2609', 'TGS2620'],
        "num_channels": 8,
    },
}


def get_dataset_info_dict(name: str) -> Dict:
    """Get dataset info dictionary by name."""
    normalized = name.lower().replace('-', '_')
    if normalized == 'smellnet':
        normalized = 'smellnet_pure'
    if normalized not in DATASET_INFO:
        available = ', '.join(sorted(DATASET_INFO.keys()))
        raise KeyError(f"Unknown dataset: {name}. Available: {available}")
    return DATASET_INFO[normalized]


def list_all_datasets() -> List[str]:
    """List all available dataset names."""
    return sorted(DATASET_INFO.keys())

# =============================================================================
# TASK TYPES (from schemas/tasks.yaml)
# =============================================================================

TASK_TYPES: Dict[str, int] = {
    "raw": 0,
    "gas_classification": 1,
    "odor_classification": 2,
    "activity_recognition": 3,
    "concentration_regression": 4,
    "drift_compensation": 5,
    "anomaly_detection": 6,
    "self_supervised": 7,
    "multi_task": 8,
}

TASK_ID_TO_NAME: Dict[int, str] = {v: k for k, v in TASK_TYPES.items()}

TASK_DATASET_COMPATIBILITY: Dict[str, List[str]] = {
    "gas_classification": ['twin_gas_sensor_arrays', 'gas_sensor_array_exposed_to_turbulent_gas_mixtures', 'gas_sensor_array_low_concentration', 'gas_sensor_array_temperature_modulation', 'gas_sensor_array_under_dynamic_gas_mixtures', 'gas_sensor_array_under_flow_modulation'],
    "odor_classification": ['smellnet_pure', 'smellnet_mixture', 'g919_55'],
    "activity_recognition": ['gas_sensors_for_home_activity_monitoring'],
    "concentration_regression": ['gas_sensor_array_exposed_to_turbulent_gas_mixtures', 'gas_sensor_array_low_concentration', 'gas_sensor_array_under_dynamic_gas_mixtures'],
    "drift_compensation": ['twin_gas_sensor_arrays'],
    "anomaly_detection": [],  # all datasets
    "self_supervised": [],  # all datasets
    "multi_task": [],  # all datasets
    "raw": [],  # all datasets
}

# =============================================================================
# GAS LABELS (from schemas/labels.yaml)
# =============================================================================

# Gas label name to ID mapping
GAS_LABEL_TO_ID: Dict[str, int] = {
    "UNKNOWN": 0,
    "CO": 1,
    "ETHANOL": 2,
    "ETHYLENE": 3,
    "METHANE": 4,
    "AMMONIA": 5,
    "ACETONE": 6,
    "TOLUENE": 7,
    "BENZENE": 8,
    "FORMALDEHYDE": 9,
    "HYDROGEN": 10,
    "PROPANE": 11,
    "BUTANE": 12,
    "NO2": 13,
    "H2S": 14,
    "ISOPROPANOL": 15,
    "PROPANOL": 16,
    "OCTANOL": 17,
    "BUTANOL": 18,
    "ISOBUTANOL": 19,
    "ETHYL_ACETATE": 20,
    "N_HEXANE": 21,
    "ACETALDEHYDE": 22,
    "MIXTURE_ETHYLENE_CO": 23,
    "MIXTURE_ETHYLENE_METHANE": 24,
    "MIXTURE_ACETONE_ETHANOL": 25,
    "BACKGROUND": 26,
}

# ID to gas label name mapping
GAS_LABEL_ID_TO_NAME: Dict[int, str] = {v: k for k, v in GAS_LABEL_TO_ID.items()}

# Dataset-local to global label mappings
GAS_LABEL_MAPPINGS: Dict[str, Dict[int, int]] = {
    "twin_gas_sensor_arrays": {
        0: 2,  # ETHANOL
        1: 1,  # CO
        2: 3,  # ETHYLENE
        3: 4,  # METHANE
    },
    "gas_sensor_array_under_dynamic_gas_mixtures": {
        0: 23,  # MIXTURE_ETHYLENE_CO
        1: 24,  # MIXTURE_ETHYLENE_METHANE
    },
    "gas_sensor_array_exposed_to_turbulent_gas_mixtures": {
        0: 24,  # MIXTURE_ETHYLENE_METHANE
        1: 23,  # MIXTURE_ETHYLENE_CO
    },
    "gas_sensor_array_low_concentration": {
        0: 2,  # ETHANOL
        1: 6,  # ACETONE
        2: 7,  # TOLUENE
        3: 20,  # ETHYL_ACETATE
        4: 15,  # ISOPROPANOL
        5: 21,  # N_HEXANE
    },
    "gas_sensor_array_under_flow_modulation": {
        0: 26,  # BACKGROUND
        1: 6,  # ACETONE
        2: 2,  # ETHANOL
        3: 25,  # MIXTURE_ACETONE_ETHANOL
    },
    "alcohol_qcm_sensor_dataset": {
        0: 17,  # OCTANOL
        1: 16,  # PROPANOL
        2: 18,  # BUTANOL
        3: 15,  # ISOPROPANOL
        4: 19,  # ISOBUTANOL
    },
    "gas_sensor_array_drift_dataset_at_different_concentrations": {
        0: 2,  # ETHANOL
        1: 3,  # ETHYLENE
        2: 5,  # AMMONIA
        3: 22,  # ACETALDEHYDE
        4: 6,  # ACETONE
        5: 7,  # TOLUENE
    },
}


def get_global_gas_label(dataset_name: str, local_label: int) -> int:
    """Convert dataset-local label to global label index."""
    mapping = GAS_LABEL_MAPPINGS.get(dataset_name, {})
    return mapping.get(local_label, 0)  # 0 = UNKNOWN
