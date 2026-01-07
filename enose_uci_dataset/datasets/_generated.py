"""
AUTO-GENERATED FILE - DO NOT EDIT MANUALLY
============================================
Generated from YAML schemas in schemas/ directory.
Timestamp: 2026-01-08 01:16:43

To regenerate this file, run:
    python scripts/generate_metadata.py

To modify sensor/dataset definitions, edit the YAML files:
    - schemas/sensors.yaml
    - schemas/datasets.yaml
    - schemas/labels.yaml
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
    "gas_sensors_for_home_activity_monitoring": 1,
    "gas_sensor_array_under_flow_modulation": 25,
    "smellnet_pure": 1,
    "smellnet_mixture": 10,
    "g919_55": 1,
}

DATASET_COLLECTION_TYPE: Dict[str, str] = {
    "twin_gas_sensor_arrays": "discrete",
    "gas_sensor_array_under_dynamic_gas_mixtures": "continuous",
    "gas_sensor_array_exposed_to_turbulent_gas_mixtures": "discrete",
    "gas_sensor_array_low_concentration": "discrete",
    "gas_sensor_array_temperature_modulation": "continuous",
    "gas_sensors_for_home_activity_monitoring": "continuous",
    "gas_sensor_array_under_flow_modulation": "discrete",
    "smellnet_pure": "discrete",
    "smellnet_mixture": "discrete",
    "g919_55": "discrete",
}

DATASET_RESPONSE_TYPE: Dict[str, str] = {
    "twin_gas_sensor_arrays": "resistance",
    "gas_sensor_array_under_dynamic_gas_mixtures": "conductance",
    "gas_sensor_array_exposed_to_turbulent_gas_mixtures": "conductance",
    "gas_sensor_array_low_concentration": "resistance_ratio",
    "gas_sensor_array_temperature_modulation": "resistance",
    "gas_sensors_for_home_activity_monitoring": "resistance",
    "gas_sensor_array_under_flow_modulation": "resistance",
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
        get_sensor_id("TEMPERATURE"),
        get_sensor_id("HUMIDITY"),
        get_sensor_id("TGS2600"),
        get_sensor_id("TGS2602"),
        get_sensor_id("TGS2602"),
        get_sensor_id("TGS2620"),
        get_sensor_id("TGS2612"),
        get_sensor_id("TGS2620"),
        get_sensor_id("TGS2611"),
        get_sensor_id("TGS2610"),
    ],
    "gas_sensor_array_low_concentration": [
        get_sensor_id("TGS2603"),
        get_sensor_id("TGS2630"),
        get_sensor_id("TGS813"),
        get_sensor_id("TGS822"),
        get_sensor_id("MQ135"),
        get_sensor_id("MQ137"),
        get_sensor_id("MQ138"),
        get_sensor_id("2M012"),
        get_sensor_id("VOCS-P"),
        get_sensor_id("2SH12"),
    ],
    "gas_sensor_array_temperature_modulation": [
        get_sensor_id("HUMIDITY"),
        get_sensor_id("TEMPERATURE"),
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
        get_sensor_id("SB-500-12"),
    ],
    "gas_sensors_for_home_activity_monitoring": [
        get_sensor_id("TGS2611"),
        get_sensor_id("TGS2612"),
        get_sensor_id("TGS2610"),
        get_sensor_id("TGS2600"),
        get_sensor_id("TGS2602"),
        get_sensor_id("TGS2602"),
        get_sensor_id("TGS2620"),
        get_sensor_id("TGS2620"),
    ],
    "gas_sensor_array_under_flow_modulation": [
        get_sensor_id("TGS2610"),
        get_sensor_id("TGS2610"),
        get_sensor_id("TGS2602"),
        get_sensor_id("TGS2600"),
        get_sensor_id("TGS2610"),
        get_sensor_id("TGS2611"),
        get_sensor_id("TGS2610"),
        get_sensor_id("TGS2620"),
        get_sensor_id("TGS2610"),
        get_sensor_id("TGS2620"),
        get_sensor_id("TGS2602"),
        get_sensor_id("TGS2611"),
        get_sensor_id("TGS2610"),
        get_sensor_id("TGS2610"),
        get_sensor_id("TGS2610"),
        get_sensor_id("TGS2600"),
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
