"""Global Reference Space (Ω_global) for heterogeneous e-nose dataset standardization.

This module defines the global reference space for standardizing heterogeneous
sensor data across multiple datasets, following the formalization:

    Ω_global = (f_std, S_all, L_all)

where:
    - f_std: Unified sampling rate (target temporal resolution)
    - S_all: Universal sensor space (union of all sensor models)
    - L_all: Universal task space (global label indices and unit standards)

The standardization transformation F maps any raw sample d_i = (X_i, Y_i, M_i)
to a standardized form d̂_i = F(d_i | Ω_global).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


# =============================================================================
# 1. Unified Sampling Rate (f_std)
# =============================================================================

# Target sampling rate for temporal standardization
# Chosen as the lowest common rate that preserves information from most datasets
# Available rates: 100, 50, 25, 10, 3, 1 Hz
# Using 10 Hz as balance between information preservation and computational efficiency
F_STD: int = 10  # Hz

# Original sampling rates by dataset (for resampling calculation)
# Verified from UCI dataset pages
DATASET_SAMPLE_RATES: Dict[str, Optional[float]] = {
    "twin_gas_sensor_arrays": 100,           # UCI: 100 Hz
    "gas_sensor_array_under_dynamic_gas_mixtures": 100,  # UCI: 100 Hz
    "gas_sensor_array_exposed_to_turbulent_gas_mixtures": 50,  # UCI: 20ms = 50 Hz
    "gas_sensor_array_under_flow_modulation": 25,  # UCI: 25 Hz
    "smellnet_pure": 1,                       # Paper: 1 Hz for Pure Substances (50 classes)
    "smellnet_mixture": 10,                   # Paper: 10 Hz for Mixture Substances (43 classes)
    "gas_sensor_array_temperature_modulation": 3.5,  # UCI: 3.5 Hz (NOT 3!)
    "gas_sensor_array_low_concentration": 1,  # UCI: 1 Hz
    "gas_sensors_for_home_activity_monitoring": 1,  # UCI: 1 sample/sec
    "alcohol_qcm_sensor_dataset": None,       # Feature vectors, not time series
    "gas_sensor_array_drift_dataset_at_different_concentrations": None,  # 128-dim feature vectors
}

# Dataset collection type: continuous vs discrete
# Continuous datasets have long recordings with dynamic concentration changes
# Discrete datasets have independent experiments with fixed conditions
DATASET_COLLECTION_TYPE: Dict[str, str] = {
    # Continuous: long recordings, segmented by concentration changes
    "gas_sensor_array_under_dynamic_gas_mixtures": "continuous",  # ~12h, segment_on_change
    "gas_sensor_array_temperature_modulation": "continuous",      # ~25h, segment_by_concentration
    "gas_sensors_for_home_activity_monitoring": "continuous",     # per-induction segmented
    # Discrete: independent experiments
    "twin_gas_sensor_arrays": "discrete",                         # 600s × 640 experiments
    "gas_sensor_array_exposed_to_turbulent_gas_mixtures": "discrete",  # 300s × 180 experiments
    "gas_sensor_array_low_concentration": "discrete",             # 15min × 90 samples
    "gas_sensor_array_under_flow_modulation": "discrete",         # 5min × 58 experiments
    "smellnet_pure": "discrete",                                  # 10min × 6 repeats/substance @ 1Hz
    "smellnet_mixture": "discrete",                               # 10min × 6 repeats/mixture @ 10Hz
    # Feature vectors (not time series)
    "alcohol_qcm_sensor_dataset": "features",
    "gas_sensor_array_drift_dataset_at_different_concentrations": "features",
}

# Dataset response type: resistance vs conductance
# CRITICAL for correct interpretation: 
#   - resistance: 值↑ = 浓度↓ (还原性气体)
#   - conductance: 值↑ = 浓度↑ (还原性气体)
DATASET_RESPONSE_TYPE: Dict[str, str] = {
    # 电阻类 (Rs, 值大=浓度低)
    "twin_gas_sensor_arrays": "resistance",                    # Rs (KOhm)
    "gas_sensor_array_temperature_modulation": "resistance",   # Rs (MOhm)
    "gas_sensors_for_home_activity_monitoring": "resistance",  # Rs (KOhm)
    "gas_sensor_array_under_flow_modulation": "resistance",    # dR (resistance change)
    # 电导类 (S/G, 值大=浓度高)
    "gas_sensor_array_under_dynamic_gas_mixtures": "conductance",  # S_i, 需 Rs=40/S 转换
    "gas_sensor_array_exposed_to_turbulent_gas_mixtures": "conductance",  # ADC, 需 Rs=10*(3110-A)/A
    # 比值类 (Rs/R0)
    "smellnet_pure": "resistance_ratio",       # Rs/R0, Grove V2
    "smellnet_mixture": "resistance_ratio",    # Rs/R0, Grove V2
    "gas_sensor_array_low_concentration": "resistance_ratio",  # 待确认
    # 特征向量 (非原始响应)
    "alcohol_qcm_sensor_dataset": "features",
    "gas_sensor_array_drift_dataset_at_different_concentrations": "features",
}


# =============================================================================
# 2. Universal Sensor Space (S_all)
# =============================================================================

@dataclass(frozen=True)
class SensorModel:
    """Global sensor model definition with target gas mapping."""
    id: int                          # Global unique index in S_all
    name: str                        # Sensor model name
    target_gases: FrozenSet[str]     # Set of target gases
    sensor_type: str                 # 'MOX', 'QCM', 'EC', etc.
    manufacturer: str                # Manufacturer name
    description: str = ""


# Global sensor model registry (M = |S_all| = 29 unique models)
# Index 0 reserved for unknown/generic sensors
SENSOR_MODELS: Tuple[SensorModel, ...] = (
    # Index 0: Generic/Unknown
    SensorModel(0, "UNKNOWN", frozenset(), "UNKNOWN", "", "Unknown sensor type"),
    
    # Figaro TGS series (MOX) - indices 1-12
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
    
    # FIS sensors - index 12
    SensorModel(12, "SB-500-12", frozenset({"CO"}), "MOX", "FIS", "CO detection"),
    
    # MQ series (generic MOX) - indices 13-17
    SensorModel(13, "MQ-3", frozenset({"Alcohol"}), "MOX", "Generic", "Alcohol sensor"),
    SensorModel(14, "MQ-5", frozenset({"LPG"}), "MOX", "Generic", "LPG/natural gas"),
    SensorModel(15, "MQ-9", frozenset({"CO"}), "MOX", "Generic", "CO sensor"),
    SensorModel(16, "MQ-135", frozenset({"NH3", "NOx", "Benzene"}), "MOX", "Generic", "Air quality"),
    SensorModel(17, "MQ-137", frozenset({"Ammonia"}), "MOX", "Generic", "Ammonia sensor"),
    SensorModel(18, "MQ-138", frozenset({"Alcohol", "Benzene"}), "MOX", "Generic", "Organic gas"),
    
    # Grove Multichannel Gas Sensor V2 (Seeed) - indices 19-22
    SensorModel(19, "GM-102B", frozenset({"NO2"}), "MOX", "Seeed", "NO2 sensor (Grove V2)"),
    SensorModel(20, "GM-302B", frozenset({"Ethanol"}), "MOX", "Seeed", "Ethanol sensor (Grove V2)"),
    SensorModel(21, "GM-502B", frozenset({"VOC"}), "MOX", "Seeed", "VOC sensor (Grove V2)"),
    SensorModel(22, "GM-702B", frozenset({"CO"}), "MOX", "Seeed", "CO sensor (Grove V2)"),
    
    # Electrochemical sensors - indices 23-25
    SensorModel(23, "2M012", frozenset({"Formaldehyde"}), "EC", "Generic", "Formaldehyde"),
    SensorModel(24, "2SH12", frozenset({"H2S"}), "EC", "Generic", "H2S sensor"),
    SensorModel(25, "VOCS-P", frozenset({"VOC"}), "PID", "Generic", "VOC PID sensor"),
    
    # QCM sensors (Quartz Crystal Microbalance) - indices 26-30
    SensorModel(26, "QCM3", frozenset({"Alcohol"}), "QCM", "Generic", "MIP:NP=1:1"),
    SensorModel(27, "QCM6", frozenset({"Alcohol"}), "QCM", "Generic", "MIP:NP=1:0"),
    SensorModel(28, "QCM7", frozenset({"Alcohol"}), "QCM", "Generic", "MIP:NP=1:0.5"),
    SensorModel(29, "QCM10", frozenset({"Alcohol"}), "QCM", "Generic", "MIP:NP=1:2"),
    SensorModel(30, "QCM12", frozenset({"Alcohol"}), "QCM", "Generic", "MIP:NP=0:1"),
    
    # Generic MOX placeholder for datasets without specific model info - index 31
    SensorModel(31, "MOX", frozenset(), "MOX", "Generic", "Generic MOX sensor"),
)

# Quick lookup: sensor name -> global index
SENSOR_NAME_TO_ID: Dict[str, int] = {s.name: s.id for s in SENSOR_MODELS}

# Total dimension of universal sensor space
M_TOTAL: int = len(SENSOR_MODELS)  # 30 (including UNKNOWN at index 0)


def get_sensor_id(name: str) -> int:
    """Get global sensor index by name. Returns 0 (UNKNOWN) if not found."""
    return SENSOR_NAME_TO_ID.get(name, 0)


def get_sensor_model(name: str) -> SensorModel:
    """Get sensor model by name. Returns UNKNOWN if not found."""
    idx = get_sensor_id(name)
    return SENSOR_MODELS[idx]


# =============================================================================
# 3. Universal Target Gas Space (G_all)
# =============================================================================

class TargetGas(IntEnum):
    """Global target gas enumeration."""
    UNKNOWN = 0
    ALCOHOL = auto()
    AMMONIA = auto()
    BENZENE = auto()
    BUTANE = auto()
    CO = auto()
    COMBUSTIBLE = auto()
    ETHANOL = auto()
    FORMALDEHYDE = auto()
    FREON = auto()
    H2S = auto()
    HYDROGEN = auto()
    LPG = auto()
    METHANE = auto()
    NH3 = auto()
    NO2 = auto()
    NOX = auto()
    PROPANE = auto()
    VOC = auto()


# String to enum mapping
GAS_NAME_TO_ENUM: Dict[str, TargetGas] = {
    "Alcohol": TargetGas.ALCOHOL,
    "Ammonia": TargetGas.AMMONIA,
    "Benzene": TargetGas.BENZENE,
    "Butane": TargetGas.BUTANE,
    "CO": TargetGas.CO,
    "Combustible": TargetGas.COMBUSTIBLE,
    "Ethanol": TargetGas.ETHANOL,
    "Formaldehyde": TargetGas.FORMALDEHYDE,
    "Freon": TargetGas.FREON,
    "H2S": TargetGas.H2S,
    "Hydrogen": TargetGas.HYDROGEN,
    "LPG": TargetGas.LPG,
    "Methane": TargetGas.METHANE,
    "NH3": TargetGas.NH3,
    "NO2": TargetGas.NO2,
    "NOx": TargetGas.NOX,
    "Propane": TargetGas.PROPANE,
    "VOC": TargetGas.VOC,
}


# =============================================================================
# 4. Universal Task Space (L_all)
# =============================================================================

class TaskType(IntEnum):
    """Global task type enumeration."""
    CLASSIFICATION = 0           # Multi-class classification (gas type, activity)
    CONCENTRATION_REGRESSION = 1  # Concentration prediction (continuous)
    DRIFT_COMPENSATION = 2       # Domain adaptation across time
    TRANSFER_LEARNING = 3        # Cross-sensor/cross-condition adaptation
    ACTIVITY_RECOGNITION = 4     # Activity/event classification


@dataclass(frozen=True)
class TaskDefinition:
    """Definition for a standardized task."""
    task_type: TaskType
    name: str
    description: str
    # For classification: number of classes
    # For regression: tuple of (unit, min_value, max_value)
    label_space: any = None


# Global task registry
TASKS: Dict[str, TaskDefinition] = {
    "classification": TaskDefinition(
        TaskType.CLASSIFICATION,
        "Gas Classification",
        "Classify gas type from sensor response",
    ),
    "concentration_prediction": TaskDefinition(
        TaskType.CONCENTRATION_REGRESSION,
        "Concentration Regression",
        "Predict gas concentration from sensor response",
        label_space=("ppm", 0, 10000),  # Standard unit: ppm
    ),
    "drift_compensation": TaskDefinition(
        TaskType.DRIFT_COMPENSATION,
        "Drift Compensation",
        "Adapt model to sensor drift over time",
    ),
    "transfer_learning": TaskDefinition(
        TaskType.TRANSFER_LEARNING,
        "Transfer Learning",
        "Transfer knowledge across sensors/conditions",
    ),
    "activity_recognition": TaskDefinition(
        TaskType.ACTIVITY_RECOGNITION,
        "Activity Recognition",
        "Recognize human activities from sensor data",
    ),
}


# =============================================================================
# 5. Unit Conversion (for target domain normalization)
# =============================================================================

# Standard concentration unit: ppm (parts per million)
STANDARD_CONCENTRATION_UNIT = "ppm"

# Conversion factors to standard unit (ppm)
UNIT_TO_PPM: Dict[str, float] = {
    "ppm": 1.0,
    "ppb": 0.001,      # 1 ppb = 0.001 ppm
    "percent": 10000,  # 1% = 10000 ppm
    "%": 10000,
}


def convert_to_ppm(value: float, unit: str) -> float:
    """Convert concentration value to standard ppm unit."""
    factor = UNIT_TO_PPM.get(unit.lower(), 1.0)
    return value * factor


# =============================================================================
# 6. Global Space Summary
# =============================================================================

@dataclass(frozen=True)
class GlobalSpace:
    """Global reference space Ω_global for dataset standardization.
    
    Attributes:
        f_std: Unified sampling rate in Hz
        sensor_space_dim: Dimension of universal sensor space |S_all|
        gas_space_dim: Dimension of target gas space |G_all|
        task_types: Available task types
    """
    f_std: int = F_STD
    sensor_space_dim: int = M_TOTAL
    gas_space_dim: int = len(TargetGas)
    task_types: Tuple[str, ...] = tuple(TASKS.keys())
    
    def get_sensor_indices(self, sensor_names: List[str]) -> List[int]:
        """Map sensor names to global indices."""
        return [get_sensor_id(name) for name in sensor_names]
    
    def get_gas_indices(self, gas_names: List[str]) -> List[int]:
        """Map gas names to global indices."""
        return [GAS_NAME_TO_ENUM.get(name, TargetGas.UNKNOWN).value for name in gas_names]


# Singleton instance
OMEGA_GLOBAL = GlobalSpace()


# =============================================================================
# 7. Dataset-to-Global Mapping Tables
# =============================================================================

# Mapping from each dataset's channel indices to global sensor indices
# This enables sensor space alignment: project X_i to S_all
DATASET_CHANNEL_TO_GLOBAL: Dict[str, List[int]] = {
    "twin_gas_sensor_arrays": [
        get_sensor_id("TGS2611"),  # ch0
        get_sensor_id("TGS2612"),  # ch1
        get_sensor_id("TGS2610"),  # ch2
        get_sensor_id("TGS2602"),  # ch3
        get_sensor_id("TGS2611"),  # ch4 (different heater voltage)
        get_sensor_id("TGS2612"),  # ch5
        get_sensor_id("TGS2610"),  # ch6
        get_sensor_id("TGS2602"),  # ch7
    ],
    "gas_sensor_array_under_dynamic_gas_mixtures": [
        get_sensor_id("TGS2602"),  # ch0-1
        get_sensor_id("TGS2602"),
        get_sensor_id("TGS2600"),  # ch2-3
        get_sensor_id("TGS2600"),
        get_sensor_id("TGS2610"),  # ch4-5
        get_sensor_id("TGS2610"),
        get_sensor_id("TGS2620"),  # ch6-7
        get_sensor_id("TGS2620"),
        get_sensor_id("TGS2602"),  # ch8-9
        get_sensor_id("TGS2602"),
        get_sensor_id("TGS2600"),  # ch10-11
        get_sensor_id("TGS2600"),
        get_sensor_id("TGS2610"),  # ch12-13
        get_sensor_id("TGS2610"),
        get_sensor_id("TGS2620"),  # ch14-15
        get_sensor_id("TGS2620"),
    ],
    "gas_sensor_array_exposed_to_turbulent_gas_mixtures": [
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
        get_sensor_id("MQ-135"),
        get_sensor_id("MQ-137"),
        get_sensor_id("MQ-138"),
        get_sensor_id("2M012"),
        get_sensor_id("VOCS-P"),
        get_sensor_id("2SH12"),
    ],
    "gas_sensor_array_temperature_modulation": [
        get_sensor_id("TGS3870-A04")] * 7 + [get_sensor_id("SB-500-12")] * 7,
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
    # SmellNet: Grove Multichannel V2 (4ch) + MQ-135 + MQ-9 + 6 more (12ch total per paper)
    # CSV order: NO2, C2H5OH, VOC, CO, Alcohol, LPG, Benzene, Temp, Pressure, Humidity, Gas_Resistance, Altitude
    # First 6 channels are gas sensors, rest are environmental
    "smellnet_pure": [
        get_sensor_id("GM-102B"),   # NO2
        get_sensor_id("GM-302B"),   # C2H5OH (Ethanol)
        get_sensor_id("GM-502B"),   # VOC
        get_sensor_id("GM-702B"),   # CO
        get_sensor_id("MQ-135"),    # Alcohol
        get_sensor_id("MQ-9"),      # LPG
    ],
    "smellnet_mixture": [
        get_sensor_id("GM-102B"),   # NO2
        get_sensor_id("GM-302B"),   # C2H5OH (Ethanol)
        get_sensor_id("GM-502B"),   # VOC
        get_sensor_id("GM-702B"),   # CO
        get_sensor_id("MQ-135"),    # Alcohol
        get_sensor_id("MQ-9"),      # LPG
    ],
    "alcohol_qcm_sensor_dataset": [
        get_sensor_id("QCM3"),
        get_sensor_id("QCM6"),
        get_sensor_id("QCM7"),
        get_sensor_id("QCM10"),
        get_sensor_id("QCM12"),
    ],
    # Drift dataset: TGS2600×4, TGS2602×4, TGS2610×4, TGS2620×4 (per paper)
    "gas_sensor_array_drift_dataset_at_different_concentrations": [
        get_sensor_id("TGS2600")] * 4 + [get_sensor_id("TGS2602")] * 4 + 
        [get_sensor_id("TGS2610")] * 4 + [get_sensor_id("TGS2620")] * 4,
    # Flow modulation: 5 TGS models, 2 voltages (5V/3.3V), per Table 1 in paper
    "gas_sensor_array_under_flow_modulation": [
        get_sensor_id("TGS2610"),  # R1: 5.0V, 21kΩ
        get_sensor_id("TGS2610"),  # R2: 5.0V, 21kΩ
        get_sensor_id("TGS2602"),  # R3: 5.0V, 21kΩ
        get_sensor_id("TGS2600"),  # R4: 5.0V, 21kΩ
        get_sensor_id("TGS2610"),  # R5: 5.0V, 21kΩ
        get_sensor_id("TGS2611"),  # R6: 5.0V, 21kΩ
        get_sensor_id("TGS2610"),  # R7: 5.0V, 21kΩ
        get_sensor_id("TGS2620"),  # R8: 5.0V, 21kΩ
        get_sensor_id("TGS2610"),  # R9: 3.3V, 82kΩ
        get_sensor_id("TGS2620"),  # R10: 3.3V, 82kΩ
        get_sensor_id("TGS2602"),  # R11: 3.3V, 82kΩ
        get_sensor_id("TGS2611"),  # R12: 3.3V, 82kΩ
        get_sensor_id("TGS2610"),  # R13: 3.3V, 82kΩ
        get_sensor_id("TGS2610"),  # R14: 3.3V, 82kΩ
        get_sensor_id("TGS2610"),  # R15: 3.3V, 82kΩ
        get_sensor_id("TGS2600"),  # R16: 3.3V, 82kΩ
    ],
}


def get_global_channel_mapping(dataset_name: str) -> List[int]:
    """Get global sensor indices for a dataset's channels.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        List of global sensor indices corresponding to each channel
    """
    return DATASET_CHANNEL_TO_GLOBAL.get(dataset_name, [])


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constants
    "F_STD",
    "M_TOTAL",
    "STANDARD_CONCENTRATION_UNIT",
    # Sensor space
    "SensorModel",
    "SENSOR_MODELS",
    "SENSOR_NAME_TO_ID",
    "get_sensor_id",
    "get_sensor_model",
    # Gas space
    "TargetGas",
    "GAS_NAME_TO_ENUM",
    # Task space
    "TaskType",
    "TaskDefinition",
    "TASKS",
    # Unit conversion
    "UNIT_TO_PPM",
    "convert_to_ppm",
    # Global space
    "GlobalSpace",
    "OMEGA_GLOBAL",
    # Mappings
    "DATASET_SAMPLE_RATES",
    "DATASET_COLLECTION_TYPE",
    "DATASET_RESPONSE_TYPE",
    "DATASET_CHANNEL_TO_GLOBAL",
    "get_global_channel_mapping",
]
