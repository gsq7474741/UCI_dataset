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

NOTE: Sensor models, dataset metadata, and channel mappings are now generated
from YAML schemas. See schemas/ directory and run:
    python scripts/generate_metadata.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union

import numpy as np

# =============================================================================
# IMPORT FROM GENERATED CODE (from schemas/*.yaml)
# =============================================================================
# These are auto-generated from YAML schemas - do not modify here!
# To change sensor/dataset definitions, edit schemas/*.yaml and regenerate.
# Auto-regeneration is triggered on import if schemas are newer.
from .schema import _auto_generate  # noqa: F401 - triggers auto-regeneration
from .schema._generated import (
    # Sensor models
    SensorModel,
    SENSOR_MODELS,
    SENSOR_NAME_TO_ID,
    M_TOTAL,
    get_sensor_id,
    get_sensor_model,
    # Dataset metadata
    DATASET_SAMPLE_RATES,
    DATASET_COLLECTION_TYPE,
    DATASET_RESPONSE_TYPE,
    DATASET_CHANNEL_TO_GLOBAL,
    get_global_channel_mapping,
    # Gas labels
    GAS_LABEL_TO_ID,
    GAS_LABEL_ID_TO_NAME,
    GAS_LABEL_MAPPINGS,
    get_global_gas_label,
)


# =============================================================================
# 1. Unified Sampling Rate (f_std)
# =============================================================================

# Target sampling rate for temporal standardization
# Chosen as the lowest common rate that preserves information from most datasets
# Available rates: 100, 50, 25, 10, 3, 1 Hz
# Using 10 Hz as balance between information preservation and computational efficiency
F_STD: int = 10  # Hz

# =============================================================================
# 2. Universal Target Gas Space (G_all) - MANUAL DEFINITIONS
# =============================================================================
# These enums require manual curation and are NOT auto-generated.

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


# -----------------------------------------------------------------------------
# 4.1 Classification Task: Global Label Spaces
# -----------------------------------------------------------------------------

class GasLabel(IntEnum):
    """Global gas type classification labels.
    
    Unified label space for gas classification across all datasets.
    Maps dataset-local gas indices to global indices.
    """
    UNKNOWN = 0
    # Common gases
    CO = auto()              # Carbon monoxide
    ETHANOL = auto()         # Ethanol / Ethyl alcohol (Ea)
    ETHYLENE = auto()        # Ethylene (C2H4) (Ey)
    METHANE = auto()         # Methane (CH4) (Me)
    AMMONIA = auto()         # Ammonia (NH3)
    ACETONE = auto()         # Acetone
    TOLUENE = auto()         # Toluene
    BENZENE = auto()         # Benzene
    FORMALDEHYDE = auto()    # Formaldehyde
    HYDROGEN = auto()        # Hydrogen (H2)
    PROPANE = auto()         # Propane
    BUTANE = auto()          # Butane
    NO2 = auto()             # Nitrogen dioxide
    H2S = auto()             # Hydrogen sulfide
    # Alcohols (for QCM dataset)
    ISOPROPANOL = auto()     # 2-propanol / Isopropanol
    PROPANOL = auto()        # 1-Propanol
    OCTANOL = auto()         # 1-Octanol
    BUTANOL = auto()         # 2-Butanol
    ISOBUTANOL = auto()      # 1-Isobutanol
    # VOCs
    ETHYL_ACETATE = auto()   # Ethyl acetate
    N_HEXANE = auto()        # n-Hexane
    ACETALDEHYDE = auto()    # Acetaldehyde
    # Mixture types
    MIXTURE_ETHYLENE_CO = auto()
    MIXTURE_ETHYLENE_METHANE = auto()
    MIXTURE_ACETONE_ETHANOL = auto()
    BACKGROUND = auto()      # Clean air / no gas


class ActivityLabel(IntEnum):
    """Global activity classification labels for home monitoring."""
    UNKNOWN = 0
    BACKGROUND = auto()      # Normal background
    WINE = auto()            # Wine opening/pouring
    BANANA = auto()          # Banana presence
    LEMON = auto()           # Lemon presence
    ONION = auto()           # Onion cutting
    GARLIC = auto()          # Garlic presence
    COOKING = auto()         # General cooking activity
    COFFEE = auto()          # Coffee brewing


class IngredientLabel(IntEnum):
    """Global ingredient labels for SmellNet (50 classes).
    
    Categories: nuts, spices, herbs, fruits, vegetables
    """
    UNKNOWN = 0
    # Nuts
    ALLSPICE = auto()
    ALMOND = auto()
    BRAZIL_NUT = auto()
    CASHEW = auto()
    CHESTNUT = auto()
    HAZELNUT = auto()
    MACADAMIA = auto()
    PECAN = auto()
    PISTACHIO = auto()
    WALNUT = auto()
    # Spices
    ANGELICA = auto()
    CARDAMOM = auto()
    CINNAMON = auto()
    CLOVE = auto()
    CORIANDER = auto()
    CUMIN = auto()
    GINGER = auto()
    NUTMEG = auto()
    PEPPER = auto()
    STAR_ANISE = auto()
    # Herbs
    BASIL = auto()
    CHIVE = auto()
    DILL = auto()
    MINT = auto()
    OREGANO = auto()
    PARSLEY = auto()
    ROSEMARY = auto()
    SAGE = auto()
    TARRAGON = auto()
    THYME = auto()
    # Fruits
    APPLE = auto()
    BANANA = auto()
    GRAPE = auto()
    KIWI = auto()
    LEMON = auto()
    MANGO = auto()
    ORANGE = auto()
    PEACH = auto()
    PEAR = auto()
    STRAWBERRY = auto()
    # Vegetables
    ASPARAGUS = auto()
    AVOCADO = auto()
    BROCCOLI = auto()
    BRUSSEL_SPROUTS = auto()
    CABBAGE = auto()
    CARROT = auto()
    CELERY = auto()
    CUCUMBER = auto()
    SPINACH = auto()
    TOMATO = auto()


# Dataset-local to global label mappings
GAS_LABEL_MAPPINGS: Dict[str, Dict[int, GasLabel]] = {
    "twin_gas_sensor_arrays": {
        # Local labels: 0=Ea(Ethanol), 1=CO, 2=Ey(Ethylene), 3=Me(Methane)
        0: GasLabel.ETHANOL,
        1: GasLabel.CO,
        2: GasLabel.ETHYLENE,
        3: GasLabel.METHANE,
    },
    "gas_sensor_array_under_dynamic_gas_mixtures": {
        # Local: mixture = 0 if "ethylene_co" else 1
        # 0=Ethylene+CO mixture, 1=Ethylene+Methane mixture
        0: GasLabel.MIXTURE_ETHYLENE_CO,
        1: GasLabel.MIXTURE_ETHYLENE_METHANE,
    },
    "gas_sensor_array_exposed_to_turbulent_gas_mixtures": {
        # Classification by target["second_gas"]: 0=Methane, 1=CO
        # All samples are Ethylene mixed with either Methane or CO
        0: GasLabel.MIXTURE_ETHYLENE_METHANE,
        1: GasLabel.MIXTURE_ETHYLENE_CO,
    },
    "gas_sensor_array_low_concentration": {
        # Local: 0=ethanol, 1=acetone, 2=toluene, 3=ethyl_acetate, 4=isopropanol, 5=n_hexane
        0: GasLabel.ETHANOL,
        1: GasLabel.ACETONE,
        2: GasLabel.TOLUENE,
        3: GasLabel.ETHYL_ACETATE,
        4: GasLabel.ISOPROPANOL,
        5: GasLabel.N_HEXANE,
    },
    "gas_sensor_array_under_flow_modulation": {
        # Local: gas_classes = ["air", "acetone", "ethanol", "mixture"]
        0: GasLabel.BACKGROUND,  # air
        1: GasLabel.ACETONE,
        2: GasLabel.ETHANOL,
        3: GasLabel.MIXTURE_ACETONE_ETHANOL,
    },
    "alcohol_qcm_sensor_dataset": {
        # Local: classes = ["1-octanol", "1-propanol", "2-butanol", "2-propanol", "1-isobutanol"]
        0: GasLabel.OCTANOL,
        1: GasLabel.PROPANOL,
        2: GasLabel.BUTANOL,
        3: GasLabel.ISOPROPANOL,  # 2-propanol = isopropanol
        4: GasLabel.ISOBUTANOL,
    },
    "gas_sensor_array_drift_dataset_at_different_concentrations": {
        # Local: classes = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
        # Original file labels 1-6 mapped to 0-5
        0: GasLabel.ETHANOL,
        1: GasLabel.ETHYLENE,
        2: GasLabel.AMMONIA,
        3: GasLabel.ACETALDEHYDE,
        4: GasLabel.ACETONE,
        5: GasLabel.TOLUENE,
    },
}

ACTIVITY_LABEL_MAPPINGS: Dict[str, Dict[int, ActivityLabel]] = {
    "gas_sensors_for_home_activity_monitoring": {
        0: ActivityLabel.BACKGROUND,
        1: ActivityLabel.WINE,
        2: ActivityLabel.BANANA,
        # Additional activities can be added based on dataset documentation
    },
}


# -----------------------------------------------------------------------------
# 4.2 Regression Task: Concentration Normalization
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ConcentrationConfig:
    """Configuration for concentration regression task.
    
    Attributes:
        unit: Original unit in dataset
        to_ppm_factor: Multiplication factor to convert to ppm
        min_value: Minimum value in original unit
        max_value: Maximum value in original unit
        log_scale: Whether to use log scale for normalization
    """
    unit: str
    to_ppm_factor: float
    min_value: float
    max_value: float
    log_scale: bool = False


CONCENTRATION_CONFIGS: Dict[str, ConcentrationConfig] = {
    "twin_gas_sensor_arrays": ConcentrationConfig(
        unit="ppm", to_ppm_factor=1.0, min_value=12.5, max_value=250.0
    ),
    "gas_sensor_array_under_dynamic_gas_mixtures": ConcentrationConfig(
        unit="ppm", to_ppm_factor=1.0, min_value=0.0, max_value=300.0
    ),
    "gas_sensor_array_exposed_to_turbulent_gas_mixtures": ConcentrationConfig(
        unit="ppm", to_ppm_factor=1.0, min_value=0.0, max_value=300.0
    ),
    "gas_sensor_array_low_concentration": ConcentrationConfig(
        unit="ppb", to_ppm_factor=0.001, min_value=50.0, max_value=200.0,
        log_scale=True  # Low concentration benefits from log scale
    ),
    "gas_sensor_array_temperature_modulation": ConcentrationConfig(
        unit="ppm", to_ppm_factor=1.0, min_value=0.0, max_value=20.0
    ),
}


def normalize_concentration(value: float, config: ConcentrationConfig) -> float:
    """Normalize concentration to [0, 1] range.
    
    Args:
        value: Concentration in original unit
        config: Concentration configuration
        
    Returns:
        Normalized value in [0, 1]
    """
    ppm = value * config.to_ppm_factor
    if config.log_scale and ppm > 0:
        import math
        min_log = math.log1p(config.min_value * config.to_ppm_factor)
        max_log = math.log1p(config.max_value * config.to_ppm_factor)
        return (math.log1p(ppm) - min_log) / (max_log - min_log + 1e-8)
    else:
        min_ppm = config.min_value * config.to_ppm_factor
        max_ppm = config.max_value * config.to_ppm_factor
        return (ppm - min_ppm) / (max_ppm - min_ppm + 1e-8)


# -----------------------------------------------------------------------------
# 4.3 Drift Compensation Task: Domain Definition
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class DriftDomainConfig:
    """Configuration for drift compensation / domain adaptation.
    
    Attributes:
        domain_field: Field name in target dict for domain ID
        num_domains: Total number of domains (time batches)
        source_domains: Domain IDs for source (training)
        target_domains: Domain IDs for target (adaptation)
    """
    domain_field: str
    num_domains: int
    source_domains: Tuple[int, ...]
    target_domains: Tuple[int, ...]


DRIFT_CONFIGS: Dict[str, DriftDomainConfig] = {
    "gas_sensor_array_drift_dataset_at_different_concentrations": DriftDomainConfig(
        domain_field="batch",
        num_domains=10,
        source_domains=(1, 2, 3),        # First 3 batches as source
        target_domains=(4, 5, 6, 7, 8, 9, 10),  # Later batches as target
    ),
    "twin_gas_sensor_arrays": DriftDomainConfig(
        domain_field="repeat",
        num_domains=10,
        source_domains=(1, 2, 3, 4, 5),
        target_domains=(6, 7, 8, 9, 10),
    ),
}


# -----------------------------------------------------------------------------
# 4.4 Transfer Learning Task: Source/Target Definition
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class TransferConfig:
    """Configuration for transfer learning across sensors/conditions.
    
    Attributes:
        transfer_field: Field name for source/target split
        source_values: Values indicating source domain
        target_values: Values indicating target domain
    """
    transfer_field: str
    source_values: Tuple[any, ...]
    target_values: Tuple[any, ...]


TRANSFER_CONFIGS: Dict[str, TransferConfig] = {
    "twin_gas_sensor_arrays": TransferConfig(
        transfer_field="board",
        source_values=(1,),    # Board 1 as source
        target_values=(2,),    # Board 2 as target
    ),
}


# -----------------------------------------------------------------------------
# 4.5 Task Definition (Enhanced)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskDefinition:
    """Enhanced task definition with structured label space.
    
    Attributes:
        task_type: Type of task (classification, regression, etc.)
        name: Human-readable task name
        description: Task description
        label_enum: Enum class for classification labels (if applicable)
        regression_config: Configuration for regression tasks (if applicable)
        drift_config: Configuration for drift compensation (if applicable)
        transfer_config: Configuration for transfer learning (if applicable)
        metrics: Evaluation metrics for this task
    """
    task_type: TaskType
    name: str
    description: str
    label_enum: Optional[type] = None
    regression_config: Optional[Dict] = None
    drift_config: Optional[Dict] = None
    transfer_config: Optional[Dict] = None
    metrics: Tuple[str, ...] = ()


# Global task registry (enhanced)
TASKS: Dict[str, TaskDefinition] = {
    "classification": TaskDefinition(
        task_type=TaskType.CLASSIFICATION,
        name="Gas Classification",
        description="Classify gas type from sensor response",
        label_enum=GasLabel,
        metrics=("accuracy", "f1_macro", "confusion_matrix"),
    ),
    "concentration_prediction": TaskDefinition(
        task_type=TaskType.CONCENTRATION_REGRESSION,
        name="Concentration Regression",
        description="Predict gas concentration (ppm) from sensor response",
        regression_config=CONCENTRATION_CONFIGS,
        metrics=("mae", "rmse", "r2", "mape"),
    ),
    "drift_compensation": TaskDefinition(
        task_type=TaskType.DRIFT_COMPENSATION,
        name="Drift Compensation",
        description="Adapt model to sensor drift over time",
        drift_config=DRIFT_CONFIGS,
        metrics=("source_acc", "target_acc", "adaptation_gain"),
    ),
    "transfer_learning": TaskDefinition(
        task_type=TaskType.TRANSFER_LEARNING,
        name="Transfer Learning",
        description="Transfer knowledge across different sensor boards",
        transfer_config=TRANSFER_CONFIGS,
        metrics=("source_acc", "target_acc", "transfer_ratio"),
    ),
    "activity_recognition": TaskDefinition(
        task_type=TaskType.ACTIVITY_RECOGNITION,
        name="Activity Recognition",
        description="Recognize human activities from gas sensor data",
        label_enum=ActivityLabel,
        metrics=("accuracy", "f1_macro", "confusion_matrix"),
    ),
    "ingredient_classification": TaskDefinition(
        task_type=TaskType.CLASSIFICATION,
        name="Ingredient Classification",
        description="Classify food ingredients from smell sensor data (SmellNet)",
        label_enum=IngredientLabel,
        metrics=("accuracy", "f1_macro", "top5_accuracy"),
    ),
}


def get_global_label(dataset_name: str, local_label: int, task: str = "classification") -> int:
    """Convert dataset-local label to global label index.
    
    Args:
        dataset_name: Name of the dataset
        local_label: Local label index from dataset
        task: Task type (classification, activity_recognition)
        
    Returns:
        Global label index
        
    Note:
        For turbulent_gas_mixtures, use target["second_gas"] as local_label.
    """
    if task == "activity_recognition":
        mapping = ACTIVITY_LABEL_MAPPINGS.get(dataset_name, {})
        return mapping.get(local_label, ActivityLabel.UNKNOWN).value
    else:
        mapping = GAS_LABEL_MAPPINGS.get(dataset_name, {})
        return mapping.get(local_label, GasLabel.UNKNOWN).value


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
# 7. Unified Multi-Label System (统一多标签系统)
# =============================================================================

class SubstanceType(IntEnum):
    """Unified substance type enumeration (gases + ingredients).
    
    统一物质类型枚举，包含气体和食材，去除混合物类型。
    混合物通过多标签向量表示，而非独立类别。
    
    Numbering scheme:
        0: UNKNOWN
        1-50: Gases (气体)
        51-100: Ingredients/Foods (食材)
    """
    UNKNOWN = 0
    
    # === Gases (1-50) ===
    # Common industrial/environmental gases
    CO = 1                   # Carbon monoxide
    ETHYLENE = 2             # Ethylene (C2H4)
    METHANE = 3              # Methane (CH4)
    ETHANOL = 4              # Ethanol / Ethyl alcohol
    ACETONE = 5              # Acetone
    AMMONIA = 6              # Ammonia (NH3)
    TOLUENE = 7              # Toluene
    BENZENE = 8              # Benzene
    FORMALDEHYDE = 9         # Formaldehyde
    HYDROGEN = 10            # Hydrogen (H2)
    PROPANE = 11             # Propane
    BUTANE = 12              # Butane
    NO2 = 13                 # Nitrogen dioxide
    H2S = 14                 # Hydrogen sulfide
    # Alcohols
    ISOPROPANOL = 15         # 2-propanol / Isopropanol
    PROPANOL = 16            # 1-Propanol
    OCTANOL = 17             # 1-Octanol
    BUTANOL = 18             # 2-Butanol
    ISOBUTANOL = 19          # 1-Isobutanol
    # VOCs
    ETHYL_ACETATE = 20       # Ethyl acetate
    N_HEXANE = 21            # n-Hexane
    ACETALDEHYDE = 22        # Acetaldehyde
    # Reserved for future gases (23-50)
    AIR = 50                 # Clean air / background
    
    # === Ingredients/Foods (51-100) ===
    # Nuts
    ALLSPICE = 51
    ALMOND = 52
    BRAZIL_NUT = 53
    CASHEW = 54
    CHESTNUT = 55
    HAZELNUT = 56
    MACADAMIA = 57
    PECAN = 58
    PISTACHIO = 59
    WALNUT = 60
    # Spices
    ANGELICA = 61
    CARDAMOM = 62
    CINNAMON = 63
    CLOVE = 64
    CORIANDER = 65
    CUMIN = 66
    GINGER = 67
    NUTMEG = 68
    PEPPER = 69
    STAR_ANISE = 70
    # Herbs
    BASIL = 71
    CHIVE = 72
    DILL = 73
    MINT = 74
    OREGANO = 75
    PARSLEY = 76
    ROSEMARY = 77
    SAGE = 78
    TARRAGON = 79
    THYME = 80
    # Fruits
    APPLE = 81
    BANANA = 82
    GRAPE = 83
    KIWI = 84
    LEMON = 85
    MANGO = 86
    ORANGE = 87
    PEACH = 88
    PEAR = 89
    STRAWBERRY = 90
    # Vegetables
    ASPARAGUS = 91
    AVOCADO = 92
    BROCCOLI = 93
    BRUSSEL_SPROUTS = 94
    CABBAGE = 95
    CARROT = 96
    CELERY = 97
    CUCUMBER = 98
    SPINACH = 99
    TOMATO = 100


# Total number of substance types (for vector dimensionality)
NUM_SUBSTANCES = 101


class LabelMode(Enum):
    """Label retrieval mode for different training tasks."""
    # Classification modes
    CLS_SINGLE = "cls_single"      # Single label (argmax): int
    CLS_MULTI = "cls_multi"        # Multi-label binary: [0,1,1,0...]
    CLS_SOFT = "cls_soft"          # Soft label (proportions): [0,0.3,0.7,0...]
    
    # Regression modes
    REG_PRIMARY = "reg_primary"    # Primary substance concentration: float
    REG_ALL = "reg_all"            # All concentrations: [0,50,100,0...] ppm
    REG_NORMALIZED = "reg_norm"    # Normalized concentrations: [0,0.1,0.2,0...]
    
    # Joint mode
    JOINT = "joint"                # (cls_multi, reg_norm) tuple


@dataclass
class SampleLabel:
    """Unified sample label supporting multi-label and concentration.
    
    统一样本标签，支持多标签分类和浓度回归。
    
    Attributes:
        presence: Multi-hot presence vector, shape (NUM_SUBSTANCES,)
                  Values: 0/1 for binary, or proportions (0~1) for mixtures
        concentration: Concentration vector in ppm, shape (NUM_SUBSTANCES,)
                       For SmellNet, stores proportion (0~1) instead of ppm
        dataset_name: Source dataset name for context
    """
    presence: np.ndarray           # shape: (NUM_SUBSTANCES,)
    concentration: np.ndarray      # shape: (NUM_SUBSTANCES,)
    dataset_name: str = ""
    
    def __post_init__(self):
        """Validate and convert arrays."""
        if not isinstance(self.presence, np.ndarray):
            self.presence = np.array(self.presence, dtype=np.float32)
        if not isinstance(self.concentration, np.ndarray):
            self.concentration = np.array(self.concentration, dtype=np.float32)
        # Ensure correct shape
        assert self.presence.shape == (NUM_SUBSTANCES,), f"Expected ({NUM_SUBSTANCES},), got {self.presence.shape}"
        assert self.concentration.shape == (NUM_SUBSTANCES,), f"Expected ({NUM_SUBSTANCES},), got {self.concentration.shape}"
    
    @property
    def is_pure(self) -> bool:
        """Check if sample contains only one substance."""
        return int((self.presence > 0).sum()) == 1
    
    @property
    def is_mixture(self) -> bool:
        """Check if sample is a mixture."""
        return int((self.presence > 0).sum()) > 1
    
    @property
    def primary(self) -> int:
        """Primary substance index (highest concentration)."""
        return int(np.argmax(self.concentration))
    
    @property
    def primary_type(self) -> SubstanceType:
        """Primary substance type."""
        return SubstanceType(self.primary)
    
    @property
    def active_indices(self) -> List[int]:
        """List of all present substance indices."""
        return np.where(self.presence > 0)[0].tolist()
    
    @property
    def active_types(self) -> List[SubstanceType]:
        """List of all present substance types."""
        return [SubstanceType(i) for i in self.active_indices]
    
    @property
    def num_components(self) -> int:
        """Number of components in the sample."""
        return int((self.presence > 0).sum())
    
    def get_label(self, mode: LabelMode) -> Union[int, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Get label in specified format for training.
        
        Args:
            mode: Label retrieval mode
            
        Returns:
            Label in the format specified by mode
        """
        if mode == LabelMode.CLS_SINGLE:
            return self.primary
        elif mode == LabelMode.CLS_MULTI:
            return (self.presence > 0).astype(np.float32)
        elif mode == LabelMode.CLS_SOFT:
            total = self.presence.sum()
            if total > 0:
                return (self.presence / total).astype(np.float32)
            return self.presence.astype(np.float32)
        elif mode == LabelMode.REG_PRIMARY:
            return float(self.concentration[self.primary])
        elif mode == LabelMode.REG_ALL:
            return self.concentration.astype(np.float32)
        elif mode == LabelMode.REG_NORMALIZED:
            return normalize_concentration_vector(self.concentration, self.dataset_name)
        elif mode == LabelMode.JOINT:
            cls_label = (self.presence > 0).astype(np.float32)
            reg_label = normalize_concentration_vector(self.concentration, self.dataset_name)
            return (cls_label, reg_label)
        else:
            raise ValueError(f"Unknown label mode: {mode}")
    
    @classmethod
    def from_single(
        cls, 
        substance: SubstanceType, 
        concentration: float = 1.0,
        dataset_name: str = ""
    ) -> "SampleLabel":
        """Create label for a single (pure) substance.
        
        Args:
            substance: Substance type
            concentration: Concentration in ppm (or proportion for SmellNet)
            dataset_name: Source dataset name
        """
        presence = np.zeros(NUM_SUBSTANCES, dtype=np.float32)
        conc = np.zeros(NUM_SUBSTANCES, dtype=np.float32)
        presence[substance.value] = 1.0
        conc[substance.value] = concentration
        return cls(presence=presence, concentration=conc, dataset_name=dataset_name)
    
    @classmethod
    def from_mixture(
        cls,
        components: List[Tuple[SubstanceType, float]],
        dataset_name: str = ""
    ) -> "SampleLabel":
        """Create label for a mixture.
        
        Args:
            components: List of (substance, concentration) tuples
            dataset_name: Source dataset name
        """
        presence = np.zeros(NUM_SUBSTANCES, dtype=np.float32)
        conc = np.zeros(NUM_SUBSTANCES, dtype=np.float32)
        for substance, concentration in components:
            presence[substance.value] = 1.0
            conc[substance.value] = concentration
        return cls(presence=presence, concentration=conc, dataset_name=dataset_name)


# =============================================================================
# 8.1 Concentration Range Configuration
# =============================================================================

# Concentration ranges per substance (min_ppm, max_ppm) for normalization
# These are derived from dataset documentation
SUBSTANCE_CONCENTRATION_RANGES: Dict[SubstanceType, Tuple[float, float]] = {
    # Gases - from UCI datasets
    SubstanceType.CO: (0, 600),           # twin: 12.5-250, turbulent: 0-460
    SubstanceType.ETHYLENE: (0, 100),     # twin: 12.5-250, dynamic: 0-20
    SubstanceType.METHANE: (0, 300),      # twin: 12.5-250, turbulent: 0-131
    SubstanceType.ETHANOL: (0, 600),      # drift: 10-600
    SubstanceType.ACETONE: (0, 1000),     # drift: 12-1000
    SubstanceType.AMMONIA: (0, 1000),     # drift: 50-1000
    SubstanceType.TOLUENE: (0, 100),      # drift: 10-100
    SubstanceType.ACETALDEHYDE: (0, 500), # drift: 5-500
    SubstanceType.ISOPROPANOL: (0, 200),  # low_conc: 50-200 ppb -> 0.05-0.2 ppm
    SubstanceType.PROPANOL: (0, 1),       # QCM: ratio
    SubstanceType.OCTANOL: (0, 1),        # QCM: ratio
    SubstanceType.BUTANOL: (0, 1),        # QCM: ratio
    SubstanceType.ISOBUTANOL: (0, 1),     # QCM: ratio
    SubstanceType.ETHYL_ACETATE: (0, 200),
    SubstanceType.N_HEXANE: (0, 200),
    # SmellNet ingredients - use proportion (0-1)
    SubstanceType.ALMOND: (0, 1),
    SubstanceType.BANANA: (0, 1),
    SubstanceType.ORANGE: (0, 1),
    # ... other ingredients default to (0, 1)
}

# Default range for substances not explicitly defined
DEFAULT_CONCENTRATION_RANGE = (0, 1)


def get_concentration_range(substance: SubstanceType) -> Tuple[float, float]:
    """Get concentration range for normalization."""
    return SUBSTANCE_CONCENTRATION_RANGES.get(substance, DEFAULT_CONCENTRATION_RANGE)


def normalize_concentration_vector(
    concentration: np.ndarray,
    dataset_name: str = ""
) -> np.ndarray:
    """Normalize concentration vector to [0, 1] range.
    
    Args:
        concentration: Raw concentration vector (NUM_SUBSTANCES,)
        dataset_name: Dataset name for context-specific normalization
        
    Returns:
        Normalized concentration vector (NUM_SUBSTANCES,)
    """
    normalized = np.zeros_like(concentration)
    for i in range(NUM_SUBSTANCES):
        if concentration[i] > 0:
            substance = SubstanceType(i)
            lo, hi = get_concentration_range(substance)
            if hi > lo:
                normalized[i] = np.clip((concentration[i] - lo) / (hi - lo), 0, 1)
            else:
                normalized[i] = concentration[i]  # Already normalized
    return normalized.astype(np.float32)


# =============================================================================
# 8.2 Dataset to SubstanceType Mappings
# =============================================================================

# Maps dataset-local label index to (SubstanceType, default_concentration)
# For mixtures, returns list of components
SUBSTANCE_MAPPINGS: Dict[str, Dict[int, Union[SubstanceType, List[Tuple[SubstanceType, float]]]]] = {
    "twin_gas_sensor_arrays": {
        # Local: 0=Ea, 1=CO, 2=Ey, 3=Me
        # Concentration from target["ppm"]
        0: SubstanceType.ETHANOL,
        1: SubstanceType.CO,
        2: SubstanceType.ETHYLENE,
        3: SubstanceType.METHANE,
    },
    "gas_sensor_array_under_dynamic_gas_mixtures": {
        # Local: 0=ethylene_co, 1=ethylene_methane
        # Always mixtures, concentrations from target["ethylene_ppm"], target["gas2_ppm"]
        0: [(SubstanceType.ETHYLENE, 0), (SubstanceType.CO, 0)],  # conc from target
        1: [(SubstanceType.ETHYLENE, 0), (SubstanceType.METHANE, 0)],
    },
    "gas_sensor_array_exposed_to_turbulent_gas_mixtures": {
        # Local second_gas: 0=Methane, 1=CO (mixed with Ethylene)
        # Concentrations from target["ethylene_ppm"], target["second_ppm"]
        0: [(SubstanceType.ETHYLENE, 0), (SubstanceType.METHANE, 0)],
        1: [(SubstanceType.ETHYLENE, 0), (SubstanceType.CO, 0)],
    },
    "gas_sensor_array_low_concentration": {
        # Local: 0-5 pure gases
        0: SubstanceType.ETHANOL,
        1: SubstanceType.ACETONE,
        2: SubstanceType.TOLUENE,
        3: SubstanceType.ETHYL_ACETATE,
        4: SubstanceType.ISOPROPANOL,
        5: SubstanceType.N_HEXANE,
    },
    "gas_sensor_array_under_flow_modulation": {
        # Local: 0=air, 1=acetone, 2=ethanol, 3=mixture
        0: SubstanceType.AIR,
        1: SubstanceType.ACETONE,
        2: SubstanceType.ETHANOL,
        3: [(SubstanceType.ACETONE, 0), (SubstanceType.ETHANOL, 0)],  # conc from target
    },
    "alcohol_qcm_sensor_dataset": {
        # Local: 0-4 alcohols
        0: SubstanceType.OCTANOL,
        1: SubstanceType.PROPANOL,
        2: SubstanceType.BUTANOL,
        3: SubstanceType.ISOPROPANOL,
        4: SubstanceType.ISOBUTANOL,
    },
    "gas_sensor_array_drift_dataset_at_different_concentrations": {
        # Local: 0-5 gases, concentration from target["ppm"]
        0: SubstanceType.ETHANOL,
        1: SubstanceType.ETHYLENE,
        2: SubstanceType.AMMONIA,
        3: SubstanceType.ACETALDEHYDE,
        4: SubstanceType.ACETONE,
        5: SubstanceType.TOLUENE,
    },
}

# SmellNet ingredient name to SubstanceType mapping
SMELLNET_INGREDIENT_TO_SUBSTANCE: Dict[str, SubstanceType] = {
    "allspice": SubstanceType.ALLSPICE,
    "almond": SubstanceType.ALMOND,
    "angelica": SubstanceType.ANGELICA,
    "apple": SubstanceType.APPLE,
    "asparagus": SubstanceType.ASPARAGUS,
    "avocado": SubstanceType.AVOCADO,
    "banana": SubstanceType.BANANA,
    "basil": SubstanceType.BASIL,
    "brazil_nut": SubstanceType.BRAZIL_NUT,
    "broccoli": SubstanceType.BROCCOLI,
    "brussel_sprouts": SubstanceType.BRUSSEL_SPROUTS,
    "cabbage": SubstanceType.CABBAGE,
    "cardamom": SubstanceType.CARDAMOM,
    "carrot": SubstanceType.CARROT,
    "cashew": SubstanceType.CASHEW,
    "celery": SubstanceType.CELERY,
    "chestnut": SubstanceType.CHESTNUT,
    "chive": SubstanceType.CHIVE,
    "cinnamon": SubstanceType.CINNAMON,
    "clove": SubstanceType.CLOVE,
    "coriander": SubstanceType.CORIANDER,
    "cucumber": SubstanceType.CUCUMBER,
    "cumin": SubstanceType.CUMIN,
    "dill": SubstanceType.DILL,
    "ginger": SubstanceType.GINGER,
    "grape": SubstanceType.GRAPE,
    "hazelnut": SubstanceType.HAZELNUT,
    "kiwi": SubstanceType.KIWI,
    "lemon": SubstanceType.LEMON,
    "macadamia": SubstanceType.MACADAMIA,
    "mango": SubstanceType.MANGO,
    "mint": SubstanceType.MINT,
    "nutmeg": SubstanceType.NUTMEG,
    "orange": SubstanceType.ORANGE,
    "oregano": SubstanceType.OREGANO,
    "parsley": SubstanceType.PARSLEY,
    "peach": SubstanceType.PEACH,
    "pear": SubstanceType.PEAR,
    "pecan": SubstanceType.PECAN,
    "pepper": SubstanceType.PEPPER,
    "pistachio": SubstanceType.PISTACHIO,
    "rosemary": SubstanceType.ROSEMARY,
    "sage": SubstanceType.SAGE,
    "spinach": SubstanceType.SPINACH,
    "star_anise": SubstanceType.STAR_ANISE,
    "strawberry": SubstanceType.STRAWBERRY,
    "tarragon": SubstanceType.TARRAGON,
    "thyme": SubstanceType.THYME,
    "tomato": SubstanceType.TOMATO,
    "walnut": SubstanceType.WALNUT,
}


def create_sample_label(
    dataset_name: str,
    local_label: int,
    concentration: Optional[float] = None,
    mixture_concentrations: Optional[Dict[str, float]] = None,
) -> SampleLabel:
    """Create SampleLabel from dataset-local label.
    
    Args:
        dataset_name: Name of the source dataset
        local_label: Local label index from dataset
        concentration: Concentration in ppm (for pure substances)
        mixture_concentrations: Dict of component concentrations (for mixtures)
        
    Returns:
        SampleLabel instance
        
    Example:
        # Pure gas
        label = create_sample_label("twin_gas_sensor_arrays", 0, concentration=100)
        
        # Mixture
        label = create_sample_label(
            "gas_sensor_array_under_dynamic_gas_mixtures", 0,
            mixture_concentrations={"ethylene_ppm": 50, "gas2_ppm": 100}
        )
    """
    mapping = SUBSTANCE_MAPPINGS.get(dataset_name, {})
    substance_info = mapping.get(local_label)
    
    if substance_info is None:
        # Unknown label
        return SampleLabel.from_single(SubstanceType.UNKNOWN, 0, dataset_name)
    
    if isinstance(substance_info, SubstanceType):
        # Pure substance
        conc = concentration if concentration is not None else 1.0
        return SampleLabel.from_single(substance_info, conc, dataset_name)
    
    elif isinstance(substance_info, list):
        # Mixture
        components = []
        for i, (substance, default_conc) in enumerate(substance_info):
            if mixture_concentrations:
                # Try to get concentration from dict
                if i == 0 and "ethylene_ppm" in mixture_concentrations:
                    conc = mixture_concentrations["ethylene_ppm"]
                elif i == 1 and "gas2_ppm" in mixture_concentrations:
                    conc = mixture_concentrations["gas2_ppm"]
                elif i == 0 and "ace_conc" in mixture_concentrations:
                    conc = mixture_concentrations["ace_conc"]
                elif i == 1 and "eth_conc" in mixture_concentrations:
                    conc = mixture_concentrations["eth_conc"]
                else:
                    conc = default_conc
            else:
                conc = default_conc
            components.append((substance, conc))
        return SampleLabel.from_mixture(components, dataset_name)
    
    return SampleLabel.from_single(SubstanceType.UNKNOWN, 0, dataset_name)


def create_smellnet_label(
    ingredients: Dict[str, float],
    dataset_name: str = "smellnet"
) -> SampleLabel:
    """Create SampleLabel from SmellNet ingredient proportions.
    
    Args:
        ingredients: Dict mapping ingredient name to proportion (0-1)
        dataset_name: Dataset name
        
    Returns:
        SampleLabel instance
        
    Example:
        label = create_smellnet_label({"banana": 0.5, "orange": 0.5})
    """
    presence = np.zeros(NUM_SUBSTANCES, dtype=np.float32)
    concentration = np.zeros(NUM_SUBSTANCES, dtype=np.float32)
    
    for name, proportion in ingredients.items():
        name_lower = name.lower().replace("-", "_").replace(" ", "_")
        substance = SMELLNET_INGREDIENT_TO_SUBSTANCE.get(name_lower)
        if substance:
            presence[substance.value] = proportion
            concentration[substance.value] = proportion
    
    return SampleLabel(presence=presence, concentration=concentration, dataset_name=dataset_name)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constants
    "F_STD",
    "M_TOTAL",
    "NUM_SUBSTANCES",
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
    # === Unified Multi-Label System ===
    # Substance types
    "SubstanceType",
    # Label modes
    "LabelMode",
    # Sample label
    "SampleLabel",
    # Concentration config
    "SUBSTANCE_CONCENTRATION_RANGES",
    "DEFAULT_CONCENTRATION_RANGE",
    "get_concentration_range",
    "normalize_concentration_vector",
    # Mappings
    "SUBSTANCE_MAPPINGS",
    "SMELLNET_INGREDIENT_TO_SUBSTANCE",
    # Factory functions
    "create_sample_label",
    "create_smellnet_label",
    # Legacy (backward compatibility)
    "GasLabel",
    "ActivityLabel",
    "IngredientLabel",
    "GAS_LABEL_MAPPINGS",
    "ACTIVITY_LABEL_MAPPINGS",
    "get_global_label",
]
