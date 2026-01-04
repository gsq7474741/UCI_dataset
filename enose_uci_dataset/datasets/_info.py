from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ExtractConfig:
    type: str = "standard"
    subdir: Optional[str] = None
    nested_zip: Optional[str] = None


@dataclass(frozen=True)
class ChannelConfig:
    """Configuration for a single sensor channel.
    
    Attributes:
        index: Channel index (0-based)
        sensor_model: Sensor model name (e.g., 'TGS2602', 'TGS2611')
        target_gases: Tuple of target gas names this sensor responds to
        unit: Output unit (e.g., 'KOhm', 'mV', 'S_i')
        heater_voltage: Heater voltage in volts (if applicable)
        description: Optional description of the channel
    """
    index: int
    sensor_model: str
    target_gases: Tuple[str, ...] = ()
    unit: str = "raw"
    heater_voltage: Optional[float] = None
    description: str = ""


@dataclass(frozen=True)
class SensorConfig:
    """Configuration for the sensor array.
    
    Attributes:
        type: General sensor type (e.g., 'MOX', 'QCM')
        count: Number of sensor channels
        channels: Tuple of per-channel configurations
        manufacturer: Sensor manufacturer (e.g., 'Figaro')
    """
    type: str = "MOX"
    count: int = 0
    channels: Tuple[ChannelConfig, ...] = ()
    manufacturer: str = ""


@dataclass(frozen=True)
class TimeSeriesConfig:
    continuous: bool = False
    sample_rate_hz: Optional[float] = None  # Supports fractional rates (e.g., 3.5 Hz)


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    uci_id: Optional[int]
    file_name: str
    sha1: str
    url: str
    description: str = ""
    tasks: List[str] = field(default_factory=list)
    sensors: SensorConfig = SensorConfig()
    time_series: Optional[TimeSeriesConfig] = None
    extract: ExtractConfig = ExtractConfig()


# =============================================================================
# Channel configurations for each dataset (gathered from UCI pages)
# =============================================================================

# Twin Gas Sensor Arrays: 8 Figaro MOX sensors at two voltage levels
_TWIN_GAS_CHANNELS = (
    ChannelConfig(0, "TGS2611", ("Methane",), "KOhm", 5.65, "Methane sensor"),
    ChannelConfig(1, "TGS2612", ("Methane", "Propane", "Butane"), "KOhm", 5.65, "Combustible gas"),
    ChannelConfig(2, "TGS2610", ("Propane",), "KOhm", 5.65, "LP gas sensor"),
    ChannelConfig(3, "TGS2602", ("Ammonia", "H2S", "VOC"), "KOhm", 5.65, "Air quality"),
    ChannelConfig(4, "TGS2611", ("Methane",), "KOhm", 5.00, "Methane sensor"),
    ChannelConfig(5, "TGS2612", ("Methane", "Propane", "Butane"), "KOhm", 5.00, "Combustible gas"),
    ChannelConfig(6, "TGS2610", ("Propane",), "KOhm", 5.00, "LP gas sensor"),
    ChannelConfig(7, "TGS2602", ("Ammonia", "H2S", "VOC"), "KOhm", 5.00, "Air quality"),
)

# Gas Sensors for Home Activity Monitoring: 8 Figaro MOX sensors
_HOME_ACTIVITY_CHANNELS = (
    ChannelConfig(0, "TGS2611", ("Methane",), "resistance", None, "R1"),
    ChannelConfig(1, "TGS2612", ("Methane", "Propane", "Butane"), "resistance", None, "R2"),
    ChannelConfig(2, "TGS2610", ("Propane",), "resistance", None, "R3"),
    ChannelConfig(3, "TGS2600", ("Hydrogen", "CO"), "resistance", None, "R4"),
    ChannelConfig(4, "TGS2602", ("Ammonia", "H2S", "VOC"), "resistance", None, "R5"),
    ChannelConfig(5, "TGS2602", ("Ammonia", "H2S", "VOC"), "resistance", None, "R6"),
    ChannelConfig(6, "TGS2620", ("Alcohol", "VOC"), "resistance", None, "R7"),
    ChannelConfig(7, "TGS2620", ("Alcohol", "VOC"), "resistance", None, "R8"),
)

# Gas Sensor Array under Dynamic Gas Mixtures: 16 Figaro MOX sensors (4 types x 4 units)
_DYNAMIC_MIXTURE_CHANNELS = (
    ChannelConfig(0, "TGS2602", ("Ammonia", "H2S", "VOC"), "resistance", 5.0),
    ChannelConfig(1, "TGS2602", ("Ammonia", "H2S", "VOC"), "resistance", 5.0),
    ChannelConfig(2, "TGS2600", ("Hydrogen", "CO"), "resistance", 5.0),
    ChannelConfig(3, "TGS2600", ("Hydrogen", "CO"), "resistance", 5.0),
    ChannelConfig(4, "TGS2610", ("Propane",), "resistance", 5.0),
    ChannelConfig(5, "TGS2610", ("Propane",), "resistance", 5.0),
    ChannelConfig(6, "TGS2620", ("Alcohol", "VOC"), "resistance", 5.0),
    ChannelConfig(7, "TGS2620", ("Alcohol", "VOC"), "resistance", 5.0),
    ChannelConfig(8, "TGS2602", ("Ammonia", "H2S", "VOC"), "resistance", 5.0),
    ChannelConfig(9, "TGS2602", ("Ammonia", "H2S", "VOC"), "resistance", 5.0),
    ChannelConfig(10, "TGS2600", ("Hydrogen", "CO"), "resistance", 5.0),
    ChannelConfig(11, "TGS2600", ("Hydrogen", "CO"), "resistance", 5.0),
    ChannelConfig(12, "TGS2610", ("Propane",), "resistance", 5.0),
    ChannelConfig(13, "TGS2610", ("Propane",), "resistance", 5.0),
    ChannelConfig(14, "TGS2620", ("Alcohol", "VOC"), "resistance", 5.0),
    ChannelConfig(15, "TGS2620", ("Alcohol", "VOC"), "resistance", 5.0),
)

# Gas Sensor Array Exposed to Turbulent Gas Mixtures: 8 Figaro MOX sensors
_TURBULENT_CHANNELS = (
    ChannelConfig(0, "TGS2600", ("Hydrogen", "CO"), "raw", 5.0),
    ChannelConfig(1, "TGS2602", ("Ammonia", "H2S", "VOC"), "raw", 5.0),
    ChannelConfig(2, "TGS2602", ("Ammonia", "H2S", "VOC"), "raw", 5.0),
    ChannelConfig(3, "TGS2620", ("Alcohol", "VOC"), "raw", 5.0),
    ChannelConfig(4, "TGS2612", ("Methane", "Propane", "Butane"), "raw", 5.0),
    ChannelConfig(5, "TGS2620", ("Alcohol", "VOC"), "raw", 5.0),
    ChannelConfig(6, "TGS2611", ("Methane",), "raw", 5.0),
    ChannelConfig(7, "TGS2610", ("Propane",), "raw", 5.0),
)

# Gas Sensor Array Temperature Modulation: 14 sensors (7 Figaro + 7 FIS)
_TEMP_MODULATION_CHANNELS = (
    ChannelConfig(0, "TGS3870-A04", ("CO", "Methane"), "MOhm", None, "Figaro R1"),
    ChannelConfig(1, "TGS3870-A04", ("CO", "Methane"), "MOhm", None, "Figaro R2"),
    ChannelConfig(2, "TGS3870-A04", ("CO", "Methane"), "MOhm", None, "Figaro R3"),
    ChannelConfig(3, "TGS3870-A04", ("CO", "Methane"), "MOhm", None, "Figaro R4"),
    ChannelConfig(4, "TGS3870-A04", ("CO", "Methane"), "MOhm", None, "Figaro R5"),
    ChannelConfig(5, "TGS3870-A04", ("CO", "Methane"), "MOhm", None, "Figaro R6"),
    ChannelConfig(6, "TGS3870-A04", ("CO", "Methane"), "MOhm", None, "Figaro R7"),
    ChannelConfig(7, "SB-500-12", ("CO",), "MOhm", None, "FIS R8"),
    ChannelConfig(8, "SB-500-12", ("CO",), "MOhm", None, "FIS R9"),
    ChannelConfig(9, "SB-500-12", ("CO",), "MOhm", None, "FIS R10"),
    ChannelConfig(10, "SB-500-12", ("CO",), "MOhm", None, "FIS R11"),
    ChannelConfig(11, "SB-500-12", ("CO",), "MOhm", None, "FIS R12"),
    ChannelConfig(12, "SB-500-12", ("CO",), "MOhm", None, "FIS R13"),
    ChannelConfig(13, "SB-500-12", ("CO",), "MOhm", None, "FIS R14"),
)

# Gas Sensor Array Low Concentration: 10 sensors (various manufacturers)
_LOW_CONC_CHANNELS = (
    ChannelConfig(0, "TGS2603", ("Ammonia", "H2S"), "resistance", None, "Figaro"),
    ChannelConfig(1, "TGS2630", ("Freon",), "resistance", None, "Figaro"),
    ChannelConfig(2, "TGS813", ("Combustible",), "resistance", None, "Figaro"),
    ChannelConfig(3, "TGS822", ("Alcohol", "VOC"), "resistance", None, "Figaro"),
    ChannelConfig(4, "MQ-135", ("NH3", "NOx", "Benzene"), "resistance", None, "Winsen"),
    ChannelConfig(5, "MQ-137", ("Ammonia",), "resistance", None, "Winsen"),
    ChannelConfig(6, "MQ-138", ("Benzene", "Alcohol"), "resistance", None, "Winsen"),
    ChannelConfig(7, "2M012", ("Formaldehyde",), "resistance", None),
    ChannelConfig(8, "VOCS-P", ("VOC",), "resistance", None),
    ChannelConfig(9, "2SH12", ("H2S",), "resistance", None),
)

# Gas Sensor Array Drift Dataset: 16 MOX sensors (specific models not documented)
_DRIFT_CHANNELS = tuple(
    ChannelConfig(i, "MOX", (), "features", None, f"Sensor {i+1}")
    for i in range(16)
)

# Gas Sensor Array under Flow Modulation: 16 sensors (specific models not documented)
_FLOW_MODULATION_CHANNELS = tuple(
    ChannelConfig(i, "MOX", (), "dR", None, f"Sensor {i+1}")
    for i in range(16)
)

# Alcohol QCM Sensor: 5 QCM sensors with 2 channels each
_ALCOHOL_QCM_CHANNELS = (
    ChannelConfig(0, "QCM3", ("Alcohol",), "frequency", None, "MIP:NP=1:1"),
    ChannelConfig(1, "QCM6", ("Alcohol",), "frequency", None, "MIP:NP=1:0"),
    ChannelConfig(2, "QCM7", ("Alcohol",), "frequency", None, "MIP:NP=1:0.5"),
    ChannelConfig(3, "QCM10", ("Alcohol",), "frequency", None, "MIP:NP=1:2"),
    ChannelConfig(4, "QCM12", ("Alcohol",), "frequency", None, "MIP:NP=0:1"),
)

# SmellNet: 6 gas sensors per paper (Appendix D)
# "we decided to keep only 6 channels (NO2, C2H5OH, VOC, CO, Alcohol, LPG)"
# Sensors: MQ-3 (Alcohol), MQ-5 (LPG), MQ-9 (CO), WSP2110 (VOC), MP503 (NO2), Grove Multichannel V2
# BME680 for environmental control (pressure, temperature, humidity)
_SMELLNET_CHANNELS = (
    ChannelConfig(0, "MP503", ("NO2",), "raw", None, "Nitrogen dioxide sensor"),
    ChannelConfig(1, "WSP2110", ("Ethanol",), "raw", None, "Ethanol/VOC sensor"),
    ChannelConfig(2, "WSP2110", ("VOC",), "raw", None, "Volatile organic compounds"),
    ChannelConfig(3, "MQ-9", ("CO",), "raw", None, "Carbon monoxide sensor"),
    ChannelConfig(4, "MQ-3", ("Alcohol",), "raw", None, "Alcohol sensor"),
    ChannelConfig(5, "MQ-5", ("LPG",), "raw", None, "Liquefied petroleum gas sensor"),
)


# =============================================================================
# Dataset registry
# =============================================================================

_DATASETS: Dict[str, DatasetInfo] = {
    "alcohol_qcm_sensor_dataset": DatasetInfo(
        name="alcohol_qcm_sensor_dataset",
        uci_id=496,
        file_name="alcohol+qcm+sensor+dataset.zip",
        sha1="c8896d2cfe94208843d5ce1fd4c86ec2b97376a3",
        url="https://archive.ics.uci.edu/static/public/496/alcohol+qcm+sensor+dataset.zip",
        description="Alcohol QCM Sensor Dataset",
        tasks=["classification"],
        sensors=SensorConfig(type="QCM", count=5, channels=_ALCOHOL_QCM_CHANNELS),
        extract=ExtractConfig(type="standard", subdir="QCM Sensor Alcohol Dataset"),
    ),
    "gas_sensor_array_drift_dataset_at_different_concentrations": DatasetInfo(
        name="gas_sensor_array_drift_dataset_at_different_concentrations",
        uci_id=270,
        file_name="gas+sensor+array+drift+dataset+at+different+concentrations.zip",
        sha1="385190f8143902b279d8592fa60d50126ccf9d7c",
        url="https://archive.ics.uci.edu/static/public/270/gas+sensor+array+drift+dataset+at+different+concentrations.zip",
        description="Gas Sensor Array Drift Dataset at Different Concentrations",
        tasks=["classification", "drift_compensation"],
        sensors=SensorConfig(type="MOX", count=16, channels=_DRIFT_CHANNELS),
        extract=ExtractConfig(type="standard"),
    ),
    "gas_sensor_array_exposed_to_turbulent_gas_mixtures": DatasetInfo(
        name="gas_sensor_array_exposed_to_turbulent_gas_mixtures",
        uci_id=309,
        file_name="gas+sensor+array+exposed+to+turbulent+gas+mixtures.zip",
        sha1="c812370aa04a1491781767b3cf606ce6fa0d221e",
        url="https://archive.ics.uci.edu/static/public/309/gas+sensor+array+exposed+to+turbulent+gas+mixtures.zip",
        description="Gas Sensor Array Exposed to Turbulent Gas Mixtures",
        tasks=["classification", "concentration_prediction"],
        sensors=SensorConfig(type="MOX", count=8, channels=_TURBULENT_CHANNELS, manufacturer="Figaro"),
        time_series=TimeSeriesConfig(continuous=True, sample_rate_hz=50),
        extract=ExtractConfig(type="turbo"),
    ),
    "gas_sensor_array_low_concentration": DatasetInfo(
        name="gas_sensor_array_low_concentration",
        uci_id=1081,
        file_name="gas+sensor+array+low-concentration.zip",
        sha1="41818a074550dd23a222856afdc19c7fbf32903e",
        url="https://archive.ics.uci.edu/static/public/1081/gas+sensor+array+low-concentration.zip",
        description="Gas Sensor Array Low-Concentration Dataset",
        tasks=["classification", "concentration_prediction"],
        sensors=SensorConfig(type="MOX", count=10, channels=_LOW_CONC_CHANNELS, manufacturer="Mixed"),
        time_series=TimeSeriesConfig(continuous=True, sample_rate_hz=1),
        extract=ExtractConfig(type="standard"),
    ),
    "gas_sensor_array_temperature_modulation": DatasetInfo(
        name="gas_sensor_array_temperature_modulation",
        uci_id=487,
        file_name="gas+sensor+array+temperature+modulation.zip",
        sha1="e9caaded42fa57086da2a56f5a3dfb2f7a7708d8",
        url="https://archive.ics.uci.edu/static/public/487/gas+sensor+array+temperature+modulation.zip",
        description="Gas Sensor Array Temperature Modulation",
        tasks=["classification"],
        sensors=SensorConfig(type="MOX", count=14, channels=_TEMP_MODULATION_CHANNELS, manufacturer="Figaro/FIS"),
        time_series=TimeSeriesConfig(continuous=True, sample_rate_hz=3.5),  # UCI: sampled at 3.5 Hz
        extract=ExtractConfig(type="turbo"),
    ),
    "gas_sensor_array_under_dynamic_gas_mixtures": DatasetInfo(
        name="gas_sensor_array_under_dynamic_gas_mixtures",
        uci_id=322,
        file_name="gas+sensor+array+under+dynamic+gas+mixtures.zip",
        sha1="12dd9c43e621d56b53f2f8894dddd6e410283d47",
        url="https://archive.ics.uci.edu/static/public/322/gas+sensor+array+under+dynamic+gas+mixtures.zip",
        description="Gas Sensor Array under Dynamic Gas Mixtures",
        tasks=["classification", "concentration_prediction"],
        sensors=SensorConfig(type="MOX", count=16, channels=_DYNAMIC_MIXTURE_CHANNELS, manufacturer="Figaro"),
        time_series=TimeSeriesConfig(continuous=True, sample_rate_hz=100),
        extract=ExtractConfig(type="standard"),
    ),
    "gas_sensor_array_under_flow_modulation": DatasetInfo(
        name="gas_sensor_array_under_flow_modulation",
        uci_id=308,
        file_name="gas+sensor+array+under+flow+modulation.zip",
        sha1="d8c3de28ece518cf57ed842c1d5b8a23ee03398b",
        url="https://archive.ics.uci.edu/static/public/308/gas+sensor+array+under+flow+modulation.zip",
        description="Gas Sensor Array under Flow Modulation",
        tasks=["classification"],
        sensors=SensorConfig(type="MOX", count=16, channels=_FLOW_MODULATION_CHANNELS),
        time_series=TimeSeriesConfig(continuous=True, sample_rate_hz=25),
        extract=ExtractConfig(type="turbo"),
    ),
    "gas_sensors_for_home_activity_monitoring": DatasetInfo(
        name="gas_sensors_for_home_activity_monitoring",
        uci_id=362,
        file_name="gas+sensors+for+home+activity+monitoring.zip",
        sha1="34101ca24e556dc14a6ee1e2910111ed49c0e6ce",
        url="https://archive.ics.uci.edu/static/public/362/gas+sensors+for+home+activity+monitoring.zip",
        description="Gas Sensors for Home Activity Monitoring",
        tasks=["classification", "activity_recognition"],
        sensors=SensorConfig(type="MOX", count=8, channels=_HOME_ACTIVITY_CHANNELS, manufacturer="Figaro"),
        time_series=TimeSeriesConfig(continuous=True, sample_rate_hz=1),
        extract=ExtractConfig(type="nested", nested_zip="HT_Sensor_dataset.zip"),
    ),
    "twin_gas_sensor_arrays": DatasetInfo(
        name="twin_gas_sensor_arrays",
        uci_id=361,
        file_name="twin+gas+sensor+arrays.zip",
        sha1="a235ab3cf55685fc8346eafea3f18e080adb77a3",
        url="https://archive.ics.uci.edu/static/public/361/twin+gas+sensor+arrays.zip",
        description="Twin Gas Sensor Arrays",
        tasks=["classification", "drift_compensation", "transfer_learning"],
        sensors=SensorConfig(type="MOX", count=8, channels=_TWIN_GAS_CHANNELS, manufacturer="Figaro"),
        time_series=TimeSeriesConfig(continuous=True, sample_rate_hz=100),
        extract=ExtractConfig(type="standard", subdir="data1"),
    ),
    "smellnet_pure": DatasetInfo(
        name="smellnet_pure",
        uci_id=None,  # Not from UCI
        file_name="",  # HuggingFace dataset: DeweiFeng/smell-net
        sha1="",
        url="https://huggingface.co/datasets/DeweiFeng/smell-net",
        description="SmellNet Pure Substances: 50 classes of pure ingredients @ 1Hz, 12-channel sensor array",
        tasks=["classification"],
        sensors=SensorConfig(type="MOX", count=6, channels=_SMELLNET_CHANNELS, manufacturer="Seeed/Generic"),
        time_series=TimeSeriesConfig(continuous=False, sample_rate_hz=1),  # Paper: 1Hz for Pure
        extract=ExtractConfig(type="huggingface"),
    ),
    "smellnet_mixture": DatasetInfo(
        name="smellnet_mixture",
        uci_id=None,  # Not from UCI
        file_name="",  # HuggingFace dataset: DeweiFeng/smell-net
        sha1="",
        url="https://huggingface.co/datasets/DeweiFeng/smell-net",
        description="SmellNet Mixtures: 43 classes of simulated mixtures @ 10Hz, 12-channel sensor array",
        tasks=["classification"],
        sensors=SensorConfig(type="MOX", count=6, channels=_SMELLNET_CHANNELS, manufacturer="Seeed/Generic"),
        time_series=TimeSeriesConfig(continuous=False, sample_rate_hz=10),  # Paper: 10Hz for Mixture
        extract=ExtractConfig(type="huggingface"),
    ),
}


def get_dataset_info(name: str) -> DatasetInfo:
    normalized = name.lower().replace("-", "_")
    # Handle smellnet alias (base class uses 'smellnet', actual subsets are smellnet_pure/mixture)
    if normalized == "smellnet":
        normalized = "smellnet_pure"  # Default to pure subset
    if normalized not in _DATASETS:
        available = ", ".join(sorted(_DATASETS.keys()))
        raise KeyError(f"Unknown dataset: {name}. Available: {available}")
    return _DATASETS[normalized]


def list_datasets() -> List[str]:
    return sorted(_DATASETS.keys())
