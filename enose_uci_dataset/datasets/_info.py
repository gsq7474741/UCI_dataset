from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ExtractConfig:
    type: str = "standard"
    subdir: Optional[str] = None
    nested_zip: Optional[str] = None


@dataclass(frozen=True)
class SensorConfig:
    type: str = "MOX"
    count: int = 0


@dataclass(frozen=True)
class TimeSeriesConfig:
    continuous: bool = False
    sample_rate_hz: Optional[int] = None


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


_DATASETS: Dict[str, DatasetInfo] = {
    "alcohol_qcm_sensor_dataset": DatasetInfo(
        name="alcohol_qcm_sensor_dataset",
        uci_id=496,
        file_name="alcohol+qcm+sensor+dataset.zip",
        sha1="c8896d2cfe94208843d5ce1fd4c86ec2b97376a3",
        url="https://archive.ics.uci.edu/static/public/496/alcohol+qcm+sensor+dataset.zip",
        description="Alcohol QCM Sensor Dataset",
        tasks=["classification"],
        sensors=SensorConfig(type="QCM", count=5),
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
        sensors=SensorConfig(type="MOX", count=16),
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
        sensors=SensorConfig(type="MOX", count=8),
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
        sensors=SensorConfig(type="MOX", count=8),
        time_series=TimeSeriesConfig(continuous=True, sample_rate_hz=100),
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
        sensors=SensorConfig(type="MOX", count=14),
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
        sensors=SensorConfig(type="MOX", count=16),
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
        sensors=SensorConfig(type="MOX", count=16),
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
        sensors=SensorConfig(type="MOX", count=8),
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
        sensors=SensorConfig(type="MOX", count=8),
        time_series=TimeSeriesConfig(continuous=True, sample_rate_hz=100),
        extract=ExtractConfig(type="standard", subdir="data1"),
    ),
}


def get_dataset_info(name: str) -> DatasetInfo:
    normalized = name.lower().replace("-", "_")
    if normalized not in _DATASETS:
        available = ", ".join(sorted(_DATASETS.keys()))
        raise KeyError(f"Unknown dataset: {name}. Available: {available}")
    return _DATASETS[normalized]


def list_datasets() -> List[str]:
    return sorted(_DATASETS.keys())
