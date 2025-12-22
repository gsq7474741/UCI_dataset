"""
数据集注册中心 - 统一管理所有UCI电子鼻数据集的元信息
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
from pathlib import Path


@dataclass
class DatasetInfo:
    """数据集信息"""
    name: str
    file_name: str
    sha1_hash: str
    url: str
    uci_id: int
    description: str = ""
    extract_subdir: Optional[str] = None  # 解压后需要移动的子目录


# 数据集注册表
DATASETS: Dict[str, DatasetInfo] = {
    "alcohol_qcm_sensor_dataset": DatasetInfo(
        name="alcohol_qcm_sensor_dataset",
        file_name="alcohol+qcm+sensor+dataset.zip",
        sha1_hash="c8896d2cfe94208843d5ce1fd4c86ec2b97376a3",
        url="https://archive.ics.uci.edu/static/public/496/alcohol+qcm+sensor+dataset.zip",
        uci_id=496,
        description="Alcohol QCM Sensor Dataset",
        extract_subdir="QCM Sensor Alcohol Dataset",
    ),
    "gas_sensor_array_drift_dataset_at_different_concentrations": DatasetInfo(
        name="gas_sensor_array_drift_dataset_at_different_concentrations",
        file_name="gas+sensor+array+drift+dataset+at+different+concentrations.zip",
        sha1_hash="385190f8143902b279d8592fa60d50126ccf9d7c",
        url="https://archive.ics.uci.edu/static/public/270/gas+sensor+array+drift+dataset+at+different+concentrations.zip",
        uci_id=270,
        description="Gas Sensor Array Drift Dataset at Different Concentrations",
    ),
    "gas_sensor_array_exposed_to_turbulent_gas_mixtures": DatasetInfo(
        name="gas_sensor_array_exposed_to_turbulent_gas_mixtures",
        file_name="gas+sensor+array+exposed+to+turbulent+gas+mixtures.zip",
        sha1_hash="c812370aa04a1491781767b3cf606ce6fa0d221e",
        url="https://archive.ics.uci.edu/static/public/309/gas+sensor+array+exposed+to+turbulent+gas+mixtures.zip",
        uci_id=309,
        description="Gas Sensor Array Exposed to Turbulent Gas Mixtures",
    ),
    "gas_sensor_array_low_concentration": DatasetInfo(
        name="gas_sensor_array_low_concentration",
        file_name="gas+sensor+array+low-concentration.zip",
        sha1_hash="41818a074550dd23a222856afdc19c7fbf32903e",
        url="https://archive.ics.uci.edu/static/public/1081/gas+sensor+array+low-concentration.zip",
        uci_id=1081,
        description="Gas Sensor Array Low-Concentration Dataset",
    ),
    "gas_sensor_array_temperature_modulation": DatasetInfo(
        name="gas_sensor_array_temperature_modulation",
        file_name="gas+sensor+array+temperature+modulation.zip",
        sha1_hash="e9caaded42fa57086da2a56f5a3dfb2f7a7708d8",
        url="https://archive.ics.uci.edu/static/public/487/gas+sensor+array+temperature+modulation.zip",
        uci_id=487,
        description="Gas Sensor Array Temperature Modulation",
    ),
    # "gas_sensor_array_under_dynamic_gas_mixtures": DatasetInfo(
    #     name="gas_sensor_array_under_dynamic_gas_mixtures",
    #     file_name="gas+sensor+array+under+dynamic+gas+mixtures.zip",
    #     sha1_hash="12dd9c43e621d56b53f2f8894dddd6e410283d47",
    #     url="https://archive.ics.uci.edu/static/public/322/gas+sensor+array+under+dynamic+gas+mixtures.zip",
    #     uci_id=322,
    #     description="Gas Sensor Array under Dynamic Gas Mixtures",
    # ),
    "gas_sensor_array_under_flow_modulation": DatasetInfo(
        name="gas_sensor_array_under_flow_modulation",
        file_name="gas+sensor+array+under+flow+modulation.zip",
        sha1_hash="d8c3de28ece518cf57ed842c1d5b8a23ee03398b",
        url="https://archive.ics.uci.edu/static/public/308/gas+sensor+array+under+flow+modulation.zip",
        uci_id=308,
        description="Gas Sensor Array under Flow Modulation",
    ),
    "gas_sensors_for_home_activity_monitoring": DatasetInfo(
        name="gas_sensors_for_home_activity_monitoring",
        file_name="gas+sensors+for+home+activity+monitoring.zip",
        sha1_hash="34101ca24e556dc14a6ee1e2910111ed49c0e6ce",
        url="https://archive.ics.uci.edu/static/public/362/gas+sensors+for+home+activity+monitoring.zip",
        uci_id=362,
        description="Gas Sensors for Home Activity Monitoring",
        extract_subdir="HT_Sensor_dataset.zip",  # 嵌套zip
    ),
    "twin_gas_sensor_arrays": DatasetInfo(
        name="twin_gas_sensor_arrays",
        file_name="twin+gas+sensor+arrays.zip",
        sha1_hash="a235ab3cf55685fc8346eafea3f18e080adb77a3",
        url="https://archive.ics.uci.edu/static/public/361/twin+gas+sensor+arrays.zip",
        uci_id=361,
        description="Twin Gas Sensor Arrays",
        extract_subdir="data1",
    ),
}


def get_dataset_info(name: str) -> DatasetInfo:
    """获取数据集信息"""
    # 支持带横杠和下划线的名称
    normalized = name.replace("-", "_")
    if normalized not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")
    return DATASETS[normalized]


def list_datasets() -> List[str]:
    """列出所有可用数据集"""
    return list(DATASETS.keys())


def get_dataset_dir(name: str) -> Path:
    """获取数据集目录路径"""
    info = get_dataset_info(name)
    base_dir = Path(__file__).parent
    # 目录名可能用横杠
    dir_name = info.name.replace("_", "-") if (base_dir / info.name.replace("_", "-")).exists() else info.name
    return base_dir / dir_name


if __name__ == "__main__":
    print("Available datasets:")
    for name in list_datasets():
        info = DATASETS[name]
        print(f"  - {name} (UCI ID: {info.uci_id})")
