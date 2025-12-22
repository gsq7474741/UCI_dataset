"""
数据集工具函数
"""

from pathlib import Path
from typing import Dict, List, Optional, Type, Union

from .base import ENoseDataset
from .home_activity import HomeActivityDataset
from .twin_sensor import TwinSensorDataset
from .dynamic_gas import DynamicGasDataset
from .low_concentration import LowConcentrationDataset


# 数据集注册表
DATASET_REGISTRY: Dict[str, Type[ENoseDataset]] = {
    "gas_sensors_for_home_activity_monitoring": HomeActivityDataset,
    "home_activity": HomeActivityDataset,  # 别名
    "twin_gas_sensor_arrays": TwinSensorDataset,
    "twin_sensor": TwinSensorDataset,  # 别名
    "gas_sensor_array_under_dynamic_gas_mixtures": DynamicGasDataset,
    "dynamic_gas": DynamicGasDataset,  # 别名
    "gas_sensor_array_low_concentration": LowConcentrationDataset,
    "low_concentration": LowConcentrationDataset,  # 别名
}


def list_available_datasets() -> List[str]:
    """列出所有可用的具体数据集类"""
    # 去重，只返回完整名称
    unique = set()
    for name, cls in DATASET_REGISTRY.items():
        unique.add(cls.name)
    return sorted(unique)


def load_dataset(
    name: str,
    root: Optional[Union[str, Path]] = None,
    split: Optional[str] = None,
    transform: Optional[callable] = None,
    target_transform: Optional[callable] = None,
    download: bool = False,
) -> ENoseDataset:
    """
    加载数据集
    
    Args:
        name: 数据集名称 (支持别名)
        root: 数据根目录
        split: 数据划分 ("train", "val", "test", None)
        transform: 数据转换
        target_transform: 标签转换
        download: 是否自动下载
    
    Returns:
        对应的数据集实例
    
    Example:
        >>> ds = load_dataset("home_activity", download=True)
        >>> ds = load_dataset("gas_sensors_for_home_activity_monitoring", split="train")
    """
    # 标准化名称
    normalized = name.lower().replace("-", "_")
    
    if normalized not in DATASET_REGISTRY:
        available = list_available_datasets()
        raise ValueError(
            f"Unknown dataset: {name}\n"
            f"Available datasets: {available}"
        )
    
    dataset_cls = DATASET_REGISTRY[normalized]
    
    return dataset_cls(
        root=root,
        split=split,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def get_dataset_info(name: str) -> dict:
    """获取数据集信息"""
    normalized = name.lower().replace("-", "_")
    
    if normalized not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")
    
    dataset_cls = DATASET_REGISTRY[normalized]
    
    return {
        "name": dataset_cls.name,
        "class": dataset_cls.__name__,
        "doc": dataset_cls.__doc__,
    }
