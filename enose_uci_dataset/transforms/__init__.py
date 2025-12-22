"""
数据转换模块

包含基础转换和时序数据增强
"""

# 从 legacy.transforms_base.py 导入基础转换
from ..legacy.transforms_base import (
    Transform,
    Compose,
    SelectColumns,
    RenameColumns,
    NormalizeSensorColumns,
    ConvertTimeUnit,
    AddRelativeTime,
    ToResistance,
    Normalize,
    FillNaN,
    Resample,
    SlidingWindow,
    SegmentByLabel,
    DropNA,
    FilterByValue,
    get_default_transforms,
)

# 时序数据增强
from .augmentation import (
    AddGaussianNoise,
    TimeWarp,
    Scaling,
    Permutation,
    Rotation,
    MagnitudeWarp,
    Jitter,
    Dropout,
    RandomCrop,
    CenterCrop,
    get_weak_augmentation,
    get_strong_augmentation,
)

__all__ = [
    # 基础
    "Transform",
    "Compose",
    "SelectColumns",
    "RenameColumns",
    "NormalizeSensorColumns",
    "ConvertTimeUnit",
    "AddRelativeTime",
    "ToResistance",
    "Normalize",
    "FillNaN",
    "Resample",
    "SlidingWindow",
    "SegmentByLabel",
    "DropNA",
    "FilterByValue",
    "get_default_transforms",
    # 增强
    "AddGaussianNoise",
    "TimeWarp",
    "Scaling",
    "Permutation",
    "Rotation",
    "MagnitudeWarp",
    "Jitter",
    "Dropout",
    "RandomCrop",
    "CenterCrop",
    "get_weak_augmentation",
    "get_strong_augmentation",
]
