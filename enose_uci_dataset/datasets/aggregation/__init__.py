"""Aggregation datasets - unified and combined interfaces."""
from .unified import (
    UnifiedEnoseDataset,
    ChannelAlignMode,
    TaskType,
    SplitType,
    NormalizeType,
    SplitConfig,
    WindowConfig,
    UnifiedDatasetConfig,
    TASK_DATASET_COMPATIBILITY,
)
from .combined import (
    CombinedEnoseDataset,
    PretrainingDataset,
)

__all__ = [
    "UnifiedEnoseDataset",
    "ChannelAlignMode",
    "TaskType",
    "SplitType",
    "NormalizeType",
    "SplitConfig",
    "WindowConfig",
    "UnifiedDatasetConfig",
    "TASK_DATASET_COMPATIBILITY",
    "CombinedEnoseDataset",
    "PretrainingDataset",
]
