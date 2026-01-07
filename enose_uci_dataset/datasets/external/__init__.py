"""External datasets (non-UCI sources)."""
from .smellnet import (
    SmellNet,
    SMELLNET_INGREDIENTS,
    SMELLNET_6_CHANNELS,
    SMELLNET_ALL_CHANNELS,
)
from .g919_55 import (
    G919SensorDataset,
    G919_CATEGORIES,
    G919_ODOR_TO_IDX,
    G919_IDX_TO_ODOR,
    G919_NUM_CLASSES,
    G919_CATEGORY_TO_IDX,
)

__all__ = [
    "SmellNet",
    "SMELLNET_INGREDIENTS",
    "SMELLNET_6_CHANNELS",
    "SMELLNET_ALL_CHANNELS",
    "G919SensorDataset",
    "G919_CATEGORIES",
    "G919_ODOR_TO_IDX",
    "G919_IDX_TO_ODOR",
    "G919_NUM_CLASSES",
    "G919_CATEGORY_TO_IDX",
]
