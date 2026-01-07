"""Single-dataset classification experiment.

Train a 1D CNN classifier on individual e-nose datasets with global label space.
"""

from .dataset import (
    SingleDatasetClassification,
    GlobalLabelEncoder,
    CLASSIFICATION_DATASETS,
    list_classification_datasets,
    collate_fn,
)
from .model import CNN1DClassifier, CNN1DEncoder
from .datamodule import ClassificationDataModule
from .grad_cam import GradCAM1D, compute_purity, compute_importance
from .callbacks import VisualizationCallback, TestOnBestCallback

__all__ = [
    "SingleDatasetClassification",
    "GlobalLabelEncoder",
    "CLASSIFICATION_DATASETS",
    "list_classification_datasets",
    "collate_fn",
    "CNN1DClassifier",
    "CNN1DEncoder",
    "ClassificationDataModule",
    "GradCAM1D",
    "compute_purity",
    "compute_importance",
    "VisualizationCallback",
    "TestOnBestCallback",
]
