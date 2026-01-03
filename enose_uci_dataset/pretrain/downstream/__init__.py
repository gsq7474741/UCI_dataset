"""Downstream task evaluation framework.

This module provides:
- Base classes for downstream tasks and models
- Concentration regression task implementation
- Evaluator for comparing original vs reconstructed inputs
"""

from .base import BaseDownstreamTask, BaseDownstreamModel
from .tasks import ConcentrationRegressionTask
from .models import MLPRegressor, TCNRegressor
from .evaluator import DownstreamEvaluator

__all__ = [
    "BaseDownstreamTask",
    "BaseDownstreamModel",
    "ConcentrationRegressionTask",
    "MLPRegressor",
    "TCNRegressor",
    "DownstreamEvaluator",
]
