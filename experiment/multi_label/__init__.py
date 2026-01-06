"""Multi-label classification experiment for gas mixture decomposition.

This experiment trains on pure gas samples and tests on mixture samples
to evaluate if the model can identify component gases in mixtures.

Datasets:
- Training (pure gases): TwinGasSensorArrays (Ethylene, Methane, CO)
- Testing (mixtures): GasSensorTurbulent, GasSensorDynamic

Gas components in mixtures:
- GasSensorTurbulent: Ethylene + Methane or Ethylene + CO
- GasSensorDynamic: Ethylene + CO or Ethylene + Methane

Task: Multi-label classification - predict which gas components are present.
"""

from .dataset import MultiLabelGasDataset, GasLabelEncoder
from .model import MultiLabelClassifier
from .datamodule import MultiLabelDataModule

__all__ = [
    "MultiLabelGasDataset",
    "GasLabelEncoder", 
    "MultiLabelClassifier",
    "MultiLabelDataModule",
]
