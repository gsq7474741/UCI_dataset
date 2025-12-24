"""E-nose pretraining module.

Self-supervised pretraining for gas sensor time series using:
- VQ-VAE for time series tokenization
- Channel masking for cross-channel modeling
- Sensor metadata conditioning
"""

from .model import EnoseVQVAE
from .datamodule import EnosePretrainingDataModule
from .trainer import create_trainer

__all__ = [
    "EnoseVQVAE",
    "EnosePretrainingDataModule", 
    "create_trainer",
]
