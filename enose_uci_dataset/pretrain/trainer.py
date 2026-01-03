"""Training utilities for e-nose pretraining."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger


class ExportBestModelCallback(Callback):
    """Callback to export best model to ONNX/TorchScript when a new best is saved."""
    
    def __init__(
        self,
        export_dir: Union[str, Path],
        export_format: str = "onnx",
        max_channels: int = 16,
        seq_len: int = 512,
    ):
        super().__init__()
        self.export_dir = Path(export_dir)
        self.export_format = export_format
        self.max_channels = max_channels
        self.seq_len = seq_len
        self._last_best_path: Optional[str] = None
    
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Check if a new best model was saved and export it."""
        ckpt_callback = trainer.checkpoint_callback
        if ckpt_callback is None:
            return
        
        current_best = ckpt_callback.best_model_path
        if current_best and current_best != self._last_best_path:
            self._last_best_path = current_best
            self._export_model(pl_module, trainer.current_epoch)
    
    def _export_model(self, pl_module: L.LightningModule, epoch: int) -> None:
        """Export the model to ONNX or TorchScript."""
        import copy
        
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Create wrapper to avoid Lightning-specific issues
        class ModelWrapper(nn.Module):
            def __init__(self, encoder, vq, decoder):
                super().__init__()
                self.encoder = encoder
                self.vq = vq
                self.decoder = decoder
            
            def forward(self, x, channel_mask, sensor_indices):
                z = self.encoder(x, channel_mask, sensor_indices)
                z_q, vq_loss, perplexity = self.vq(z)
                x_recon = self.decoder(z_q, sensor_indices)
                return x_recon, z_q, vq_loss, perplexity
        
        # Use deepcopy to avoid modifying the original model during training
        wrapper = ModelWrapper(
            copy.deepcopy(pl_module.encoder).cpu(),
            copy.deepcopy(pl_module.vq).cpu(),
            copy.deepcopy(pl_module.decoder).cpu(),
        )
        wrapper.eval()
        
        # Create dummy inputs
        dummy_x = torch.randn(1, self.max_channels, self.seq_len)
        dummy_mask = torch.zeros(1, self.max_channels, dtype=torch.bool)
        dummy_indices = torch.zeros(1, self.max_channels, dtype=torch.long)
        
        if self.export_format == "onnx":
            output_path = self.export_dir / f"best_epoch{epoch:03d}.onnx"
            try:
                torch.onnx.export(
                    wrapper,
                    (dummy_x, dummy_mask, dummy_indices),
                    str(output_path),
                    input_names=["x", "channel_mask", "sensor_indices"],
                    output_names=["reconstruction", "z_q", "vq_loss", "perplexity"],
                    opset_version=14,
                    dynamo=False,
                )
                print(f"\n✓ Exported best model to: {output_path}")
            except Exception as e:
                print(f"\n✗ Failed to export ONNX: {e}")
        else:  # torchscript
            output_path = self.export_dir / f"best_epoch{epoch:03d}.pt"
            try:
                with torch.no_grad():
                    traced = torch.jit.trace(wrapper, (dummy_x, dummy_mask, dummy_indices))
                traced.save(str(output_path))
                print(f"\n✓ Exported best model to: {output_path}")
            except Exception as e:
                print(f"\n✗ Failed to export TorchScript: {e}")

try:
    from lightning.pytorch.loggers import MLFlowLogger
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


def create_trainer(
    max_epochs: int = 100,
    accelerator: str = "auto",
    devices: Union[int, str] = "auto",
    precision: str = "16-mixed",
    log_dir: Union[str, Path] = "logs",
    experiment_name: str = "enose_pretrain",
    run_name: Optional[str] = None,
    use_wandb: bool = False,
    wandb_project: str = "enose-pretrain",
    use_mlflow: bool = False,
    mlflow_tracking_uri: Optional[str] = None,
    early_stopping_patience: int = 10,
    checkpoint_monitor: str = "val/loss",
    gradient_clip_val: float = 1.0,
    accumulate_grad_batches: int = 1,
    val_check_interval: Optional[float] = None,
    export_best: bool = True,
    export_format: str = "torchscript",  # torchscript is more compatible than onnx
    max_channels: int = 20,
    additional_callbacks: Optional[List[Callback]] = None,
    **kwargs,
) -> L.Trainer:
    """Create a Lightning Trainer with sensible defaults for pretraining.
    
    Args:
        max_epochs: Maximum training epochs
        accelerator: Accelerator type ('cpu', 'gpu', 'auto')
        devices: Number of devices or 'auto'
        precision: Training precision ('32', '16-mixed', 'bf16-mixed')
        log_dir: Directory for logs and checkpoints
        experiment_name: Name for this experiment
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: W&B project name
        early_stopping_patience: Epochs to wait before early stopping
        checkpoint_monitor: Metric to monitor for checkpointing
        gradient_clip_val: Gradient clipping value
        accumulate_grad_batches: Gradient accumulation steps
        val_check_interval: Validation check interval (None for every epoch)
        **kwargs: Additional trainer arguments
        
    Returns:
        Configured Lightning Trainer
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for this run (YYMMDDHHMMSS format)
    run_timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    
    # Base run name (without timestamp)
    base_run_name = run_name if run_name else experiment_name
    
    # Full run name with timestamp (for MLflow)
    full_run_name = f"{base_run_name}_{run_timestamp}"
    
    # Directory structure: log_dir/experiment_name/base_run_name/timestamp/
    run_dir = log_dir / experiment_name / base_run_name / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="{epoch:03d}-loss{val-loss:.4f}",
            monitor=checkpoint_monitor,
            mode="min",
            save_top_k=1,  # Only save the best checkpoint
            save_last=True,
        ),
        EarlyStopping(
            monitor=checkpoint_monitor,
            patience=early_stopping_patience,
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]
    
    # Auto-export best model for Netron visualization
    if export_best:
        callbacks.append(
            ExportBestModelCallback(
                export_dir=run_dir / "exports",
                export_format=export_format,
                max_channels=max_channels,
            )
        )
    
    # Add any additional callbacks (e.g., DownstreamProbeCallback)
    if additional_callbacks:
        callbacks.extend(additional_callbacks)
    
    # Loggers - use multiple loggers for comprehensive tracking
    loggers = []
    
    # TensorBoard logger (always enabled)
    # Logs directly to run_dir (no extra subdirectory)
    tb_logger = TensorBoardLogger(
        save_dir=str(run_dir),
        name="",
        version="",
    )
    loggers.append(tb_logger)
    
    # CSV logger (always enabled)
    csv_logger = CSVLogger(
        save_dir=str(run_dir),
        name="csv",
        version="",
    )
    loggers.append(csv_logger)
    
    # MLflow logger (disabled, using TensorBoard instead)
    # if use_mlflow:
    #     if not HAS_MLFLOW:
    #         print("Warning: MLflow not installed. Run `pip install mlflow` to enable MLflow logging.")
    #     else:
    #         tracking_uri = mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI", f"file://{log_dir.absolute()}/mlruns")
    #         mlflow_logger = MLFlowLogger(
    #             experiment_name=experiment_name,
    #             run_name=full_run_name,
    #             tracking_uri=tracking_uri,
    #             log_model=True,
    #         )
    #         loggers.append(mlflow_logger)
    #         print(f"MLflow tracking URI: {tracking_uri}")
    
    # # W&B logger (optional)
    # if use_wandb:
    #     wandb_logger = WandbLogger(
    #         project=wandb_project,
    #         name=run_name,
    #         save_dir=str(run_dir),
    #     )
    #     loggers.append(wandb_logger)
    
    print(f"Run name: {full_run_name}")
    print(f"Logging to: {run_dir}")
    print(f"Active loggers: {[type(l).__name__ for l in loggers]}")
    
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=val_check_interval,
        log_every_n_steps=10,
        enable_progress_bar=True,
        **kwargs,
    )
    
    return trainer, run_dir


def train(
    datasets: Optional[List[str]] = None,
    root: str = ".cache",
    batch_size: int = 32,
    max_epochs: int = 100,
    learning_rate: float = 1e-4,
    d_model: int = 128,
    num_encoder_layers: int = 4,
    num_decoder_layers: int = 4,
    patch_size: int = 16,
    max_length: int = 4096,
    max_channels: int = 20,
    mask_ratio: float = 0.25,
    num_embeddings: int = 512,
    num_workers: int = 4,
    accelerator: str = "auto",
    devices: Union[int, str] = "auto",
    precision: str = "16-mixed",
    log_dir: str = "logs",
    experiment_name: str = "enose_pretrain",
    run_name: Optional[str] = None,
    use_wandb: bool = False,
    use_mlflow: bool = False,
    mlflow_tracking_uri: Optional[str] = None,
    resume_from: Optional[str] = None,
    download: bool = False,
) -> str:
    """Run pretraining with the specified configuration.
    
    Args:
        datasets: List of dataset names to use
        root: Root directory for datasets
        batch_size: Batch size
        max_epochs: Maximum epochs
        learning_rate: Learning rate
        d_model: Model dimension
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        patch_size: Patch size for time series
        max_length: Maximum sequence length
        max_channels: Maximum number of channels
        mask_ratio: Channel masking ratio
        num_embeddings: VQ codebook size
        num_workers: DataLoader workers
        accelerator: Accelerator type
        devices: Number of devices
        precision: Training precision
        log_dir: Log directory
        experiment_name: Experiment name
        use_wandb: Use Weights & Biases
        resume_from: Checkpoint path to resume from
        download: Download datasets if missing
    """
    from .datamodule import EnosePretrainingDataModule
    from .model import EnoseVQVAE
    
    # Create data module
    datamodule = EnosePretrainingDataModule(
        root=root,
        datasets=datasets,
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=max_length,
        max_channels=max_channels,
        download=download,
    )
    
    # Create model
    if resume_from:
        model = EnoseVQVAE.load_from_checkpoint(resume_from)
        print(f"Resumed from checkpoint: {resume_from}")
    else:
        model = EnoseVQVAE(
            d_model=d_model,
            nhead=8,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            patch_size=patch_size,
            num_embeddings=num_embeddings,
            max_channels=max_channels,
            learning_rate=learning_rate,
            mask_ratio=mask_ratio,
        )
    
    # Create trainer
    trainer, run_dir = create_trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        log_dir=log_dir,
        experiment_name=experiment_name,
        run_name=run_name,
        use_wandb=use_wandb,
        use_mlflow=use_mlflow,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
    
    # Train
    trainer.fit(model, datamodule=datamodule)
    
    print(f"Training complete. Best model: {trainer.checkpoint_callback.best_model_path}")
    print(f"Run directory: {run_dir}")
    return trainer.checkpoint_callback.best_model_path
