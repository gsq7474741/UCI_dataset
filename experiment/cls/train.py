#!/usr/bin/env python
"""Training script for single-dataset gas classification.

Usage:
    # Train on twin_gas_sensor_arrays
    python experiment/cls/train.py --dataset twin_gas_sensor_arrays
    
    # Train on drift dataset
    python experiment/cls/train.py --dataset gas_sensor_array_drift_dataset_at_different_concentrations
    
    # Train with custom hyperparameters
    python experiment/cls/train.py --dataset twin_gas_sensor_arrays --hidden-dim 128 --num-layers 6
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

# Optional: MLflow
try:
    from lightning.pytorch.loggers import MLFlowLogger
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from experiment.cls.dataset import CLASSIFICATION_DATASETS, list_classification_datasets
from experiment.cls.model import CNN1DClassifier, DualViewClassifier
from experiment.cls.datamodule import ClassificationDataModule
from experiment.cls.callbacks import VisualizationCallback, TestOnBestCallback


def parse_args():
    parser = argparse.ArgumentParser(description="E-nose Gas Classification")
    
    # Data arguments
    parser.add_argument("--dataset", type=str, default="twin_gas_sensor_arrays",
                        choices=list_classification_datasets(),
                        help="Dataset to train on")
    parser.add_argument("--root", type=str, default=".cache",
                        help="Dataset root directory")
    parser.add_argument("--download", action="store_true",
                        help="Download missing dataset")
    
    # Model arguments
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Number of CNN layers")
    parser.add_argument("--kernel-size", type=int, default=7,
                        help="Convolution kernel size")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing factor")
    parser.add_argument("--mixup-alpha", type=float, default=0.0,
                        help="Mixup alpha (0=disabled, typical: 0.2-0.4)")
    parser.add_argument("--use-class-weights", action="store_true",
                        help="Use class weights for imbalanced data")
    parser.add_argument("--channel-wise", action="store_true",
                        help="Use channel-wise encoder for interpretability")
    parser.add_argument("--encoder-type", type=str, default="cnn",
                        choices=["cnn", "tcn"],
                        help="Encoder type: cnn or tcn")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--num-channels", type=int, default=8,
                        help="Number of sensor channels")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="Training data ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Validation data ratio")
    parser.add_argument("--trim-start", type=int, default=0,
                        help="Start from this time index (absolute)")
    parser.add_argument("--trim-end", type=int, default=0,
                        help="End at this time index (absolute, 0=no limit)")
    parser.add_argument("--frequency-domain", action="store_true",
                        help="Use FFT frequency domain instead of time domain")
    parser.add_argument("--fft-cutoff-hz", type=float, default=0,
                        help="FFT cutoff frequency in Hz (0=all, e.g. 15 for 0-15Hz)")
    parser.add_argument("--subset", type=str, default=None,
                        help="Dataset subset (e.g., 'pure' or 'mixture' for smellnet)")
    parser.add_argument("--lag", type=int, default=0,
                        help="Lag for difference features (0=disabled, typical: 25)")
    parser.add_argument("--dual-view", action="store_true",
                        help="Use dual-view fusion (time + frequency domain)")
    parser.add_argument("--fusion", type=str, default="concat",
                        choices=["concat", "add", "attention"],
                        help="Fusion method for dual-view")
    parser.add_argument("--window-size", type=int, default=0,
                        help="Sliding window size (0=full sequence, e.g. 100)")
    parser.add_argument("--window-stride", type=int, default=0,
                        help="Sliding window stride (0=window_size//2)")
    parser.add_argument("--no-visualization", action="store_true",
                        help="Disable visualization callback (faster training)")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--max-epochs", type=int, default=200,
                        help="Maximum epochs")
    parser.add_argument("--learning-rate", type=float, default=3e-3,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Hardware
    parser.add_argument("--accelerator", type=str, default="gpu",
                        choices=["cpu", "gpu", "auto"],
                        help="Accelerator")
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                        choices=["32", "16-mixed", "bf16-mixed"],
                        help="Training precision")
    
    # Logging
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Log directory")
    parser.add_argument("--experiment-name", type=str, default="cls",
                        help="Experiment name")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name (auto-generated if not specified)")
    parser.add_argument("--use-mlflow", action="store_true",
                        help="Use MLflow logger")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="http://localhost:5000",
                        help="MLflow tracking URI (default: http://localhost:5000)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    L.seed_everything(args.seed)
    torch.set_float32_matmul_precision('medium')
    
    # Get dataset info
    dataset_info = CLASSIFICATION_DATASETS.get(args.dataset, {})
    
    print("=" * 60)
    print("E-nose Gas Classification")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Description: {dataset_info.get('description', 'N/A')}")
    print(f"Task: {dataset_info.get('task', 'classification')}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Num layers: {args.num_layers}")
    
    # Create data module
    datamodule = ClassificationDataModule(
        root=args.root,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        num_channels=args.num_channels,
        download=args.download,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        trim_start=args.trim_start,
        trim_end=args.trim_end,
        frequency_domain=args.frequency_domain,
        fft_cutoff_hz=args.fft_cutoff_hz,
        subset=args.subset,
        window_size=args.window_size,
        window_stride=args.window_stride,
        lag=args.lag,
        dual_view=args.dual_view,
    )
    
    # Log settings
    if args.trim_start > 0 or args.trim_end > 0:
        print(f"Time trimming: start={args.trim_start}, end={args.trim_end}")
    if args.window_size > 0:
        stride = args.window_stride if args.window_stride > 0 else args.window_size // 2
        print(f"Sliding window: size={args.window_size}, stride={stride}")
    if args.lag > 0:
        print(f"Using difference features with lag={args.lag}")
    
    # Setup to get metadata
    datamodule.setup("fit")
    num_classes = datamodule.num_classes
    class_names = datamodule.get_class_names()
    sample_rate = datamodule.sample_rate
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Sample rate: {sample_rate}Hz (from dataset metadata)")
    if args.frequency_domain:
        cutoff_str = f", cutoff={args.fft_cutoff_hz}Hz" if args.fft_cutoff_hz > 0 else ""
        print(f"Using FREQUENCY DOMAIN (FFT magnitude spectrum{cutoff_str})")
    
    # Get class weights if requested
    class_weights = None
    if args.use_class_weights:
        class_weights = datamodule.get_class_weights()
        print(f"Class weights: {class_weights}")
    
    # Create model
    if args.dual_view:
        model = DualViewClassifier(
            num_classes=num_classes,
            in_channels=args.num_channels,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            class_weights=class_weights.tolist() if class_weights is not None else None,
            label_smoothing=args.label_smoothing,
            class_names=class_names,
            encoder_type=args.encoder_type,
            mixup_alpha=args.mixup_alpha,
            fusion=args.fusion,
        )
        print(f"Using Dual-View fusion ({args.fusion}) with time+frequency domain")
    else:
        model = CNN1DClassifier(
            num_classes=num_classes,
            in_channels=args.num_channels,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            class_weights=class_weights.tolist() if class_weights is not None else None,
            label_smoothing=args.label_smoothing,
            class_names=class_names,
            channel_wise=args.channel_wise,
            encoder_type=args.encoder_type,
            mixup_alpha=args.mixup_alpha,
        )
    
    if args.mixup_alpha > 0:
        print(f"Using Mixup augmentation (alpha={args.mixup_alpha})")
    
    if not args.dual_view:
        if args.channel_wise:
            print("Using Channel-Wise encoder for interpretability (per-channel CAM)")
        elif args.encoder_type == "tcn":
            print(f"Using TCN encoder (receptive field: {model.encoder.receptive_field})")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup logging directory
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    # Shorten dataset name for run directory
    short_name = args.dataset.replace("gas_sensor_array_", "").replace("_dataset", "")[:20]
    run_name = args.run_name or f"{short_name}_{args.hidden_dim}d_{args.num_layers}l"
    run_dir = Path(args.log_dir) / args.experiment_name / run_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Run directory: {run_dir}")
    
    # Loggers
    loggers = [
        TensorBoardLogger(save_dir=str(run_dir), name="", version=""),
        CSVLogger(save_dir=str(run_dir), name="csv", version=""),
    ]
    
    # Add MLflow if requested and available
    if args.use_mlflow:
        if HAS_MLFLOW:
            mlflow_logger = MLFlowLogger(
                experiment_name=f"cls_{args.dataset}",
                tracking_uri=args.mlflow_tracking_uri,
                run_name=run_name,
                log_model=True,
            )
            loggers.append(mlflow_logger)
            print(f"MLflow tracking URI: {args.mlflow_tracking_uri}")
        else:
            print("Warning: MLflow not installed, skipping MLflow logging")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="{epoch:03d}-{val/acc:.4f}",
            monitor="val/acc",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/acc",
            patience=args.early_stopping_patience,
            mode="max",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
        # Test on best callback - runs test when best improves
        TestOnBestCallback(),
    ]
    
    # Add visualization callback if not disabled
    if not args.no_visualization:
        callbacks.append(VisualizationCallback(
            output_dir=run_dir / "visualizations",
            class_names=class_names,
            max_samples=64,
            frequency_domain=args.frequency_domain,
        ))
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        precision=args.precision,
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    trainer.fit(model, datamodule=datamodule)
    
    # Test
    print("\n" + "=" * 60)
    print("Testing...")
    print("=" * 60 + "\n")
    
    datamodule.setup("test")
    trainer.test(model, datamodule=datamodule, ckpt_path="best")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best model: {trainer.checkpoint_callback.best_model_path}")
    print(f"Run directory: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
