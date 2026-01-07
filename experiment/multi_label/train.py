#!/usr/bin/env python
"""Training script for multi-label gas classification.

Usage:
    # Train with default settings
    python experiment/multi_label/train.py
    
    # Train with SmellNet local data
    python experiment/multi_label/train.py --smellnet-root /root/SmellNet-iclr
    
    # Train with TCN encoder
    python experiment/multi_label/train.py --encoder tcn --hidden-dim 128
    
    # Train with MLP encoder
    python experiment/multi_label/train.py --encoder mlp --hidden-dim 256
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

from experiment.multi_label.dataset import GasLabelEncoder
from experiment.multi_label.model import MultiLabelClassifier
from experiment.multi_label.datamodule import MultiLabelDataModule


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-label Gas Classification")
    
    # Data arguments
    parser.add_argument("--root", type=str, default=".cache",
                        help="Dataset root directory")
    parser.add_argument("--smellnet-root", type=str, default=None,
                        help="Path to local SmellNet data (optional)")
    parser.add_argument("--download", action="store_true",
                        help="Download missing datasets")
    parser.add_argument("--train-sources", nargs="+", 
                        default=["twin_gas_pure"],
                        help="Training data sources (pure gases: Ethylene, Methane, CO)")
    parser.add_argument("--test-sources", nargs="+",
                        default=["gas_sensor_turbulent", "gas_sensor_dynamic"],
                        help="Test data sources (mixtures)")
    
    # Model arguments
    parser.add_argument("--encoder", type=str, default="tcn",
                        choices=["tcn", "mlp"],
                        help="Encoder architecture")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Number of encoder layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing factor")
    
    # Data processing
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--num-channels", type=int, default=6,
                        help="Number of sensor channels")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--max-epochs", type=int, default=200,
                        help="Maximum epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--early-stopping-patience", type=int, default=30,
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
    parser.add_argument("--experiment-name", type=str, default="multi_label",
                        help="Experiment name")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name (auto-generated if not specified)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    L.seed_everything(args.seed)
    torch.set_float32_matmul_precision('medium')
    
    print("=" * 60)
    print("Multi-label Gas Classification")
    print("=" * 60)
    print(f"Encoder: {args.encoder}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Train sources: {args.train_sources}")
    print(f"Test sources: {args.test_sources}")
    
    # Create data module
    datamodule = MultiLabelDataModule(
        root=args.root,
        train_sources=args.train_sources,
        test_sources=args.test_sources,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        num_channels=args.num_channels,
        download=args.download,
        seed=args.seed,
        smellnet_root=args.smellnet_root,
    )
    
    # Setup to get num_classes
    datamodule.setup("fit")
    num_classes = datamodule.num_classes
    
    print(f"Number of classes: {num_classes}")
    
    # Create model
    model = MultiLabelClassifier(
        num_classes=num_classes,
        in_channels=args.num_channels,
        seq_len=args.max_length,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        encoder_type=args.encoder,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup logging
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    run_name = args.run_name or f"{args.encoder}_{args.hidden_dim}d_{args.num_layers}l"
    run_dir = Path(args.log_dir) / args.experiment_name / run_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Run directory: {run_dir}")
    
    # Loggers
    loggers = [
        TensorBoardLogger(save_dir=str(run_dir), name="", version=""),
        CSVLogger(save_dir=str(run_dir), name="csv", version=""),
    ]
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="{epoch:03d}-{val/f1_micro:.4f}",
            monitor="val/f1_micro",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/f1_micro",
            patience=args.early_stopping_patience,
            mode="max",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]
    
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
    
    # Test on mixture data
    print("\n" + "=" * 60)
    print("Testing on mixture samples...")
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
