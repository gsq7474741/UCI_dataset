#!/usr/bin/env python
"""Downstream task training and evaluation script.

This script handles:
1. Training downstream models (concentration regression, etc.)
2. Evaluating pretrained backbone + downstream on test set

Backbone training should use pretrain.py, not this script.

Usage:
    # Train downstream model only
    python scripts/downstream_train.py --mode train --downstream-model mlp
    
    # Evaluate with pretrained backbone
    python scripts/downstream_train.py --mode eval \
        --backbone-ckpt path/to/backbone.ckpt \
        --downstream-ckpt path/to/downstream.ckpt
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger, MLFlowLogger
import shutil

from enose_uci_dataset.pretrain import EnoseVQVAE, MLPAutoencoder, TCNAutoencoder
from enose_uci_dataset.pretrain.downstream import (
    BaseDownstreamModel,
    ConcentrationRegressionTask,
    MLPRegressor,
    TCNRegressor,
    DownstreamEvaluator,
)


BACKBONE_MODELS = {
    "vqvae": EnoseVQVAE,
    "mlp": MLPAutoencoder,
    "tcn": TCNAutoencoder,
}

DOWNSTREAM_MODELS = {
    "mlp": MLPRegressor,
    "tcn": TCNRegressor,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Downstream Task Training & Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train downstream model
  python scripts/downstream_train.py --mode train --downstream-model mlp --max-epochs 200
  
  # Evaluate backbone + downstream
  python scripts/downstream_train.py --mode eval \\
      --backbone-model tcn --backbone-ckpt logs/backbone.ckpt \\
      --downstream-ckpt logs/downstream.ckpt
"""
    )
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "eval"],
                        help="Mode: train downstream model or eval with backbone")
    
    # Task configuration
    parser.add_argument("--task", type=str, default="concentration",
                        choices=["concentration"],
                        help="Downstream task type")
    parser.add_argument("--data-root", type=str, default=".cache",
                        help="Dataset root directory")
    parser.add_argument("--seq-len", type=int, default=1000,
                        help="Sequence length")
    
    # Model selection
    parser.add_argument("--backbone-model", type=str, default="tcn",
                        choices=list(BACKBONE_MODELS.keys()),
                        help="Backbone model type (for eval mode)")
    parser.add_argument("--downstream-model", type=str, default="mlp",
                        choices=list(DOWNSTREAM_MODELS.keys()),
                        help="Downstream model type")
    
    # Checkpoint paths
    parser.add_argument("--backbone-ckpt", type=str, default=None,
                        help="Path to pretrained backbone checkpoint (required for eval)")
    parser.add_argument("--downstream-ckpt", type=str, default=None,
                        help="Path to pretrained downstream checkpoint (for eval or resume)")
    
    # Downstream model hyperparameters
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden dimension for downstream model")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    
    # Training configuration
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--max-epochs", type=int, default=200,
                        help="Max epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--early-stopping-patience", type=int, default=30,
                        help="Early stopping patience")
    parser.add_argument("--optimizer", type=str, default="soap",
                        choices=["adamw", "soap"],
                        help="Optimizer type")
    
    # Evaluation configuration
    parser.add_argument("--mask-channels", type=int, nargs="+", default=[0],
                        help="Channels to mask during evaluation")
    
    # Hardware
    parser.add_argument("--accelerator", type=str, default="auto",
                        choices=["cpu", "gpu", "auto"],
                        help="Accelerator")
    parser.add_argument("--devices", type=int, default=1,
                        help="Number of devices")
    
    # Logging
    parser.add_argument("--log-dir", type=str, default="logs/downstream",
                        help="Log directory")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name")
    parser.add_argument("--use-mlflow", action="store_true",
                        help="Use MLflow logging")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="http://localhost:5000",
                        help="MLflow tracking URI")
    
    return parser.parse_args()


# Fixed path for best downstream model (auto-loaded by backbone probe)
BEST_DOWNSTREAM_CKPT = "logs/downstream/best_downstream.ckpt"


def create_task(args) -> ConcentrationRegressionTask:
    """Create downstream task."""
    if args.task == "concentration":
        task = ConcentrationRegressionTask(
            data_root=args.data_root,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    task.setup()
    return task


def load_backbone(args) -> Optional[L.LightningModule]:
    """Load pretrained backbone model."""
    if not args.backbone_ckpt:
        return None
    
    ckpt_path = Path(args.backbone_ckpt)
    if not ckpt_path.exists():
        print(f"Warning: Backbone checkpoint not found: {ckpt_path}")
        return None
    
    model_cls = BACKBONE_MODELS[args.backbone_model]
    print(f"Loading {args.backbone_model.upper()} backbone from: {ckpt_path}")
    return model_cls.load_from_checkpoint(str(ckpt_path))


def create_downstream(args, task: ConcentrationRegressionTask) -> BaseDownstreamModel:
    """Create or load downstream model."""
    model_cls = DOWNSTREAM_MODELS[args.downstream_model]
    
    if args.downstream_ckpt and Path(args.downstream_ckpt).exists():
        print(f"Loading downstream from: {args.downstream_ckpt}")
        return model_cls.load_from_checkpoint(args.downstream_ckpt)
    
    print(f"Creating new {args.downstream_model.upper()} downstream model")
    return model_cls(
        in_channels=task.get_num_channels(),
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
    )


def train_downstream(args, task: ConcentrationRegressionTask) -> BaseDownstreamModel:
    """Train downstream model."""
    print("\n" + "=" * 60)
    print("Training Downstream Model")
    print("=" * 60)
    
    model = create_downstream(args, task)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Setup trainer
    run_name = args.run_name or f"downstream_{args.downstream_model}_{args.task}"
    
    callbacks = [
        ModelCheckpoint(
            dirpath=f"{args.log_dir}/{run_name}/checkpoints",
            filename="best-{epoch:02d}-{val/r2:.4f}",
            monitor="val/r2",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/r2",
            mode="max",
            patience=args.early_stopping_patience,
        ),
    ]
    
    loggers = [
        TensorBoardLogger(args.log_dir, name=run_name),
        CSVLogger(args.log_dir, name=run_name),
    ]
    
    if args.use_mlflow:
        mlflow_logger = MLFlowLogger(
            experiment_name="downstream_tasks",
            tracking_uri=args.mlflow_tracking_uri,
            run_name=run_name,
        )
        loggers.append(mlflow_logger)
        print(f"MLflow tracking URI: {args.mlflow_tracking_uri}")
    
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=True,
    )
    
    # Train
    trainer.fit(
        model,
        train_dataloaders=task.train_dataloader(),
        val_dataloaders=task.val_dataloader(),
    )
    
    # Load best checkpoint and save to fixed path
    best_path = callbacks[0].best_model_path
    if best_path:
        print(f"Loading best checkpoint: {best_path}")
        model = type(model).load_from_checkpoint(best_path)
        
        # Copy to fixed path for auto-loading by backbone probe
        fixed_path = Path(BEST_DOWNSTREAM_CKPT)
        fixed_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_path, fixed_path)
        print(f"Best model saved to: {fixed_path} (auto-loaded by backbone probe)")
    
    return model


def evaluate(
    args,
    backbone: L.LightningModule,
    downstream: BaseDownstreamModel,
    task: ConcentrationRegressionTask,
) -> Dict[str, float]:
    """Evaluate backbone + downstream on test set."""
    print("\n" + "=" * 60)
    print("Evaluating Models")
    print("=" * 60)
    
    evaluator = DownstreamEvaluator(
        backbone_model=backbone,
        downstream_model=downstream,
        task=task,
        mask_channels=args.mask_channels,
    )
    
    # Full evaluation
    results = evaluator.evaluate()
    
    print("\n--- Overall Results ---")
    print(f"R² (original input):      {results['r2_original']:.4f}")
    print(f"R² (reconstructed input): {results['r2_reconstructed']:.4f}")
    print(f"R² (zeroed input):        {results['r2_zeroed']:.4f}")
    print(f"R² degradation:           {results['r2_degradation']:.4f}")
    print(f"R² recovery rate:         {results['r2_recovery_rate']:.4f}")
    
    # Masked imputation evaluation
    if args.mask_channels:
        imputation_results = evaluator.evaluate_masked_imputation(
            mask_channels=args.mask_channels
        )
        
        print(f"\n--- Masked Imputation (channels {args.mask_channels}) ---")
        print(f"R² (full input):          {imputation_results['r2_full_input']:.4f}")
        print(f"R² (imputed):             {imputation_results['r2_imputed']:.4f}")
        print(f"R² (zeroed):              {imputation_results['r2_zeroed']:.4f}")
        print(f"R² recovery vs full:      {imputation_results['r2_recovery_vs_full']:.4f}")
        print(f"R² gain over zeroed:      {imputation_results['r2_gain_over_zeroed']:.4f}")
        print(f"Masked channel MSE:       {imputation_results['masked_channel_recon_mse']:.4f}")
        
        results.update({f"imputation_{k}": v for k, v in imputation_results.items()})
    
    return results


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Downstream Task Training & Evaluation")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Task: {args.task}")
    print(f"Downstream model: {args.downstream_model}")
    
    # Create task
    task = create_task(args)
    
    if args.mode == "train":
        # Train downstream model
        downstream = train_downstream(args, task)
        print(f"\nDownstream training complete!")
        print(f"Checkpoint saved to: {args.log_dir}/downstream_{args.downstream_model}_{args.task}/checkpoints/")
    
    elif args.mode == "eval":
        # Evaluate with backbone
        if not args.backbone_ckpt:
            print("Error: --backbone-ckpt required for eval mode")
            sys.exit(1)
        
        backbone = load_backbone(args)
        if backbone is None:
            sys.exit(1)
        
        downstream = create_downstream(args, task)
        if args.downstream_ckpt is None:
            print("Warning: No --downstream-ckpt provided, using untrained downstream model")
        
        evaluate(args, backbone, downstream, task)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
