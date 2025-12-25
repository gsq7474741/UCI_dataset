#!/usr/bin/env python
"""CLI script for e-nose pretraining.

Usage:
    # Basic pretraining with all datasets
    python scripts/pretrain.py
    
    # Custom datasets
    python scripts/pretrain.py --datasets twin_gas_sensor_arrays gas_sensors_for_home_activity_monitoring
    
    # GPU training with mixed precision
    python scripts/pretrain.py --accelerator gpu --precision 16-mixed
    
    # Resume from checkpoint
    python scripts/pretrain.py --resume-from logs/checkpoints/enose_pretrain/last.ckpt
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from enose_uci_dataset.pretrain import create_trainer, EnoseVQVAE, EnosePretrainingDataModule


def parse_args():
    parser = argparse.ArgumentParser(description="E-nose VQ-VAE Pretraining")
    
    # "alcohol_qcm_sensor_dataset": AlcoholQCMSensor,
    # "gas_sensor_array_drift_dataset_at_different_concentrations": GasSensorArrayDrift,
    # "gas_sensor_array_exposed_to_turbulent_gas_mixtures": GasSensorTurbulent,
    # "gas_sensor_array_low_concentration": GasSensorLowConcentration,
    # "gas_sensor_array_temperature_modulation": GasSensorTemperatureModulation,
    # "gas_sensor_array_under_dynamic_gas_mixtures": GasSensorDynamic,
    # "gas_sensor_array_under_flow_modulation": GasSensorFlowModulation,
    # "gas_sensors_for_home_activity_monitoring": GasSensorsForHomeActivityMonitoring,
    # "twin_gas_sensor_arrays": TwinGasSensorArrays,

    # Data arguments
    parser.add_argument("--root", type=str, default=".cache", help="Dataset root directory")
    parser.add_argument("--datasets", nargs="+", default=None, 
                        help="Datasets to use (default: all available)")
    parser.add_argument("--download", action="store_true", help="Download missing datasets")
    
    # Model arguments
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--num-encoder-layers", type=int, default=4, help="Encoder layers")
    parser.add_argument("--num-decoder-layers", type=int, default=4, help="Decoder layers")
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size")
    parser.add_argument("--num-embeddings", type=int, default=512, help="VQ codebook size")
    parser.add_argument("--commitment-cost", type=float, default=0.25, help="VQ commitment cost")
    parser.add_argument("--mask-ratio", type=float, default=0.25, help="Channel mask ratio")
    parser.add_argument("--lambda-visible", type=float, default=1.0, help="Weight for visible channel reconstruction loss")
    parser.add_argument("--lambda-masked", type=float, default=1.0, help="Weight for masked channel prediction loss")
    parser.add_argument("--disable-vq", action="store_true", help="Disable VQ layer (pure autoencoder)")
    parser.add_argument("--loss-type", type=str, default="mse",
                        choices=["mse", "mae", "huber", "cosine", "correlation", "mse_corr", "mse_mmd"],
                        help="Reconstruction loss type")
    parser.add_argument("--huber-delta", type=float, default=1.0, help="Delta for Huber loss")
    parser.add_argument("--optimizer", type=str, default="soap",
                        choices=["adamw", "muon", "soap"],
                        help="Optimizer type")
    
    # LR scheduler arguments
    parser.add_argument("--lr-scheduler", type=str, default="cosine_warmup", 
                        choices=["cosine_warmup", "cosine", "constant", "onecycle", "plateau", "warmup_cosine", "cosine_restart_decay"],
                        help="LR scheduler type")
    parser.add_argument("--lr-warmup-steps", type=int, default=200, help="Warmup steps for cosine_warmup scheduler")
    parser.add_argument("--lr-T-mult", type=int, default=2, help="T_mult for cosine_warmup scheduler")
    parser.add_argument("--lr-min", type=float, default=1e-5, help="Minimum learning rate")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-epochs", type=int, default=500, help="Max epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length (1024 aligns with downstream TCN seq_len=1000)")
    parser.add_argument("--max-channels", type=int, default=16, help="Max channels")
    parser.add_argument("--num-workers", type=int, default=16, help="DataLoader workers")
    parser.add_argument("--early-stopping-patience", type=int, default=300, help="Early stopping patience (epochs)")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1, help="Gradient accumulation steps (effective batch = batch_size * accumulate)")
    parser.add_argument("--val-check-interval", type=float, default=None, help="Validation check interval (None for every epoch, 0.5 for twice per epoch)")
    parser.add_argument("--checkpoint-monitor", type=str, default="val/loss", help="Metric to monitor for checkpointing")
    
    # Hardware arguments
    parser.add_argument("--accelerator", type=str, default="gpu", 
                        choices=["cpu", "gpu", "auto"], help="Accelerator")
    parser.add_argument("--devices", type=str, default="auto", help="Number of devices")
    parser.add_argument("--precision", type=str, default="bf16-mixed", 
                        choices=["32", "16-mixed", "bf16-mixed"], help="Precision")
    
    # Logging arguments
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--experiment-name", type=str, default="enose_pretrain", 
                        help="Experiment name")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name (default: experiment_name_YYYYMMDD_HHMMSS)")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--use-mlflow", action="store_true", help="Use MLflow logging")
    parser.add_argument("--mlflow-tracking-uri", type=str, default='http://localhost:5000',
                        help="MLflow tracking URI (default: http://localhost:5000)")
    
    # Export arguments
    parser.add_argument("--export-best", action="store_true", default=True,
                        help="Auto-export best model for Netron (default: True)")
    parser.add_argument("--no-export-best", action="store_false", dest="export_best",
                        help="Disable auto-export of best model")
    parser.add_argument("--export-format", type=str, default="torchscript",
                        choices=["onnx", "torchscript"], help="Export format (torchscript recommended)")
    
    # Resume
    parser.add_argument("--resume-from", type=str, default=None, help="Checkpoint to resume")
    
    # Downstream probe arguments
    parser.add_argument("--enable-probe", action="store_true", 
                        help="Enable downstream task probe during validation")
    parser.add_argument("--probe-every-n-epochs", type=int, default=10,
                        help="Run downstream probe every N epochs")
    parser.add_argument("--tcn-checkpoint", type=str, default='/root/UCI_dataset/runs/tcn_baseline_normalized_man/version_0/checkpoints/epochepoch=614-val_r2val/r2=0.7372.ckpt',
                        help="Path to trained TCN checkpoint for R² evaluation (must use input_norm=per_sample)")
    parser.add_argument("--probe-mask-channels", type=int, nargs="+", default=[0, 1],
                        help="Channels to mask during probing (default: [0, 1])")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("E-nose VQ-VAE Pretraining")
    print("=" * 60)
    
    # Parse devices
    devices = args.devices
    if devices != "auto":
        try:
            devices = int(devices)
        except ValueError:
            pass
    

    # Create data module
    print(f"\nLoading datasets from: {args.root}")
    datamodule = EnosePretrainingDataModule(
        root=args.root,
        datasets=args.datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        max_channels=args.max_channels,
        download=args.download,
    )
    
    # Setup to see dataset info
    datamodule.setup()
    
    # Create or load model
    if args.resume_from:
        print(f"\nResuming from: {args.resume_from}")
        model = EnoseVQVAE.load_from_checkpoint(args.resume_from)
    else:
        print(f"\nCreating new model:")
        print(f"  d_model: {args.d_model}")
        print(f"  encoder_layers: {args.num_encoder_layers}")
        print(f"  decoder_layers: {args.num_decoder_layers}")
        print(f"  patch_size: {args.patch_size}")
        print(f"  codebook_size: {args.num_embeddings}")
        print(f"  mask_ratio: {args.mask_ratio}")
        
        model = EnoseVQVAE(
            d_model=args.d_model,
            nhead=8,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.d_model * 4,
            dropout=0.1,
            patch_size=args.patch_size,
            num_embeddings=args.num_embeddings,
            commitment_cost=args.commitment_cost,
            max_channels=args.max_channels,
            learning_rate=args.learning_rate,
            mask_ratio=args.mask_ratio,
            lambda_visible=args.lambda_visible,
            lambda_masked=args.lambda_masked,
            disable_vq=args.disable_vq,
            loss_type=args.loss_type,
            huber_delta=args.huber_delta,
            optimizer_type=args.optimizer,
            lr_scheduler=args.lr_scheduler,
            lr_warmup_steps=args.lr_warmup_steps,
            lr_T_mult=args.lr_T_mult,
            lr_min=args.lr_min,
        )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  total_params: {total_params:,}")
    print(f"  trainable_params: {trainable_params:,}")
    
    # Create trainer
    print(f"\nTraining config:")
    print(f"  accelerator: {args.accelerator}")
    print(f"  devices: {devices}")
    print(f"  precision: {args.precision}")
    print(f"  max_epochs: {args.max_epochs}")
    print(f"  batch_size: {args.batch_size}")
    
    # Enable Tensor Core optimization (also helps with numerical stability in mixed precision)
    import torch
    torch.set_float32_matmul_precision('medium')
    
    # Prepare additional callbacks
    additional_callbacks = []
    if args.enable_probe:
        from enose_uci_dataset.pretrain import DownstreamProbeCallback
        probe_callback = DownstreamProbeCallback(
            twin_gas_root=args.root,
            tcn_checkpoint=args.tcn_checkpoint,
            probe_every_n_epochs=args.probe_every_n_epochs,
            mask_channels=args.probe_mask_channels,
        )
        additional_callbacks.append(probe_callback)
        print(f"Downstream probe enabled: every {args.probe_every_n_epochs} epochs, mask channels {args.probe_mask_channels}")
    
    trainer, run_dir = create_trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=devices,
        precision=args.precision,
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        use_wandb=args.use_wandb,
        use_mlflow=args.use_mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        early_stopping_patience=args.early_stopping_patience,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_check_interval=args.val_check_interval,
        checkpoint_monitor=args.checkpoint_monitor,
        export_best=args.export_best,
        export_format=args.export_format,
        max_channels=args.max_channels,
        additional_callbacks=additional_callbacks,
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    trainer.fit(model, datamodule=datamodule)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best model: {trainer.checkpoint_callback.best_model_path}")
    print(f"Run directory: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
