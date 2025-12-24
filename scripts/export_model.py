#!/usr/bin/env python
"""Export pretrained model to ONNX/TorchScript for visualization in Netron.

Usage:
    # Export to ONNX
    python scripts/export_model.py --checkpoint logs/enose_pretrain/run_xxx/checkpoints/last.ckpt --format onnx
    
    # Export to TorchScript
    python scripts/export_model.py --checkpoint logs/enose_pretrain/run_xxx/checkpoints/last.ckpt --format torchscript
    
    # View in Netron
    netron model.onnx
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from enose_uci_dataset.pretrain import EnoseVQVAE


class EnoseVQVAEWrapper(nn.Module):
    """Wrapper to extract core model components for export."""
    
    def __init__(self, lightning_model: EnoseVQVAE):
        super().__init__()
        self.encoder = lightning_model.encoder
        self.vq = lightning_model.vq
        self.decoder = lightning_model.decoder
    
    def forward(self, x: torch.Tensor, channel_mask: torch.Tensor, sensor_indices: torch.Tensor):
        z = self.encoder(x, channel_mask, sensor_indices)
        z_q, vq_loss, perplexity = self.vq(z)
        # Decoder only takes z and sensor_indices (no channel_mask)
        x_recon = self.decoder(z_q, sensor_indices)
        return x_recon, z_q, vq_loss, perplexity


def export_onnx(model: EnoseVQVAE, output_path: Path, max_channels: int = 16, seq_len: int = 512):
    """Export model to ONNX format."""
    # Wrap the model to avoid Lightning-specific issues
    wrapper = EnoseVQVAEWrapper(model)
    wrapper = wrapper.cpu()
    wrapper.eval()
    
    # Create dummy inputs on CPU
    batch_size = 1
    
    dummy_x = torch.randn(batch_size, max_channels, seq_len)
    dummy_mask = torch.zeros(batch_size, max_channels, dtype=torch.bool)
    dummy_sensor_indices = torch.zeros(batch_size, max_channels, dtype=torch.long)
    
    # Use dynamo=False for older ONNX export without torch.export
    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_mask, dummy_sensor_indices),
        str(output_path),
        input_names=["x", "channel_mask", "sensor_indices"],
        output_names=["reconstruction", "z_q", "vq_loss", "perplexity"],
        dynamic_axes={
            "x": {0: "batch_size", 1: "num_channels", 2: "seq_len"},
            "channel_mask": {0: "batch_size", 1: "num_channels"},
            "sensor_indices": {0: "batch_size", 1: "num_channels"},
            "reconstruction": {0: "batch_size", 1: "num_channels", 2: "seq_len"},
        },
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,  # Use legacy export to avoid torch.export issues
    )
    print(f"Exported ONNX model to: {output_path}")


def export_torchscript(model: EnoseVQVAE, output_path: Path, max_channels: int = 16, seq_len: int = 512):
    """Export model to TorchScript format."""
    # Wrap the model to avoid Lightning-specific issues
    wrapper = EnoseVQVAEWrapper(model)
    wrapper = wrapper.cpu()
    wrapper.eval()
    
    # Create dummy inputs for tracing on CPU
    batch_size = 1
    dummy_x = torch.randn(batch_size, max_channels, seq_len)
    dummy_mask = torch.zeros(batch_size, max_channels, dtype=torch.bool)
    dummy_sensor_indices = torch.zeros(batch_size, max_channels, dtype=torch.long)
    
    # Use tracing
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapper, (dummy_x, dummy_mask, dummy_sensor_indices))
    
    traced_model.save(str(output_path))
    print(f"Exported TorchScript model to: {output_path}")


def export_state_dict(model: EnoseVQVAE, output_path: Path):
    """Export only model weights (state_dict)."""
    torch.save(model.state_dict(), output_path)
    print(f"Exported state_dict to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export E-nose VQ-VAE model")
    parser.add_argument("--checkpoint", "-c", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output path (auto-generated if not specified)")
    parser.add_argument("--format", "-f", type=str, default="onnx", choices=["onnx", "torchscript", "state_dict"],
                        help="Export format")
    parser.add_argument("--max-channels", type=int, default=16, help="Max channels for dummy input")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length for dummy input")
    
    args = parser.parse_args()
    
    # Load model from checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    model = EnoseVQVAE.load_from_checkpoint(args.checkpoint)
    model.eval()
    
    # Determine output path
    if args.output is None:
        ckpt_path = Path(args.checkpoint)
        if args.format == "onnx":
            output_path = ckpt_path.with_suffix(".onnx")
        elif args.format == "torchscript":
            output_path = ckpt_path.with_suffix(".pt")
        else:
            output_path = ckpt_path.with_suffix(".pth")
    else:
        output_path = Path(args.output)
    
    # Export
    if args.format == "onnx":
        export_onnx(model, output_path, args.max_channels, args.seq_len)
    elif args.format == "torchscript":
        export_torchscript(model, output_path, args.max_channels, args.seq_len)
    else:
        export_state_dict(model, output_path)
    
    print(f"\nTo visualize in Netron:")
    print(f"  netron {output_path}")
    print(f"  # Or open https://netron.app and upload the file")


if __name__ == "__main__":
    main()
