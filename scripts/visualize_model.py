#!/usr/bin/env python
"""Generate paper-quality model architecture diagrams using torchview."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from enose_uci_dataset.pretrain import EnoseVQVAE


def visualize_model(
    checkpoint_path: str = None,
    output_path: str = "model_architecture",
    max_channels: int = 16,
    seq_len: int = 512,
    depth: int = 3,
    format: str = "png",
):
    """Generate hierarchical model architecture diagram.
    
    Args:
        checkpoint_path: Path to checkpoint (optional, creates new model if None)
        output_path: Output file path (without extension)
        max_channels: Number of input channels
        seq_len: Sequence length
        depth: Depth of module hierarchy to show (1=top-level, higher=more detail)
        format: Output format (png, pdf, svg)
    """
    from torchview import draw_graph
    
    # Load or create model
    if checkpoint_path:
        print(f"Loading model from: {checkpoint_path}")
        model = EnoseVQVAE.load_from_checkpoint(checkpoint_path, map_location="cpu")
    else:
        print("Creating new model with default config...")
        model = EnoseVQVAE(
            d_model=128,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            patch_size=16,
            num_embeddings=512,
            max_channels=max_channels,
        )
    
    model.eval()
    
    # Create dummy inputs
    x = torch.randn(1, max_channels, seq_len)
    channel_mask = torch.zeros(1, max_channels, dtype=torch.bool)
    sensor_indices = torch.zeros(1, max_channels, dtype=torch.long)
    
    print(f"Generating architecture diagram (depth={depth})...")
    
    # Generate hierarchical graph
    model_graph = draw_graph(
        model,
        input_data=(x, channel_mask, sensor_indices),
        depth=depth,
        expand_nested=True,
        hide_inner_tensors=True,
        hide_module_functions=False,
        roll=True,  # Collapse repeated modules
        graph_name="EnoseVQVAE",
        graph_dir="TB",  # Top to Bottom layout
        device="cpu",
    )
    
    # Save graph
    output_file = f"{output_path}"
    model_graph.visual_graph.render(output_file, format=format, cleanup=True)
    print(f"✓ Saved architecture diagram to: {output_file}.{format}")
    
    # Also generate component-level diagrams
    print("\nGenerating component diagrams...")
    
    # Encoder diagram
    encoder_graph = draw_graph(
        model.encoder,
        input_data=(x, channel_mask, sensor_indices),
        depth=depth,
        expand_nested=True,
        hide_inner_tensors=True,
        roll=True,
        graph_name="EnoseEncoder",
        graph_dir="TB",
        device="cpu",
    )
    encoder_file = f"{output_path}_encoder"
    encoder_graph.visual_graph.render(encoder_file, format=format, cleanup=True)
    print(f"✓ Saved encoder diagram to: {encoder_file}.{format}")
    
    # VQ diagram
    z_dummy = torch.randn(1, max_channels, seq_len // 16, 128)
    vq_graph = draw_graph(
        model.vq,
        input_data=z_dummy,
        depth=depth,
        expand_nested=True,
        hide_inner_tensors=True,
        graph_name="VectorQuantizer",
        graph_dir="TB",
        device="cpu",
    )
    vq_file = f"{output_path}_vq"
    vq_graph.visual_graph.render(vq_file, format=format, cleanup=True)
    print(f"✓ Saved VQ diagram to: {vq_file}.{format}")
    
    # Decoder diagram
    decoder_graph = draw_graph(
        model.decoder,
        input_data=(z_dummy, sensor_indices),
        depth=depth,
        expand_nested=True,
        hide_inner_tensors=True,
        roll=True,
        graph_name="EnoseDecoder",
        graph_dir="TB",
        device="cpu",
    )
    decoder_file = f"{output_path}_decoder"
    decoder_graph.visual_graph.render(decoder_file, format=format, cleanup=True)
    print(f"✓ Saved decoder diagram to: {decoder_file}.{format}")
    
    print(f"\nAll diagrams saved to: {Path(output_path).parent}")


def main():
    parser = argparse.ArgumentParser(description="Generate model architecture diagrams")
    parser.add_argument("-c", "--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("-o", "--output", type=str, default="model_architecture",
                        help="Output file path (without extension)")
    parser.add_argument("--max-channels", type=int, default=16,
                        help="Number of input channels")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Input sequence length")
    parser.add_argument("--depth", type=int, default=3,
                        help="Module hierarchy depth (1=top-level, higher=more detail)")
    parser.add_argument("--format", type=str, default="png",
                        choices=["png", "pdf", "svg"],
                        help="Output format")
    
    args = parser.parse_args()
    
    visualize_model(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        max_channels=args.max_channels,
        seq_len=args.seq_len,
        depth=args.depth,
        format=args.format,
    )


if __name__ == "__main__":
    main()
