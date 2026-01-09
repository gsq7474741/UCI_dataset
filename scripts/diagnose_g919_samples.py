#!/usr/bin/env python
"""Diagnose G919 dataset samples - plot all train/val/test samples to find anomalies."""

import sys
sys.path.insert(0, '/root/UCI_dataset')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd


def load_and_plot_samples(data_dir: Path, output_dir: Path, split_name: str):
    """Load all CSV files and plot each sample."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_files = sorted(data_dir.glob("*.csv"))
    print(f"[{split_name}] Found {len(csv_files)} CSV files in {data_dir}")
    
    if len(csv_files) == 0:
        print(f"[{split_name}] No CSV files found!")
        return
    
    # Collect all samples info
    samples_info = []
    all_data = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Get numeric columns (sensor channels)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) == 0:
                continue
            
            data = df[numeric_cols].values  # [T, C]
            samples_info.append({
                'file': csv_file.name,
                'length': len(data),
                'channels': len(numeric_cols),
                'end_values': data[-1] if len(data) > 0 else None,
                'start_values': data[0] if len(data) > 0 else None,
            })
            all_data.append((csv_file.name, data))
            
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    print(f"[{split_name}] Loaded {len(all_data)} samples")
    
    # Print length statistics
    lengths = [s['length'] for s in samples_info]
    print(f"[{split_name}] Length stats: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
    
    # Find anomalies: samples that don't return to baseline
    # Check if end value is significantly different from start
    anomalies = []
    for info in samples_info:
        if info['end_values'] is not None and info['start_values'] is not None:
            diff = np.abs(info['end_values'] - info['start_values'])
            if np.max(diff) > 1.0:  # Threshold for "not returning to baseline"
                anomalies.append(info)
                print(f"[{split_name}] ANOMALY: {info['file']} - end-start diff max={np.max(diff):.2f}")
    
    # Plot all samples in one figure (per channel)
    num_channels = all_data[0][1].shape[1] if all_data else 8
    fig, axes = plt.subplots(num_channels, 1, figsize=(16, 2.5 * num_channels))
    if num_channels == 1:
        axes = [axes]
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_data)))
    
    for idx, (filename, data) in enumerate(all_data):
        T, C = data.shape
        for ch in range(min(C, num_channels)):
            axes[ch].plot(data[:, ch], color=colors[idx], alpha=0.7, linewidth=0.8, label=filename if ch == 0 else None)
    
    for ch in range(num_channels):
        axes[ch].set_ylabel(f'Ch{ch}')
        axes[ch].set_xlabel('Time')
        axes[ch].grid(True, alpha=0.3)
    
    fig.suptitle(f'{split_name}: All {len(all_data)} samples (each color = one sample)', fontsize=14)
    plt.tight_layout()
    
    save_path = output_dir / f'{split_name}_all_samples.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[{split_name}] Saved: {save_path}")
    
    # Plot individual samples for anomalies
    for info in anomalies:
        filename = info['file']
        for name, data in all_data:
            if name == filename:
                fig, axes = plt.subplots(data.shape[1], 1, figsize=(12, 2 * data.shape[1]))
                if data.shape[1] == 1:
                    axes = [axes]
                for ch in range(data.shape[1]):
                    axes[ch].plot(data[:, ch], 'b-', linewidth=1)
                    axes[ch].axhline(data[0, ch], color='g', linestyle='--', alpha=0.5, label='start')
                    axes[ch].axhline(data[-1, ch], color='r', linestyle='--', alpha=0.5, label='end')
                    axes[ch].set_ylabel(f'Ch{ch}')
                    axes[ch].legend(loc='upper right')
                fig.suptitle(f'ANOMALY: {filename} (len={info["length"]})', fontsize=12)
                plt.tight_layout()
                save_path = output_dir / f'anomaly_{filename.replace(".csv", ".png")}'
                fig.savefig(save_path, dpi=150)
                plt.close(fig)
                print(f"[{split_name}] Saved anomaly plot: {save_path}")
                break
    
    return samples_info, anomalies


def main():
    # G919 data directories (lowercase)
    base_dir = Path("/root/UCI_dataset/.cache/g919_55")
    output_dir = Path("/root/UCI_dataset/logs/diagnose_g919")
    
    # Check directories
    single_dir = base_dir / "single"  # Processed single files
    raw_dir = base_dir / "raw"  # Raw files
    
    print("=" * 60)
    print("G919 Dataset Diagnosis")
    print("=" * 60)
    
    # Check what directories exist
    for d in [single_dir, raw_dir]:
        print(f"{d}: exists={d.exists()}")
    
    # List subdirectories in single
    if single_dir.exists():
        subdirs = [d for d in single_dir.iterdir() if d.is_dir()]
        print(f"Subdirs in single: {[d.name for d in subdirs]}")
        
        # Load from each category subdirectory
        for subdir in sorted(subdirs):
            # Check for train/test subdirs
            for split_subdir in subdir.iterdir():
                if split_subdir.is_dir():
                    load_and_plot_samples(split_subdir, output_dir, f"{subdir.name}_{split_subdir.name}")
    
    print("=" * 60)
    print(f"Diagnosis complete. Check {output_dir} for plots.")
    print("=" * 60)


if __name__ == "__main__":
    main()
