#!/usr/bin/env python
"""Visualize raw SmellNet training data to understand data characteristics."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# SmellNet sensor columns
SENSOR_COLUMNS = ['NO2', 'C2H5OH', 'VOC', 'CO', 'Alcohol', 'LPG']
ENV_COLUMNS = ['Temperature', 'Pressure', 'Humidity', 'Gas_Resistance', 'Altitude']

def load_raw_csv(csv_path: Path) -> pd.DataFrame:
    """Load raw CSV and subtract first row (baseline subtraction)."""
    df = pd.read_csv(csv_path)
    # Original paper subtracts first row
    df_subtracted = df - df.iloc[0]
    return df, df_subtracted

def plot_ingredient_samples(data_dir: Path, ingredients: list, num_samples: int = 3, save_path: str = None):
    """Plot raw sensor data for multiple ingredients."""
    
    num_ingredients = len(ingredients)
    fig, axes = plt.subplots(num_ingredients, 2, figsize=(16, 4 * num_ingredients))
    if num_ingredients == 1:
        axes = axes.reshape(1, -1)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(SENSOR_COLUMNS)))
    
    for i, ingredient in enumerate(ingredients):
        ingredient_dir = data_dir / ingredient
        csv_files = list(ingredient_dir.glob("*.csv"))
        
        # Sample files
        sample_files = random.sample(csv_files, min(num_samples, len(csv_files)))
        
        ax_raw = axes[i, 0]
        ax_sub = axes[i, 1]
        
        for j, csv_file in enumerate(sample_files):
            df_raw, df_sub = load_raw_csv(csv_file)
            
            # Plot raw data (left)
            for k, col in enumerate(SENSOR_COLUMNS):
                if col in df_raw.columns:
                    alpha = 0.3 + 0.7 * (j == 0)  # First sample darker
                    linestyle = ['-', '--', ':'][j % 3]
                    ax_raw.plot(df_raw[col].values, color=colors[k], 
                               alpha=alpha, linestyle=linestyle,
                               label=col if j == 0 else None)
            
            # Plot baseline-subtracted data (right)
            for k, col in enumerate(SENSOR_COLUMNS):
                if col in df_sub.columns:
                    alpha = 0.3 + 0.7 * (j == 0)
                    linestyle = ['-', '--', ':'][j % 3]
                    ax_sub.plot(df_sub[col].values, color=colors[k],
                               alpha=alpha, linestyle=linestyle,
                               label=col if j == 0 else None)
        
        ax_raw.set_title(f'{ingredient} - Raw ({len(csv_files)} files, showing {len(sample_files)})')
        ax_raw.set_xlabel('Time (seconds @ 1Hz)')
        ax_raw.set_ylabel('Sensor Value')
        ax_raw.legend(loc='upper right', fontsize=8)
        ax_raw.grid(True, alpha=0.3)
        
        ax_sub.set_title(f'{ingredient} - Baseline Subtracted')
        ax_sub.set_xlabel('Time (seconds @ 1Hz)')
        ax_sub.set_ylabel('Sensor Value (relative)')
        ax_sub.legend(loc='upper right', fontsize=8)
        ax_sub.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig

def plot_single_sample_all_channels(csv_path: Path, save_path: str = None):
    """Plot all channels of a single sample in detail."""
    df_raw, df_sub = load_raw_csv(csv_path)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    all_cols = SENSOR_COLUMNS + ENV_COLUMNS
    
    for i, col in enumerate(all_cols):
        if i >= len(axes):
            break
        ax = axes[i]
        if col in df_raw.columns:
            ax.plot(df_raw[col].values, 'b-', alpha=0.7, label='Raw')
            ax.plot(df_sub[col].values, 'r-', alpha=0.7, label='Subtracted')
            ax.set_title(col)
            ax.set_xlabel('Time (s)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for i in range(len(all_cols), len(axes)):
        axes[i].set_visible(False)
    
    ingredient = csv_path.parent.name
    plt.suptitle(f'Sample: {ingredient}/{csv_path.name}\nLength: {len(df_raw)} samples @ 1Hz = {len(df_raw)}s', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig

def plot_class_comparison_raw(data_dir: Path, num_classes: int = 8, save_path: str = None):
    """Compare raw signal patterns across multiple classes."""
    
    ingredients = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])[:num_classes]
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    # Plot one sample per class, overlay on same axes for each sensor
    sensor_data = {col: [] for col in SENSOR_COLUMNS}
    
    for i, ingredient in enumerate(ingredients):
        ingredient_dir = data_dir / ingredient
        csv_files = list(ingredient_dir.glob("*.csv"))
        if csv_files:
            df_raw, df_sub = load_raw_csv(csv_files[0])
            for col in SENSOR_COLUMNS:
                if col in df_sub.columns:
                    sensor_data[col].append((ingredient, df_sub[col].values))
    
    for i, col in enumerate(SENSOR_COLUMNS):
        if i >= len(axes):
            break
        ax = axes[i]
        for j, (ingredient, values) in enumerate(sensor_data[col]):
            ax.plot(values, color=colors[j], alpha=0.8, label=ingredient)
        ax.set_title(f'{col}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Value (baseline subtracted)')
        ax.grid(True, alpha=0.3)
    
    # Add legend to last axes
    axes[-2].legend(loc='center', fontsize=8, ncol=2)
    axes[-1].set_visible(False)
    axes[-2].axis('off')
    
    plt.suptitle(f'Sensor Response Comparison ({num_classes} Classes, Baseline Subtracted)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def analyze_data_statistics(data_dir: Path):
    """Analyze data statistics across all samples."""
    
    ingredients = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    
    lengths = []
    stats = []
    
    for ingredient in ingredients:
        ingredient_dir = data_dir / ingredient
        csv_files = list(ingredient_dir.glob("*.csv"))
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            lengths.append(len(df))
            stats.append({
                'ingredient': ingredient,
                'file': csv_file.name,
                'length': len(df),
            })
    
    print("=" * 60)
    print("Data Statistics")
    print("=" * 60)
    print(f"Total ingredients: {len(ingredients)}")
    print(f"Total CSV files: {len(stats)}")
    print(f"Sample lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
    print(f"At 1Hz sampling: {np.mean(lengths):.0f}s = {np.mean(lengths)/60:.1f} min per sample")
    print("=" * 60)
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize SmellNet raw data")
    parser.add_argument("--data-dir", type=str, default="/root/SmellNet-iclr/data/base_training",
                        help="Path to training data directory")
    parser.add_argument("--output-dir", type=str, default="/root/UCI_dataset/logs/smellnet_raw_viz",
                        help="Output directory for visualizations")
    parser.add_argument("--ingredients", type=str, nargs="+", 
                        default=["allspice", "banana", "garlic", "lemon"],
                        help="Ingredients to visualize")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Analyze statistics
    analyze_data_statistics(data_dir)
    
    # 2. Plot ingredient samples
    plot_ingredient_samples(
        data_dir, 
        args.ingredients,
        num_samples=3,
        save_path=str(output_dir / "ingredient_samples.png")
    )
    
    # 3. Plot single sample in detail
    sample_csv = data_dir / args.ingredients[0] / list((data_dir / args.ingredients[0]).glob("*.csv"))[0].name
    plot_single_sample_all_channels(
        sample_csv,
        save_path=str(output_dir / "single_sample_detail.png")
    )
    
    # 4. Plot class comparison
    plot_class_comparison_raw(
        data_dir,
        num_classes=8,
        save_path=str(output_dir / "class_comparison_raw.png")
    )
