#!/usr/bin/env python3
"""Visualize G919 split data - compare low/mid/high concentration responses."""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# Set up matplotlib
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def load_split_data(base_path: Path, odor_name: str, is_test_interp: bool = False):
    """Load low/mid/high concentration data for an odor."""
    data = {}
    for conc in ['low', 'mid', 'high']:
        if is_test_interp:
            csv_path = base_path / f"{odor_name}_interp_{conc}.csv"
        else:
            csv_path = base_path / f"{odor_name}_{conc}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            data[conc] = df
    return data

def plot_odor_comparison(data: dict, odor_name: str, output_path: Path):
    """Plot sensor responses for low/mid/high concentrations."""
    if not data:
        return
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    colors = {'low': 'blue', 'mid': 'green', 'high': 'red'}
    alphas = {'low': 0.7, 'mid': 0.7, 'high': 0.7}
    
    sensor_cols = [f"sen.{i}" for i in range(1, 9)]
    
    for i, sensor in enumerate(sensor_cols):
        ax = axes[i]
        for conc, df in data.items():
            if sensor in df.columns:
                time = df.iloc[:, 0].values  # First column is time
                values = df[sensor].values
                ax.plot(time, values, color=colors[conc], alpha=alphas[conc], 
                       label=conc, linewidth=1)
        
        ax.set_title(f"Sensor {i+1}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Response")
        if i == 0:
            ax.legend()
    
    fig.suptitle(f"{odor_name} - Low/Mid/High Concentration Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_category_overview(data_dir: Path, category: str, output_path: Path, 
                          split: str = 'train', is_test_interp: bool = False):
    """Plot overview of all odors in a category.
    
    Args:
        data_dir: Base data directory
        category: Category name
        output_path: Output image path
        split: 'train' or 'test'
        is_test_interp: If True, look for *_interp_*.csv files (test interpolated)
    """
    if split == 'train':
        sub_dir = data_dir / category / f"{category}s_train"
        if not sub_dir.exists():
            sub_dir = data_dir / category / f"{category}_train"
    else:
        sub_dir = data_dir / category / f"{category}s_test"
        if not sub_dir.exists():
            sub_dir = data_dir / category / f"{category}_test"
    
    if not sub_dir.exists():
        return
    
    # Find unique odor names
    odor_names = set()
    if is_test_interp:
        for csv_path in sub_dir.glob("*_interp_low.csv"):
            odor_name = csv_path.stem.replace("_interp_low", "")
            odor_names.add(odor_name)
    else:
        for csv_path in sub_dir.glob("*_low.csv"):
            odor_name = csv_path.stem.replace("_low", "")
            odor_names.add(odor_name)
    
    odor_names = sorted(odor_names)[:6]  # Limit to 6 odors for readability
    
    if not odor_names:
        return
    
    n_odors = len(odor_names)
    fig, axes = plt.subplots(n_odors, 3, figsize=(15, 3 * n_odors))
    if n_odors == 1:
        axes = axes.reshape(1, -1)
    
    conc_labels = ['Low', 'Mid', 'High']
    conc_keys = ['low', 'mid', 'high']
    
    for row, odor_name in enumerate(odor_names):
        data = load_split_data(sub_dir, odor_name, is_test_interp=is_test_interp)
        
        for col, (conc_label, conc_key) in enumerate(zip(conc_labels, conc_keys)):
            ax = axes[row, col]
            
            if conc_key in data:
                df = data[conc_key]
                time = df.iloc[:, 0].values
                
                # Plot all 8 sensors
                for i in range(1, 9):
                    sensor_col = f"sen.{i}"
                    if sensor_col in df.columns:
                        ax.plot(time, df[sensor_col].values, label=f"S{i}", linewidth=0.8)
            
            if row == 0:
                ax.set_title(f"{conc_label} Concentration")
            if col == 0:
                ax.set_ylabel(f"{odor_name[:15]}", fontsize=9)
            if row == n_odors - 1:
                ax.set_xlabel("Time (s)")
    
    split_label = "Train" if split == 'train' else "Test (Interp)"
    fig.suptitle(f"G919 {category.title()} [{split_label}] - Concentration Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_length_distribution(data_dir: Path, output_path: Path, 
                            split: str = 'train', is_test_interp: bool = False):
    """Plot distribution of sample lengths across all split files."""
    categories = ['condiment', 'drink', 'fruit', 'milk', 'perfume', 'spice', 'vegetable', 'wine']
    
    lengths = {'low': [], 'mid': [], 'high': []}
    
    for category in categories:
        if split == 'train':
            sub_dir = data_dir / category / f"{category}s_train"
            if not sub_dir.exists():
                sub_dir = data_dir / category / f"{category}_train"
        else:
            sub_dir = data_dir / category / f"{category}s_test"
            if not sub_dir.exists():
                sub_dir = data_dir / category / f"{category}_test"
        
        if not sub_dir.exists():
            continue
        
        for conc in ['low', 'mid', 'high']:
            if is_test_interp:
                pattern = f"*_interp_{conc}.csv"
            else:
                pattern = f"*_{conc}.csv"
            for csv_path in sub_dir.glob(pattern):
                df = pd.read_csv(csv_path)
                lengths[conc].append(len(df))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = {'low': 'blue', 'mid': 'green', 'high': 'red'}
    
    for ax, (conc, lens) in zip(axes, lengths.items()):
        if lens:
            ax.hist(lens, bins=30, color=colors[conc], alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(lens), color='black', linestyle='--', 
                      label=f'Mean: {np.mean(lens):.0f}')
            ax.axvline(np.median(lens), color='gray', linestyle=':',
                      label=f'Median: {np.median(lens):.0f}')
            ax.set_title(f"{conc.upper()} Concentration\n(n={len(lens)}, std={np.std(lens):.0f})")
            ax.set_xlabel("Sample Length (rows)")
            ax.set_ylabel("Count")
            ax.legend()
    
    split_label = "Train" if split == 'train' else "Test (Interp)"
    fig.suptitle(f"G919 [{split_label}] Split Sample Length Distribution", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_train_vs_test_comparison(data_dir: Path, output_path: Path):
    """Plot comparison between train and test (interpolated) samples for same odor."""
    # Find common odors between train and test
    category = 'condiment'
    train_dir = data_dir / category / f"{category}s_train"
    test_dir = data_dir / category / f"{category}s_test"
    
    if not train_dir.exists() or not test_dir.exists():
        return
    
    # Use Chili oil as example (train files have _1/_2 suffix)
    odor_name_train = "Chili oil_1"  # Use first repetition
    odor_name_test = "Chili oil"
    train_data = load_split_data(train_dir, odor_name_train, is_test_interp=False)
    test_data = load_split_data(test_dir, odor_name_test, is_test_interp=True)
    
    if not train_data or not test_data:
        return
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    conc_keys = ['low', 'mid', 'high']
    
    for row, conc in enumerate(conc_keys):
        # Train
        ax_train = axes[row, 0]
        if conc in train_data:
            df = train_data[conc]
            time = df.iloc[:, 0].values
            mean_signal = np.mean([df[f"sen.{i}"].values for i in range(1, 9)], axis=0)
            ax_train.plot(time, mean_signal, 'b-', linewidth=1, label='8-ch mean')
            for i in range(1, 9):
                ax_train.plot(time, df[f"sen.{i}"].values, alpha=0.3, linewidth=0.5)
        ax_train.set_title(f"Train - {conc.upper()} (len={len(train_data.get(conc, []))})")
        ax_train.set_ylabel("Response")
        if row == 2:
            ax_train.set_xlabel("Time (s)")
        
        # Test
        ax_test = axes[row, 1]
        if conc in test_data:
            df = test_data[conc]
            time = df.iloc[:, 0].values
            mean_signal = np.mean([df[f"sen.{i}"].values for i in range(1, 9)], axis=0)
            ax_test.plot(time, mean_signal, 'r-', linewidth=1, label='8-ch mean')
            for i in range(1, 9):
                ax_test.plot(time, df[f"sen.{i}"].values, alpha=0.3, linewidth=0.5)
        ax_test.set_title(f"Test (Interp) - {conc.upper()} (len={len(test_data.get(conc, []))})")
        if row == 2:
            ax_test.set_xlabel("Time (s)")
    
    fig.suptitle(f"G919 Train vs Test Comparison - Chili oil", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    data_dir = Path("/root/UCI_dataset/.cache/g919_55/single")
    output_dir = Path("/root/UCI_dataset/tests/outputs/g919")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== Train set ==========
    # Plot train sample length distribution
    plot_length_distribution(data_dir, output_dir / "length_distribution.png", 
                            split='train', is_test_interp=False)
    
    # Plot train category overviews
    categories = ['condiment', 'fruit', 'milk', 'wine']
    for cat in categories:
        plot_category_overview(data_dir, cat, output_dir / f"{cat}_overview.png",
                              split='train', is_test_interp=False)
    
    # ========== Test set (interpolated) ==========
    # Plot test sample length distribution
    plot_length_distribution(data_dir, output_dir / "test_length_distribution.png",
                            split='test', is_test_interp=True)
    
    # Plot test category overviews
    for cat in categories:
        plot_category_overview(data_dir, cat, output_dir / f"test_{cat}_overview.png",
                              split='test', is_test_interp=True)
    
    # ========== Train vs Test comparison ==========
    plot_train_vs_test_comparison(data_dir, output_dir / "train_vs_test_comparison.png")
    
    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
