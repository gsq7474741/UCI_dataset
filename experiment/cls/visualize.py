"""Visualization utilities for 1D Grad-CAM and classification analysis.

Generates publication-quality figures showing:
1. Original time series with class labels
2. 1D Grad-CAM weight maps
3. Importance maps (qualitative evaluation)
4. Contribution maps (quantitative evaluation)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to prevent blocking
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12


def plot_gradcam_analysis(
    data: torch.Tensor,
    labels: torch.Tensor,
    cam_results: Dict[str, torch.Tensor],
    class_names: List[str],
    purity: Optional[torch.Tensor] = None,
    channel_idx: int = 0,
    num_samples_per_class: int = 1,
    save_path: Optional[str] = None,
    title: str = "1D Grad-CAM Analysis",
) -> plt.Figure:
    """Generate comprehensive Grad-CAM analysis figure.
    
    Args:
        data: [N, C, T] input data
        labels: [N] ground truth labels
        cam_results: Dict from GradCAM1D with weight_map, cam, contribution
        class_names: List of class names
        purity: [T] optional purity scores
        channel_idx: Which channel to visualize
        num_samples_per_class: Number of samples per class to show
        save_path: Path to save figure
        title: Figure title
        
    Returns:
        matplotlib Figure
    """
    num_classes = len(class_names)
    
    # Select representative samples per class
    selected_indices = []
    for c in range(num_classes):
        mask = (labels == c).nonzero(as_tuple=True)[0]
        if len(mask) > 0:
            # Select samples with highest prediction confidence for correct class
            probs = cam_results.get('probs', None)
            if probs is not None:
                class_probs = probs[mask, c]
                top_idx = class_probs.argsort(descending=True)[:num_samples_per_class]
                selected_indices.extend(mask[top_idx].tolist())
            else:
                selected_indices.extend(mask[:num_samples_per_class].tolist())
    
    n_samples = len(selected_indices)
    if n_samples == 0:
        print("No samples to visualize")
        return None
    
    # Create figure with subplots
    # Layout: 4 rows (Original, Weight Map, Importance, Contribution) × n_samples cols
    n_rows = 4 if purity is not None else 3
    fig = plt.figure(figsize=(4 * min(n_samples, 4), 3 * n_rows))
    gs = GridSpec(n_rows, min(n_samples, 4), figure=fig, hspace=0.3, wspace=0.25)
    
    # Color map for classes
    colors = plt.cm.Set1(np.linspace(0, 1, num_classes))
    
    for i, idx in enumerate(selected_indices[:4]):  # Max 4 samples
        x = data[idx, channel_idx].cpu().numpy()  # [T]
        label = labels[idx].item()
        pred = cam_results['pred'][idx].item()
        
        weight_map = cam_results['weight_map'][idx].cpu().numpy()
        cam = cam_results['cam'][idx].cpu().numpy()
        
        T = len(x)
        t = np.arange(T)
        
        # Row 0: Original signal
        ax0 = fig.add_subplot(gs[0, i])
        ax0.plot(t, x, color=colors[label], linewidth=0.8, alpha=0.8)
        ax0.fill_between(t, x.min(), x, alpha=0.3, color=colors[label])
        ax0.set_title(f"Original\nTrue: {class_names[label]}, Pred: {class_names[pred]}")
        ax0.set_ylabel("Intensity")
        if i == 0:
            ax0.legend([f"Class: {class_names[label]}"], loc='upper right', fontsize=8)
        
        # Row 1: Weight Map (1D Grad-CAM)
        ax1 = fig.add_subplot(gs[1, i])
        ax1.plot(t, weight_map, color='darkgreen', linewidth=0.8)
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax1.fill_between(t, 0, weight_map, where=weight_map > 0, 
                        alpha=0.3, color='green', label='Positive')
        ax1.fill_between(t, 0, weight_map, where=weight_map < 0,
                        alpha=0.3, color='red', label='Negative')
        ax1.set_title("1D Grad-CAM")
        ax1.set_ylabel("Weight")
        
        # Row 2: Importance Map (if purity available)
        row_idx = 2
        if purity is not None:
            ax2 = fig.add_subplot(gs[2, i])
            purity_np = purity.cpu().numpy() if isinstance(purity, torch.Tensor) else purity
            # Ensure purity matches length
            if len(purity_np) != T:
                purity_np = np.interp(np.linspace(0, 1, T), 
                                      np.linspace(0, 1, len(purity_np)), purity_np)
            importance = weight_map * purity_np
            ax2.plot(t, importance, color='purple', linewidth=0.8)
            ax2.fill_between(t, 0, importance, alpha=0.3, color='purple')
            ax2.set_title("Importance (Qualitative)")
            ax2.set_ylabel("Importance")
            row_idx = 3
        
        # Row 3 (or 2): Contribution Map
        if 'contribution' in cam_results:
            ax3 = fig.add_subplot(gs[row_idx, i])
            contribution = cam_results['contribution'][idx].cpu().numpy()
            ax3.plot(t, contribution, color='darkorange', linewidth=0.8)
            ax3.fill_between(t, 0, contribution, alpha=0.3, color='orange')
            ax3.set_title("Contribution (Quantitative)")
            ax3.set_ylabel("Contribution")
            ax3.set_xlabel("Time Step")
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    try:
        plt.tight_layout()
    except Exception:
        pass
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_single_sample_gradcam(
    data: torch.Tensor,
    labels: torch.Tensor,
    cam_results: Dict[str, torch.Tensor],
    class_names: List[str],
    sample_indices: List[int],
    num_channels: int = 8,
    accuracy: Optional[float] = None,
    save_path: Optional[str] = None,
    epoch: int = 0,
) -> plt.Figure:
    """Plot Grad-CAM heatmap for individual samples (CV-style visualization).
    
    Args:
        data: [N, C, T] input data
        labels: [N] ground truth labels
        cam_results: Dict from GradCAM1D with 'cam' and optionally 'channel_cam'
        class_names: List of class names
        sample_indices: List of sample indices to visualize
        num_channels: Number of channels to display
        accuracy: Optional accuracy to display in title
        save_path: Path to save figure
        epoch: Current epoch number
        
    Returns:
        matplotlib Figure
    """
    from scipy.ndimage import gaussian_filter1d
    
    n_samples = len(sample_indices)
    actual_channels = min(num_channels, data.shape[1])
    
    # Create figure: rows = samples, cols = channels
    fig, axes = plt.subplots(n_samples, actual_channels, 
                             figsize=(2.5 * actual_channels, 2 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    if actual_channels == 1:
        axes = axes.reshape(-1, 1)
    
    T = data.shape[-1]
    has_channel_cam = 'channel_cam' in cam_results
    
    for row, sample_idx in enumerate(sample_indices):
        if sample_idx >= len(data):
            continue
            
        sample_data = data[sample_idx].cpu().numpy()  # [C, T]
        sample_label = labels[sample_idx].item()
        sample_pred = cam_results['pred'][sample_idx].item()
        
        # Get CAM for this sample
        if has_channel_cam:
            sample_cam = cam_results['channel_cam'][sample_idx].cpu().numpy()  # [C, T_cam]
        else:
            sample_cam = cam_results['cam'][sample_idx].cpu().numpy()  # [T_cam]
            sample_cam = np.tile(sample_cam, (actual_channels, 1))  # [C, T_cam]
        
        is_correct = sample_label == sample_pred
        
        for ch in range(actual_channels):
            ax = axes[row, ch]
            
            signal = sample_data[ch]  # [T]
            cam = sample_cam[ch] if sample_cam.ndim > 1 else sample_cam  # [T_cam]
            
            # Interpolate CAM to signal length if needed
            if len(cam) != T:
                cam = np.interp(np.arange(T), np.linspace(0, T, len(cam)), cam)
            
            # Smooth CAM
            cam_smooth = gaussian_filter1d(cam, sigma=T // 20)
            cam_norm = (cam_smooth - cam_smooth.min()) / (cam_smooth.max() - cam_smooth.min() + 1e-8)
            
            # Plot signal
            t = np.arange(T)
            ax.plot(t, signal, 'k-', linewidth=0.8, alpha=0.9)
            
            # Overlay CAM as colored background
            y_min, y_max = signal.min(), signal.max()
            y_range = y_max - y_min
            y_min -= y_range * 0.1
            y_max += y_range * 0.1
            
            cam_2d = cam_norm.reshape(1, -1)
            ax.imshow(cam_2d, aspect='auto', extent=[0, T, y_min, y_max],
                     cmap='Reds', alpha=0.4, origin='lower', vmin=0, vmax=1)
            
            ax.set_xlim(0, T)
            ax.set_ylim(y_min, y_max)
            
            # Labels
            if row == 0:
                ax.set_title(f'Ch{ch}', fontsize=9)
            if ch == 0:
                color = 'green' if is_correct else 'red'
                true_name = class_names[sample_label] if sample_label < len(class_names) else f'Class{sample_label}'
                pred_name = class_names[sample_pred] if sample_pred < len(class_names) else f'Class{sample_pred}'
                # Truncate names
                true_name = true_name[:12] + '..' if len(true_name) > 14 else true_name
                pred_name = pred_name[:12] + '..' if len(pred_name) > 14 else pred_name
                ax.set_ylabel(f'{true_name}\n→{pred_name}', fontsize=7, color=color)
            
            ax.tick_params(labelsize=6)
            if row < n_samples - 1:
                ax.set_xticklabels([])
    
    title = f'Single Sample Grad-CAM (Epoch {epoch})'
    if accuracy is not None:
        title += f' | Acc: {accuracy:.1%}'
    fig.suptitle(title, fontsize=11)
    
    try:
        plt.tight_layout()
    except:
        pass
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def select_fixed_samples(
    labels: torch.Tensor,
    num_classes: int = 12,
    seed: int = 42,
) -> List[int]:
    """Select one sample per class for consistent visualization across epochs.
    
    Args:
        labels: [N] ground truth labels
        num_classes: Number of classes to select
        seed: Random seed for reproducibility
        
    Returns:
        List of sample indices (one per class)
    """
    np.random.seed(seed)
    selected = []
    
    for c in range(min(num_classes, labels.max().item() + 1)):
        mask = (labels == c).numpy()
        indices = np.where(mask)[0]
        if len(indices) > 0:
            # Always pick the first one (deterministic)
            selected.append(int(indices[0]))
    
    return selected


def plot_class_comparison(
    data: torch.Tensor,
    labels: torch.Tensor,
    cam_results: Dict[str, torch.Tensor],
    class_names: List[str],
    num_channels: int = 8,
    accuracy: Optional[float] = None,
    save_path: Optional[str] = None,
    frequency_domain: bool = False,
    max_classes: int = 10,  # Limit classes for many-class datasets
) -> plt.Figure:
    """Plot class-averaged signals for ALL channels with important regions marked.
    
    Shows average response per class per channel with highlighted discriminative regions.
    
    Args:
        data: [N, C, T] input data (or [N, C, F] if frequency_domain)
        labels: [N] ground truth labels
        cam_results: Dict from GradCAM1D
        class_names: List of class names
        num_channels: Number of channels to display
        accuracy: Classification accuracy to display in title
        save_path: Path to save figure
        frequency_domain: If True, x-axis is frequency bins
        max_classes: Maximum number of classes to display (for many-class datasets)
        
    Returns:
        matplotlib Figure
    """
    total_classes = len(class_names)
    # Limit classes for visualization
    num_classes = min(total_classes, max_classes)
    actual_channels = min(num_channels, data.shape[1])
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))  # tab10 better for many classes
    
    # Create figure: rows = channels, cols = 2 (signal + CAM)
    # Do NOT share x-axis since CAM has different length than signal
    fig, axes = plt.subplots(actual_channels, 2, figsize=(16, 2.5 * actual_channels), sharex=False)
    
    T = data.shape[-1]
    t = np.arange(T)
    
    # Store important regions for each class/channel
    important_regions = {}  # {(class_idx, channel_idx): [(start, end, peak_pos), ...]}
    
    # Get per-channel CAM if available
    has_channel_cam = 'channel_cam' in cam_results
    
    for ch in range(actual_channels):
        ax_signal = axes[ch, 0]
        ax_cam = axes[ch, 1]
        
        # Compute class averages for this channel
        class_means = []
        class_channel_cams = []  # Per-channel CAM
        
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                class_data = data[mask, ch].cpu().numpy()
                class_mean = class_data.mean(axis=0)
                class_std = class_data.std(axis=0)
                class_means.append((class_mean, class_std))
                
                # Get per-channel CAM for this class
                if has_channel_cam:
                    ch_cam = cam_results['channel_cam'][mask, ch].cpu().numpy()
                    class_cam = ch_cam.mean(axis=0)
                else:
                    # Fallback: use global CAM
                    class_cam = cam_results['cam'][mask].cpu().numpy().mean(axis=0)
                class_channel_cams.append(class_cam)
            else:
                class_means.append((np.zeros(T), np.zeros(T)))
                class_channel_cams.append(np.zeros(T if not has_channel_cam else cam_results['channel_cam'].shape[-1]))
        
        # Get CAM time dimension
        T_cam = len(class_channel_cams[0]) if class_channel_cams else T
        t_cam = np.arange(T_cam)
        
        # Compute average CAM across all classes for this channel (for heatmap overlay)
        all_cams = np.array(class_channel_cams)  # [num_classes, T_cam]
        avg_cam = all_cams.mean(axis=0)  # Average attention across classes
        # Interpolate CAM to match signal length
        if len(avg_cam) != T:
            avg_cam_interp = np.interp(np.arange(T), np.linspace(0, T, len(avg_cam)), avg_cam)
        else:
            avg_cam_interp = avg_cam
        # Smooth CAM to remove high-frequency noise (Gaussian filter)
        from scipy.ndimage import gaussian_filter1d
        avg_cam_smooth = gaussian_filter1d(avg_cam_interp, sigma=T // 20)  # ~5% of signal length
        # Normalize to [0, 1] for colormap
        avg_cam_norm = (avg_cam_smooth - avg_cam_smooth.min()) / (avg_cam_smooth.max() - avg_cam_smooth.min() + 1e-8)
        
        # Plot CAM heatmap as background on signal plot (like CV Grad-CAM)
        # Use imshow with extent to overlay heatmap
        y_min, y_max = None, None
        for c in range(num_classes):
            mean, _ = class_means[c]
            if y_min is None:
                y_min, y_max = mean.min(), mean.max()
            else:
                y_min = min(y_min, mean.min())
                y_max = max(y_max, mean.max())
        y_range = y_max - y_min
        y_min -= y_range * 0.1
        y_max += y_range * 0.1
        
        # Draw heatmap as colored background
        cam_2d = avg_cam_norm.reshape(1, -1)  # [1, T] for imshow
        ax_signal.imshow(cam_2d, aspect='auto', extent=[0, T, y_min, y_max],
                        cmap='Reds', alpha=0.3, origin='lower', vmin=0, vmax=1)
        
        # Plot signals on top of heatmap
        for c in range(num_classes):
            mean, std = class_means[c]
            ax_signal.plot(t, mean, color=colors[c], linewidth=1.0, label=class_names[c])
            ax_signal.fill_between(t, mean - std, mean + std, color=colors[c], alpha=0.15)
        
        ax_signal.set_ylim(y_min, y_max)
        ax_signal.set_ylabel(f'Ch{ch}', fontsize=10, fontweight='bold')
        if ch == 0:
            ax_signal.set_title('Original Signal + CAM Heatmap (Mean ± Std)', fontsize=11)
            # Skip legend for many classes to avoid overflow
            if num_classes <= 10:
                ax_signal.legend(loc='upper right', ncol=min(5, num_classes), fontsize=7)
            elif num_classes <= 20:
                ax_signal.legend(loc='upper right', ncol=5, fontsize=6, handlelength=1)
        
        # Plot per-channel CAM (use CAM index, not time step)
        for c in range(num_classes):
            cam = class_channel_cams[c]
            ax_cam.plot(t_cam, cam, color=colors[c], linewidth=1.0, label=class_names[c])
            
            # Find important regions and store for summary (no colored spans on signal)
            threshold = np.percentile(cam, 70)
            regions = _find_high_regions(cam, threshold)
            merged_regions = _merge_regions(regions, min_gap=T_cam//20)
            
            for start, end in merged_regions:
                if end - start > T_cam * 0.05:
                    scale = T / T_cam
                    orig_start, orig_end = int(start * scale), int(end * scale)
                    
                    key = (c, ch)
                    if key not in important_regions:
                        important_regions[key] = []
                    important_regions[key].append((orig_start, orig_end))
                    
                    # Mark region on CAM plot only (with annotation)
                    ax_cam.axvspan(start, end, alpha=0.2, color=colors[c])
                    if len(important_regions[key]) <= 2:
                        mid = (start + end) // 2
                        ax_cam.annotate(f'[{orig_start}-{orig_end}]', xy=(mid, cam[mid] if mid < len(cam) else 0),
                                       xytext=(0, 10), textcoords='offset points',
                                       fontsize=6, color=colors[c], alpha=0.9, ha='center')
        
        if ch == 0:
            title = 'Per-Channel CAM' if has_channel_cam else 'Global CAM'
            ax_cam.set_title(title, fontsize=11)
            # Skip legend for many classes
            if num_classes <= 10:
                ax_cam.legend(loc='upper right', ncol=min(5, num_classes), fontsize=7)
            elif num_classes <= 20:
                ax_cam.legend(loc='upper right', ncol=5, fontsize=6, handlelength=1)
        
        # Yellow regions removed - now using CAM heatmap overlay instead
    
    # Set x-axis labels based on domain
    if frequency_domain:
        axes[-1, 0].set_xlabel('Frequency Bin', fontsize=10)
        axes[-1, 1].set_xlabel('CAM Index', fontsize=10)
    else:
        axes[-1, 0].set_xlabel('Time Step', fontsize=10)
        axes[-1, 1].set_xlabel('CAM Index', fontsize=10)
    
    # Build title with accuracy
    domain_str = 'Frequency Domain' if frequency_domain else 'Time Domain'
    class_info = f'{num_classes} Classes' if num_classes == total_classes else f'{num_classes}/{total_classes} Classes'
    title = f'Multi-Channel {domain_str} & Grad-CAM ({class_info})'
    if accuracy is not None:
        title += f' | Accuracy: {accuracy*100:.2f}%'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    try:
        plt.tight_layout()
    except Exception:
        pass  # Ignore tight_layout warnings for complex layouts
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    # Important regions summary - disabled (too verbose for many classes)
    # print("\n=== Important Regions per Class per Channel ===")
    # for c in range(num_classes):
    #     print(f"\n{class_names[c]}:")
    #     for ch in range(actual_channels):
    #         key = (c, ch)
    #         if key in important_regions and important_regions[key]:
    #             regions_str = ", ".join([f"[{r[0]}-{r[1]}]" for r in important_regions[key][:3]])
    #             print(f"  Ch{ch}: {regions_str}")
    
    return fig


def plot_multichannel_comparison(
    data: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    num_channels: int = 8,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot all channels for each class to show sensor response patterns.
    
    Args:
        data: [N, C, T] input data
        labels: [N] ground truth labels
        class_names: List of class names
        num_channels: Number of channels to plot
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    num_classes = len(class_names)
    actual_channels = min(num_channels, data.shape[1])
    
    # Create figure: rows = classes, cols = channels
    fig, axes = plt.subplots(
        num_classes, actual_channels, 
        figsize=(2.5 * actual_channels, 2 * num_classes),
        sharex=True
    )
    
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    
    colors = plt.cm.Set1(np.linspace(0, 1, num_classes))
    
    T = data.shape[-1]
    t = np.arange(T)
    
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        
        class_data = data[mask].cpu().numpy()  # [N_c, C, T]
        
        for ch in range(actual_channels):
            ax = axes[c, ch]
            
            # Plot mean and std
            ch_data = class_data[:, ch, :]  # [N_c, T]
            mean = ch_data.mean(axis=0)
            std = ch_data.std(axis=0)
            
            ax.plot(t, mean, color=colors[c], linewidth=0.8)
            ax.fill_between(t, mean - std, mean + std, color=colors[c], alpha=0.2)
            
            # Labels
            if c == 0:
                ax.set_title(f'Ch{ch}', fontsize=9)
            if ch == 0:
                ax.set_ylabel(class_names[c], fontsize=9)
            
            ax.tick_params(labelsize=7)
            
            # Only show x-axis label on bottom row
            if c == num_classes - 1:
                ax.set_xlabel('Time', fontsize=8)
    
    fig.suptitle('Multi-Channel Signal Comparison (Mean ± Std per Class)', fontsize=12)
    try:
        plt.tight_layout()
    except Exception:
        pass
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_channel_importance(
    data: torch.Tensor,
    labels: torch.Tensor,
    cam_results: Dict[str, torch.Tensor],
    class_names: List[str],
    num_channels: int = 8,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot channel-wise importance based on signal variance and CAM.
    
    Args:
        data: [N, C, T] input data
        labels: [N] ground truth labels
        cam_results: Dict from GradCAM1D
        class_names: List of class names
        num_channels: Number of channels
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    num_classes = len(class_names)
    actual_channels = min(num_channels, data.shape[1])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = plt.cm.Set1(np.linspace(0, 1, num_classes))
    
    # Plot 1: Channel-wise signal variance per class
    ax1 = axes[0]
    x = np.arange(actual_channels)
    width = 0.8 / num_classes
    
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        
        class_data = data[mask].cpu().numpy()  # [N_c, C, T]
        # Compute variance across time for each channel
        channel_vars = class_data.var(axis=2).mean(axis=0)[:actual_channels]
        
        ax1.bar(x + c * width, channel_vars, width, 
                label=class_names[c], color=colors[c], alpha=0.8)
    
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Signal Variance')
    ax1.set_title('Channel-wise Signal Variance per Class')
    ax1.set_xticks(x + width * (num_classes - 1) / 2)
    ax1.set_xticklabels([f'Ch{i}' for i in range(actual_channels)])
    ax1.legend(loc='upper right')
    
    # Plot 2: Channel contribution (from signal range)
    ax2 = axes[1]
    
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        
        class_data = data[mask].cpu().numpy()  # [N_c, C, T]
        # Compute range (max - min) for each channel
        channel_ranges = (class_data.max(axis=2) - class_data.min(axis=2)).mean(axis=0)[:actual_channels]
        
        ax2.bar(x + c * width, channel_ranges, width,
                label=class_names[c], color=colors[c], alpha=0.8)
    
    ax2.set_xlabel('Channel')
    ax2.set_ylabel('Signal Range (max - min)')
    ax2.set_title('Channel-wise Signal Range per Class')
    ax2.set_xticks(x + width * (num_classes - 1) / 2)
    ax2.set_xticklabels([f'Ch{i}' for i in range(actual_channels)])
    ax2.legend(loc='upper right')
    
    try:
        plt.tight_layout()
    except Exception:
        pass
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def _find_high_regions(values: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
    """Find contiguous regions above threshold."""
    above = values > threshold
    regions = []
    start = None
    
    for i, val in enumerate(above):
        if val and start is None:
            start = i
        elif not val and start is not None:
            regions.append((start, i))
            start = None
    
    if start is not None:
        regions.append((start, len(values)))
    
    return regions


def _merge_regions(regions: List[Tuple[int, int]], min_gap: int = 100) -> List[Tuple[int, int]]:
    """Merge nearby regions that are within min_gap of each other."""
    if not regions:
        return []
    
    # Sort by start position
    sorted_regions = sorted(regions, key=lambda x: x[0])
    
    merged = [sorted_regions[0]]
    for start, end in sorted_regions[1:]:
        prev_start, prev_end = merged[-1]
        
        # If this region is close to the previous one, merge them
        if start - prev_end <= min_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    
    return merged


def plot_confusion_with_cam(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    class_cams: Dict[int, np.ndarray],
    save_path: Optional[str] = None,
    max_classes: int = 60,  # Support up to 60 classes
) -> plt.Figure:
    """Plot confusion matrix with representative CAM for each class.
    
    Args:
        confusion_matrix: [num_classes, num_classes] confusion matrix
        class_names: List of class names
        class_cams: Dict mapping class index to average CAM
        save_path: Path to save figure
        max_classes: Maximum classes to show in confusion matrix
        
    Returns:
        matplotlib Figure
    """
    total_classes = len(class_names)
    num_classes = min(total_classes, max_classes)
    
    # Truncate confusion matrix if too many classes
    cm_display = confusion_matrix[:num_classes, :num_classes]
    names_display = class_names[:num_classes]
    
    # Adjust figure size for many classes
    if num_classes > 40:
        fig_size = (18, 16)  # Larger for 40+ classes
    elif num_classes > 20:
        fig_size = (14, 12)
    else:
        fig_size = (max(10, num_classes * 0.5), max(8, num_classes * 0.4))
    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 0.25])
    
    # Confusion matrix
    ax_cm = fig.add_subplot(gs[0, :])
    im = ax_cm.imshow(cm_display, cmap='Blues')
    
    ax_cm.set_xticks(range(num_classes))
    ax_cm.set_yticks(range(num_classes))
    # Smaller font for many classes
    if num_classes > 40:
        fontsize = 5
        rotation = 90
    elif num_classes > 20:
        fontsize = 6
        rotation = 60
    else:
        fontsize = max(6, 10 - num_classes // 5)
        rotation = 45
    ax_cm.set_xticklabels(names_display, rotation=rotation, ha='right', fontsize=fontsize)
    ax_cm.set_yticklabels(names_display, fontsize=fontsize)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('True')
    title = f'Confusion Matrix ({num_classes}/{total_classes} classes)' if num_classes < total_classes else 'Confusion Matrix'
    ax_cm.set_title(title)
    
    # Add text annotations (skip for many classes)
    if num_classes <= 10:
        for i in range(num_classes):
            for j in range(num_classes):
                ax_cm.text(j, i, f'{cm_display[i, j]:.0f}',
                          ha='center', va='center',
                          color='white' if cm_display[i, j] > cm_display.max()/2 else 'black')
    
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    
    # CAM examples for top 2 classes
    colors = plt.cm.Set1(np.linspace(0, 1, num_classes))
    
    for plot_idx, c in enumerate(list(class_cams.keys())[:2]):
        ax = fig.add_subplot(gs[1, plot_idx])
        cam = class_cams[c]
        t = np.arange(len(cam))
        ax.fill_between(t, 0, cam, alpha=0.5, color=colors[c])
        ax.plot(t, cam, color=colors[c], linewidth=1)
        ax.set_title(f"Avg CAM: {class_names[c]}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Weight")
    
    try:
        plt.tight_layout()
    except Exception:
        pass
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def generate_all_visualizations(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: List[str],
    output_dir: str,
    device: str = 'cuda',
    max_samples: int = 100,
    epoch: int = 0,
    frequency_domain: bool = False,
) -> Dict[str, str]:
    """Generate all visualization figures.
    
    Args:
        model: Trained model
        dataloader: Test dataloader
        class_names: List of class names
        output_dir: Directory to save figures
        device: Device to use
        max_samples: Maximum samples to process
        epoch: Current epoch number
        frequency_domain: If True, data is in frequency domain
        
    Returns:
        Dict mapping figure names to file paths
    """
    from .grad_cam import GradCAM1D, compute_purity
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    model.to(device)
    
    # Collect data and compute Grad-CAM
    all_data = []
    all_labels = []
    all_results = []
    
    grad_cam = GradCAM1D(model)
    
    # Check if model supports channel-wise CAM
    has_channel_cam = hasattr(model, 'get_channel_cam') and hasattr(model, 'channel_wise')
    
    n_samples = 0
    for batch in dataloader:
        if n_samples >= max_samples:
            break
        
        data = batch['data'].to(device)
        labels = batch['label'].to(device)
        
        with torch.enable_grad():
            results = grad_cam.compute_contribution(data)
            
            # Compute per-channel CAM if available
            if has_channel_cam:
                try:
                    channel_cam, channel_weights, _ = model.get_channel_cam(data.clone())
                    results['channel_cam'] = channel_cam
                    results['channel_weights'] = channel_weights
                except Exception as e:
                    print(f"[Warning] Failed to compute channel CAM: {e}")
        
        all_data.append(data.cpu())
        all_labels.append(labels.cpu())
        all_results.append({k: v.cpu() if isinstance(v, torch.Tensor) else v 
                          for k, v in results.items()})
        
        n_samples += len(data)
    
    # Concatenate results - handle variable length by padding
    # Find max length across all batches
    max_len = max(d.shape[-1] for d in all_data)
    num_channels = all_data[0].shape[1]
    
    # Pad data to max length
    padded_data = []
    for d in all_data:
        if d.shape[-1] < max_len:
            pad_size = max_len - d.shape[-1]
            d = F.pad(d, (0, pad_size), mode='constant', value=0)
        padded_data.append(d)
    
    data = torch.cat(padded_data, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    # Combine results - also pad CAM/weight maps
    combined_results = {}
    for key in all_results[0].keys():
        if isinstance(all_results[0][key], torch.Tensor):
            tensors = [r[key] for r in all_results]
            # Check if this is a time-dimension tensor that needs padding (2D: [B, T])
            if tensors[0].dim() == 2:
                # Pad to max_len
                padded_tensors = []
                for t in tensors:
                    if t.shape[-1] < max_len:
                        pad_size = max_len - t.shape[-1]
                        t = F.pad(t, (0, pad_size), mode='constant', value=0)
                    elif t.shape[-1] > max_len:
                        t = t[..., :max_len]
                    padded_tensors.append(t)
                combined_results[key] = torch.cat(padded_tensors, dim=0)
            else:
                # 1D tensors (pred, target, etc.) - just concatenate
                combined_results[key] = torch.cat(tensors, dim=0)
        else:
            combined_results[key] = all_results[0][key]
    
    # Compute purity
    purity = compute_purity(
        data[:, 0, :],  # First channel
        labels,
        len(class_names)
    )
    
    saved_files = {}
    
    # # Figure 1: Grad-CAM Analysis
    # fig1_path = output_dir / f"gradcam_analysis_epoch{epoch:03d}.png"
    # plot_gradcam_analysis(
    #     data, labels, combined_results, class_names,
    #     purity=purity,
    #     save_path=str(fig1_path),
    #     title=f"1D Grad-CAM Analysis (Epoch {epoch})"
    # )
    # saved_files['gradcam_analysis'] = str(fig1_path)
    # plt.close()
    
    # Compute accuracy for title
    preds = combined_results['pred']
    accuracy = (preds == labels).float().mean().item()
    
    # Figure 2: Class Comparison (All 8 channels with important regions)
    fig2_path = output_dir / f"class_comparison_epoch{epoch:03d}.png"
    plot_class_comparison(
        data, labels, combined_results, class_names,
        num_channels=8,
        accuracy=accuracy,
        save_path=str(fig2_path),
        frequency_domain=frequency_domain,
    )
    saved_files['class_comparison'] = str(fig2_path)
    plt.close()
    
    # Figure 3: Single Sample Grad-CAM (CV-style, first 12 classes)
    sample_indices = select_fixed_samples(labels, num_classes=12, seed=42)
    if len(sample_indices) > 0:
        fig3_path = output_dir / f"single_sample_cam_epoch{epoch:03d}.png"
        plot_single_sample_gradcam(
            data, labels, combined_results, class_names,
            sample_indices=sample_indices,
            num_channels=8,
            accuracy=accuracy,
            save_path=str(fig3_path),
            epoch=epoch,
        )
        saved_files['single_sample_cam'] = str(fig3_path)
        plt.close()
    
    # # Figure 4: Per-class CAM averages
    # num_classes = len(class_names)
    # class_cams = {}
    # for c in range(num_classes):
    #     mask = labels == c
    #     if mask.sum() > 0:
    #         class_cams[c] = combined_results['cam'][mask].mean(dim=0).numpy()
    
    # # Compute confusion matrix
    # conf_matrix = np.zeros((num_classes, num_classes))
    # for true, pred in zip(labels.numpy(), preds.numpy()):
    #     conf_matrix[true, pred] += 1
    
    # fig3_path = output_dir / f"confusion_cam_epoch{epoch:03d}.png"
    # plot_confusion_with_cam(
    #     conf_matrix, class_names, class_cams,
    #     save_path=str(fig3_path)
    # )
    # saved_files['confusion_cam'] = str(fig3_path)
    # plt.close()
    
    # Figure 4: Multi-channel comparison - disabled for many classes (slow)
    # if num_classes <= 10:
    #     fig4_path = output_dir / f"multichannel_epoch{epoch:03d}.png"
    #     plot_multichannel_comparison(
    #         data, labels, class_names,
    #         num_channels=8,
    #         save_path=str(fig4_path)
    #     )
    #     saved_files['multichannel'] = str(fig4_path)
    #     plt.close()
    
    # Figure 5: Channel importance analysis - disabled (slow)
    # fig5_path = output_dir / f"channel_importance_epoch{epoch:03d}.png"
    # plot_channel_importance(
    #     data, labels, combined_results, class_names,
    #     num_channels=8,
    #     save_path=str(fig5_path)
    # )
    # saved_files['channel_importance'] = str(fig5_path)
    # plt.close()
    
    return saved_files
