#!/usr/bin/env python3
"""Split G919 training CSV files into 3 concentration segments (low, medium, high).

Each training CSV contains 3 consecutive concentration tests.
This script splits them into separate files.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


# Prior knowledge from previous split statistics
EXPECTED_SEGMENT_LENGTH = 2350  # Mean of low/mid segments
SEGMENT_LENGTH_MIN = 1800
SEGMENT_LENGTH_MAX = 3000


def find_rise_start(signal: np.ndarray, start_idx: int, window: int = 50) -> int:
    """Find where response starts rising using adaptive difference method.
    
    Based on paper: start from baseline, find point where difference 
    exceeds previous difference (indicating rising edge start).
    """
    if start_idx >= len(signal) - window:
        return start_idx
    
    # Look for rising start in a window before the peak
    # Go backwards from peak to find where rise begins
    baseline = np.mean(signal[max(0, start_idx-200):max(1, start_idx-100)])
    
    # Find where signal first exceeds baseline + threshold
    threshold = (signal[start_idx] - baseline) * 0.05  # 5% of rise
    
    for i in range(start_idx, max(0, start_idx - 500), -1):
        if signal[i] < baseline + threshold:
            return i
    
    return max(0, start_idx - 200)


def find_response_end(signal: np.ndarray, start_idx: int = 0) -> int:
    """Find where response ends (returns to baseline after falling edge).
    
    Target length ~2350 to match low/mid segments.
    Uses derivative to detect end of falling edge.
    """
    from scipy.ndimage import uniform_filter1d
    
    if len(signal) < 500:
        return len(signal)
    
    # Smooth the signal
    smoothed = uniform_filter1d(signal, size=50)
    
    # Compute derivative
    derivative = np.concatenate([[0], np.diff(smoothed)])
    derivative_smooth = uniform_filter1d(derivative, size=50)
    
    # Find where derivative becomes ~0 after the falling phase (recovery complete)
    # Start searching from expected position, target ~2350 like low/mid
    search_start = min(len(signal) - 100, 2000)
    search_end = min(len(signal) - 50, 2800)
    
    # Look for where derivative stabilizes near zero
    for i in range(search_start, search_end):
        # Check if derivative is stable (near zero) for a window
        window = derivative_smooth[i:i+50]
        if np.abs(np.mean(window)) < 0.0003 and np.std(window) < 0.0008:
            return min(len(signal), i + 150)
    
    # If not found, use expected length to match low/mid
    return min(len(signal), EXPECTED_SEGMENT_LENGTH + 50)


def find_split_points(df: pd.DataFrame, n_segments: int = 3) -> list:
    """Find split points at the START of rising edges (not peak).
    
    1. Find derivative peaks (rising edge locations)
    2. For each peak, backtrack to find where rise actually begins
    3. Use prior knowledge to select best split points
    """
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks
    
    sensor_cols = [c for c in df.columns if c.startswith("sen.")]
    if len(sensor_cols) < 8:
        segment_len = len(df) // n_segments
        return [segment_len * i for i in range(1, n_segments)]
    
    # Compute 8-channel mean
    mean_signal = np.mean(df[sensor_cols].values, axis=1)
    smoothed = uniform_filter1d(mean_signal, size=30)
    
    # Compute derivative
    derivative = np.concatenate([[0], np.diff(smoothed)])
    derivative_smooth = uniform_filter1d(derivative, size=30)
    
    # Find derivative peaks (these are middle of rising edges)
    pos_deriv = derivative_smooth[derivative_smooth > 0]
    if len(pos_deriv) == 0:
        segment_len = len(df) // n_segments
        return [segment_len * i for i in range(1, n_segments)]
    
    threshold = np.percentile(pos_deriv, 30)
    peaks, _ = find_peaks(derivative_smooth, height=threshold, distance=300)
    
    if len(peaks) < n_segments:
        segment_len = len(df) // n_segments
        return [segment_len * i for i in range(1, n_segments)]
    
    # For each peak, find the START of the rise (backtrack)
    rise_starts = []
    for peak in peaks:
        rise_start = find_rise_start(smoothed, peak)
        rise_starts.append(rise_start)
    
    # Greedy selection using prior length constraints
    split_points = []
    current_pos = 0
    
    for _ in range(n_segments - 1):
        # Find rise_starts in valid range
        valid_starts = [s for s in rise_starts 
                       if SEGMENT_LENGTH_MIN <= (s - current_pos) <= SEGMENT_LENGTH_MAX]
        
        if valid_starts:
            best_start = min(valid_starts,
                           key=lambda s: abs(s - current_pos - EXPECTED_SEGMENT_LENGTH))
            split_points.append(best_start)
            current_pos = best_start
        else:
            # Find nearest rise_start to expected position
            expected_pos = current_pos + EXPECTED_SEGMENT_LENGTH
            candidates = [s for s in rise_starts if s > current_pos + 500]
            if candidates:
                nearest = min(candidates, key=lambda s: abs(s - expected_pos))
                split_points.append(nearest)
                current_pos = nearest
            else:
                split_points.append(current_pos + EXPECTED_SEGMENT_LENGTH)
                current_pos += EXPECTED_SEGMENT_LENGTH
    
    return split_points


def split_csv(csv_path: Path, output_dir: Path = None, dry_run: bool = False):
    """Split a single CSV file into 3 concentration segments."""
    df = pd.read_csv(csv_path)
    
    # Get sensor columns for signal analysis
    sensor_cols = [c for c in df.columns if c.startswith("sen.")]
    
    # Get split points
    split_indices = find_split_points(df, n_segments=3)
    
    # Create segments
    segments = []
    prev_idx = 0
    for split_idx in split_indices:
        segments.append(df.iloc[prev_idx:split_idx].copy())
        prev_idx = split_idx
    
    # For HIGH segment, truncate the tail to match low/mid length
    high_segment = df.iloc[prev_idx:].copy()
    if len(high_segment) > SEGMENT_LENGTH_MAX and len(sensor_cols) >= 8:
        # Use 8-channel mean to find response end
        mean_signal = np.mean(high_segment[sensor_cols].values, axis=1)
        end_idx = find_response_end(mean_signal, start_idx=0)
        high_segment = high_segment.iloc[:end_idx].copy()
    segments.append(high_segment)
    
    # Reset time for each segment to start from 0
    for seg in segments:
        time_col = seg.columns[0]
        seg[time_col] = seg[time_col] - seg[time_col].iloc[0]
    
    # Output
    if output_dir is None:
        output_dir = csv_path.parent
    
    stem = csv_path.stem
    # Keep the original stem (including _1, _2 suffix for repeated experiments)
    
    concentration_names = ['low', 'mid', 'high']
    output_paths = []
    
    for i, (seg, conc) in enumerate(zip(segments, concentration_names)):
        out_name = f"{stem}_{conc}.csv"
        out_path = output_dir / out_name
        output_paths.append(out_path)
        
        if not dry_run:
            seg.to_csv(out_path, index=False)
            print(f"  Saved: {out_path.name} ({len(seg)} rows)")
        else:
            print(f"  [DRY RUN] Would save: {out_path.name} ({len(seg)} rows)")
    
    return output_paths


def process_directory(data_dir: Path, dry_run: bool = False):
    """Process all training CSV files in the G919 dataset directory."""
    categories = ['condiment', 'drink', 'fruit', 'milk', 'perfume', 'spice', 'vegetable', 'wine']
    
    total_processed = 0
    for category in categories:
        train_dir = data_dir / category / f"{category}s_train"
        if not train_dir.exists():
            # Try alternative naming
            train_dir = data_dir / category / f"{category}_train"
        if not train_dir.exists():
            continue
        
        print(f"\nProcessing {category}...")
        
        for csv_path in sorted(train_dir.glob("*.csv")):
            # Skip already split files
            if any(csv_path.stem.endswith(f"_{c}") for c in ['low', 'mid', 'high']):
                continue
            # Skip processed files
            if csv_path.name.startswith("processed_"):
                continue
            
            print(f"  Splitting: {csv_path.name}")
            split_csv(csv_path, dry_run=dry_run)
            total_processed += 1
    
    print(f"\nTotal files processed: {total_processed}")


def main():
    parser = argparse.ArgumentParser(description="Split G919 training CSVs into concentration segments")
    parser.add_argument("--data-dir", type=Path, default=Path("/root/UCI_dataset/.cache/G919-55"),
                        help="Path to G919 dataset directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing files")
    parser.add_argument("--single", type=Path, help="Process a single CSV file")
    
    args = parser.parse_args()
    
    if args.single:
        print(f"Splitting single file: {args.single}")
        split_csv(args.single, dry_run=args.dry_run)
    else:
        print(f"Processing G919 dataset at: {args.data_dir}")
        process_directory(args.data_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
