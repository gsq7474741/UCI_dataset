#!/usr/bin/env python3
"""Split G919 interpolated test CSV files into 3 concentration segments (low, medium, high).

Each interpolated test CSV contains 3 consecutive concentration tests.
This script splits them into separate files using the same algorithm as training data.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from split_g919_train import find_split_points, find_response_end, EXPECTED_SEGMENT_LENGTH, SEGMENT_LENGTH_MAX


def split_interp_csv(csv_path: Path, output_dir: Path = None, dry_run: bool = False):
    """Split a single interpolated test CSV file into 3 concentration segments."""
    df = pd.read_csv(csv_path)
    
    # Get sensor columns for signal analysis
    sensor_cols = [c for c in df.columns if c.startswith("sen.")]
    
    # Get split points using same algorithm as training data
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
        if len(seg) > 0:
            time_col = seg.columns[0]
            seg[time_col] = seg[time_col] - seg[time_col].iloc[0]
    
    # Output
    if output_dir is None:
        output_dir = csv_path.parent
    
    # Remove _interp suffix for cleaner naming
    stem = csv_path.stem
    if stem.endswith('_interp'):
        stem = stem[:-7]
    
    concentration_names = ['low', 'mid', 'high']
    output_paths = []
    
    for i, (seg, conc) in enumerate(zip(segments, concentration_names)):
        out_name = f"{stem}_interp_{conc}.csv"
        out_path = output_dir / out_name
        output_paths.append(out_path)
        
        if not dry_run:
            seg.to_csv(out_path, index=False)
            print(f"  Saved: {out_path.name} ({len(seg)} rows)")
        else:
            print(f"  [DRY RUN] Would save: {out_path.name} ({len(seg)} rows)")
    
    return output_paths


def process_directory(data_dir: Path, dry_run: bool = False):
    """Process all interpolated test CSV files in the G919 dataset directory."""
    categories = ['condiment', 'drink', 'fruit', 'milk', 'perfume', 'spice', 'vegetable', 'wine']
    
    total_files = 0
    
    for category in categories:
        # Find test directory
        test_dir = data_dir / category / f"{category}s_test"
        if not test_dir.exists():
            test_dir = data_dir / category / f"{category}_test"
        if not test_dir.exists():
            continue
        
        # Find all interpolated test CSV files
        interp_files = [f for f in test_dir.glob("*_interp.csv") 
                       if not any(x in f.stem for x in ['_low', '_mid', '_high'])]
        
        if interp_files:
            print(f"\nProcessing {category} ({len(interp_files)} files):")
            for csv_path in sorted(interp_files):
                print(f"  Splitting: {csv_path.name}")
                split_interp_csv(csv_path, dry_run=dry_run)
                total_files += 1
    
    print(f"\nTotal files processed: {total_files}")


def main():
    parser = argparse.ArgumentParser(description="Split G919 interpolated test CSV files")
    parser.add_argument("--data-dir", type=Path, 
                       default=Path(".cache/g919_55/single"),
                       help="Path to G919 dataset directory")
    parser.add_argument("--dry-run", action="store_true",
                       help="Don't actually save files")
    
    args = parser.parse_args()
    process_directory(args.data_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
