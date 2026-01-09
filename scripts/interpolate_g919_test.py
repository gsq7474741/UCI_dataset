#!/usr/bin/env python3
"""Interpolate G919 test samples back to original time resolution.

Test samples contain 90 key points selected by adaptive down-sampling.
This script interpolates them back to 1-second resolution based on timestamps.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


# Target length to match training set (mean of train samples)
TARGET_LENGTH = 7000  # Training set mean: 6947


def interpolate_test_sample(df: pd.DataFrame, method: str = 'linear', 
                           target_length: int = TARGET_LENGTH) -> pd.DataFrame:
    """Interpolate test sample from sparse key points to full time resolution.
    
    Args:
        df: DataFrame with Time column and sensor columns (90 rows)
        method: Interpolation method ('linear', 'cubic', 'quadratic')
        target_length: Target output length (extend with last value if needed)
    
    Returns:
        DataFrame with interpolated values at 1-second intervals
    """
    time_col = df.columns[0]  # First column is Time
    sensor_cols = [c for c in df.columns if c.startswith('sen.')]
    
    # Original sparse time points
    time_sparse = df[time_col].values
    time_start = int(time_sparse[0])
    time_end = int(time_sparse[-1])
    
    # Extend time_end to reach target_length if needed
    actual_end = max(time_end, time_start + target_length - 1)
    
    # Create dense time grid (1-second intervals)
    time_dense = np.arange(time_start, actual_end + 1, 1)
    
    # Interpolate each sensor channel
    result_data = {time_col: time_dense}
    
    for col in sensor_cols:
        values_sparse = df[col].values
        
        # Create interpolation function (only for original time range)
        f = interp1d(time_sparse, values_sparse, kind=method, 
                    bounds_error=False, fill_value=(values_sparse[0], values_sparse[-1]))
        
        # Interpolate to dense grid
        # For times beyond original range, use last value (response returned to baseline)
        values_dense = f(time_dense)
        result_data[col] = values_dense
    
    return pd.DataFrame(result_data)


def process_test_file(csv_path: Path, output_dir: Path = None, 
                      method: str = 'linear', dry_run: bool = False) -> Path:
    """Process a single test CSV file."""
    df = pd.read_csv(csv_path)
    
    # Interpolate
    df_interp = interpolate_test_sample(df, method=method)
    
    # Output path
    if output_dir is None:
        output_dir = csv_path.parent
    
    out_name = f"{csv_path.stem}_interp.csv"
    out_path = output_dir / out_name
    
    if not dry_run:
        df_interp.to_csv(out_path, index=False)
        print(f"  Saved: {out_path.name} ({len(df)} -> {len(df_interp)} rows)")
    else:
        print(f"  [DRY RUN] Would save: {out_path.name} ({len(df)} -> {len(df_interp)} rows)")
    
    return out_path


def process_directory(data_dir: Path, method: str = 'linear', dry_run: bool = False):
    """Process all test CSV files in the G919 dataset directory."""
    categories = ['condiment', 'drink', 'fruit', 'milk', 'perfume', 'spice', 'vegetable', 'wine']
    
    total_files = 0
    
    for category in categories:
        test_dir = data_dir / category / f"{category}s_test"
        if not test_dir.exists():
            test_dir = data_dir / category / f"{category}_test"
        if not test_dir.exists():
            continue
        
        # Find all test CSV files (exclude already interpolated)
        csv_files = [f for f in test_dir.glob("*.csv") 
                    if not f.stem.endswith('_interp')]
        
        if csv_files:
            print(f"\nProcessing {category} ({len(csv_files)} files):")
            for csv_path in sorted(csv_files):
                print(f"  Interpolating: {csv_path.name}")
                process_test_file(csv_path, method=method, dry_run=dry_run)
                total_files += 1
    
    print(f"\nTotal files processed: {total_files}")


def main():
    parser = argparse.ArgumentParser(description="Interpolate G919 test samples")
    parser.add_argument("--data-dir", type=Path, 
                       default=Path(".cache/g919_55/single"),
                       help="Path to G919 dataset directory")
    parser.add_argument("--method", type=str, default="linear",
                       choices=['linear', 'cubic', 'quadratic'],
                       help="Interpolation method")
    parser.add_argument("--dry-run", action="store_true",
                       help="Don't actually save files")
    parser.add_argument("--single", type=Path, default=None,
                       help="Process a single CSV file")
    
    args = parser.parse_args()
    
    if args.single:
        print(f"Interpolating single file: {args.single}")
        process_test_file(Path(args.single), method=args.method, dry_run=args.dry_run)
    else:
        process_directory(args.data_dir, method=args.method, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
