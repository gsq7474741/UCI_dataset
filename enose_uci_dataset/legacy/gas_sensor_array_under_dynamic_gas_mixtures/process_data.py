import os
from typing import Literal
import pandas as pd
import numpy as np
from tqdm import tqdm


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR_A = os.path.join(THIS_DIR, "raw")
RAW_DIR_B = os.path.join(THIS_DIR, ".raw")
RAW_DIR = RAW_DIR_A if os.path.isdir(RAW_DIR_A) else RAW_DIR_B
OUT_DIR = os.path.join(THIS_DIR, "processed", "v1", "ssl_samples")
os.makedirs(OUT_DIR, exist_ok=True)


def convert_to_csv(input_file: str) -> str:
    """Parse raw text into a well-formed CSV with header.
    Input lines are whitespace separated, with possible variable spaces. Returns the CSV path.
    Columns: time, gas1_conc, ethylene_conc, sensor_0..sensor_15
    """
    csv_path = input_file + ".csv"
    with open(input_file, "r") as fin, open(csv_path, "w") as fout:
        header = ["time", "gas1_conc", "ethylene_conc"] + [f"sensor_{i}" for i in range(16)]
        fout.write(",".join(header) + "\n")
        # Skip the first header line in raw
        first = fin.readline()
        for line in fin:
            parts = line.strip().split()
            if not parts:
                continue
            # Some files might repeat header mid-file; skip non-numeric lines
            try:
                float(parts[0])
            except Exception:
                continue
            fout.write(",".join(parts) + "\n")
    return csv_path


def process_file(input_csv: str, out_dir: str, gas_type: Literal["co", "methane"], sample_rate_hz: int = 100):
    df = pd.read_csv(input_csv)
    # Ensure columns
    df.columns = ["time", "gas1_conc", "ethylene_conc"] + [f"sensor_{i}" for i in range(16)]

    # Coerce numeric types and sanitize
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["gas1_conc"] = pd.to_numeric(df["gas1_conc"], errors="coerce")
    df["ethylene_conc"] = pd.to_numeric(df["ethylene_conc"], errors="coerce")
    for i in range(16):
        col = f"sensor_{i}"
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop leading non-numeric rows if any
    df = df.dropna(how="all")
    df = df.reset_index(drop=True)

    # Convert sensor readings to resistance in kOhm: 40.0 / S_i (safe division)
    eps = 1e-12
    for i in range(16):
        col = f"sensor_{i}"
        vals = df[col].values.astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            res_kohm = np.where(np.isfinite(vals) & (np.abs(vals) > eps), 40.0 / vals, np.nan)
        df[col] = res_kohm

    # Forward/backward fill sensor columns to remove NaNs caused by zeros or bad lines
    df[[f"sensor_{i}" for i in range(16)]] = df[[f"sensor_{i}" for i in range(16)]].ffill().bfill()
    # If still any NaNs remain in critical columns, drop those rows
    df = df.dropna(subset=["ethylene_conc", "gas1_conc"] + [f"sensor_{i}" for i in range(16)])
    df = df.reset_index(drop=True)
    if df.empty:
        return

    # Build specific gas concentration columns
    if gas_type == "co":
        df = df.rename(columns={"gas1_conc": "co_conc"})
        df["methane_conc"] = 0.0
    else:  # methane
        df = df.rename(columns={"gas1_conc": "methane_conc"})
        df["co_conc"] = 0.0

    # Keep useful columns and segment when any concentration changes
    keep_cols = [f"sensor_{i}" for i in range(16)] + ["ethylene_conc", "co_conc", "methane_conc"]
    df = df[keep_cols]

    conc_cols = ["ethylene_conc", "co_conc", "methane_conc"]
    sample_id = 0
    start_idx = 0
    if len(df) == 0:
        return
    prev = df.loc[0, conc_cols].values.astype(float)

    for i in tqdm(range(1, len(df)), desc=f"Segmenting {gas_type}"):
        cur = df.loc[i, conc_cols].values.astype(float)
        if not np.array_equal(cur, prev):
            seg = df.iloc[start_idx:i].copy()
            if len(seg) > 0:
                # Add time within segment in seconds
                seg["t_s"] = np.arange(len(seg), dtype=float) / float(sample_rate_hz)
                out_path = os.path.join(out_dir, f"{gas_type}_{sample_id:05d}.csv")
                # Reorder to sensors, t_s, labels
                seg = seg[[f"sensor_{k}" for k in range(16)] + ["t_s"] + conc_cols]
                # Final sanity: drop rows with non-finite
                seg = seg.replace([np.inf, -np.inf], np.nan).dropna()
                if len(seg) == 0:
                    start_idx = i
                    prev = cur
                    continue
                seg.to_csv(out_path, index=False)
                sample_id += 1
            start_idx = i
            prev = cur

    # Last segment
    seg = df.iloc[start_idx:].copy()
    if len(seg) > 0:
        seg["t_s"] = np.arange(len(seg), dtype=float) / float(sample_rate_hz)
        out_path = os.path.join(out_dir, f"{gas_type}_{sample_id:05d}.csv")
        seg = seg[[f"sensor_{k}" for k in range(16)] + ["t_s"] + conc_cols]
        seg = seg.replace([np.inf, -np.inf], np.nan).dropna()
        if len(seg) == 0:
            return
        seg.to_csv(out_path, index=False)


def main():
    # Resolve input files
    in_co = os.path.join(RAW_DIR, "ethylene_CO.txt")
    in_methane = os.path.join(RAW_DIR, "ethylene_methane.txt")
    if not os.path.isfile(in_co) or not os.path.isfile(in_methane):
        raise FileNotFoundError(f"Could not find input files in {RAW_DIR}. Expected ethylene_CO.txt and ethylene_methane.txt")

    # Convert to csv and process
    co_csv = convert_to_csv(in_co)
    process_file(co_csv, OUT_DIR, gas_type="co", sample_rate_hz=100)

    methane_csv = convert_to_csv(in_methane)
    process_file(methane_csv, OUT_DIR, gas_type="methane", sample_rate_hz=100)

    print(f"Done. Wrote samples to {OUT_DIR}")


if __name__ == "__main__":
    main()
