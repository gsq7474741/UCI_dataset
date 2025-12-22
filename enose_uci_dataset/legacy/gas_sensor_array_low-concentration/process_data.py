import os
import csv
import pandas as pd

# Input/Output paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(THIS_DIR, "raw")
INPUT_CSV = os.path.join(RAW_DIR, "gsalc.csv")
OUT_DIR = os.path.join(THIS_DIR, "processed", "v1", "ssl_samples")
os.makedirs(OUT_DIR, exist_ok=True)

# Sensor order in gsalc.csv description (10 sensors, 900 points each, 1Hz)
SENSOR_NAMES = [
    "sensor0",  # TGS2603
    "sensor1",  # TGS2630
    "sensor2",  # TGS813
    "sensor3",  # TGS822
    "sensor4",  # MQ-135
    "sensor5",  # MQ-137
    "sensor6",  # MQ-138
    "sensor7",  # 2M012
    "sensor8",  # VOCS-P
    "sensor9",  # 2SH12
]
POINTS_PER_SENSOR = 900  # per note.md
TOTAL_POINTS = POINTS_PER_SENSOR * len(SENSOR_NAMES)  # 9000

# Gas label mapping (alphabetical indices assigned deterministically)
# You may adjust mapping if you want a specific id ordering.
GASES = [
    "acetone",          # 丙酮
    "ethyl acetate",   # 乙酸乙酯
    "ethanol",         # 乙醇
    "hexane",          # 正己烷
    "isopropanol",     # 异丙醇
    "toluene",         # 甲苯
]
GAS2ID = {g: i for i, g in enumerate(GASES)}


def parse_sample_row(row_values):
    """
    Parse one row from gsalc.csv.
    Row format: [gas_label, conc_label, v0, v1, ..., v8999]
    Returns:
      gas_label (str), conc_label (str), X [POINTS_PER_SENSOR, 10] as DataFrame
    """
    gas_label = row_values[0].strip()
    conc_label = row_values[1].strip()
    floats = list(map(float, row_values[2:]))
    if len(floats) != TOTAL_POINTS:
        raise ValueError(f"Expected {TOTAL_POINTS} values per row, got {len(floats)}")

    # Reconstruct matrix: for time t in [0..899], channel c reads floats[c*900 + t]
    data = {
        f"sensor_{i}": [floats[i * POINTS_PER_SENSOR + t] for t in range(POINTS_PER_SENSOR)]
        for i in range(10)
    }
    df = pd.DataFrame(data)
    # Add time in seconds since start (no explicit induction time in this dataset file)
    df["t_s"] = list(range(POINTS_PER_SENSOR))
    # Add labels
    df["label_gas"] = GAS2ID.get(gas_label, -1)
    df["label_gas_name"] = gas_label
    df["conc_ppb"] = conc_label
    return gas_label, conc_label, df


def main():
    if not os.path.isfile(INPUT_CSV):
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}. Run extract.sh to populate raw/gsalc.csv")

    # Read with csv module to avoid dtype issues
    with open(INPUT_CSV, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # If file contains a header, try to detect and skip non-numeric length row
    start_idx = 0
    if rows:
        maybe_header = rows[0]
        if len(maybe_header) < 2 or maybe_header[0].lower() in ("label", "gas", "class"):
            start_idx = 1

    count = 0
    for i in range(start_idx, len(rows)):
        row = rows[i]
        gas_label, conc_label, df = parse_sample_row(row)
        # File name: idx_gas_conc.csv (safe chars)
        gas_safe = gas_label.replace(" ", "_")
        conc_safe = conc_label.replace(" ", "").replace("/", "-")
        out_path = os.path.join(OUT_DIR, f"{i-start_idx:03d}_{gas_safe}_{conc_safe}.csv")
        df.to_csv(out_path, index=False)
        count += 1

    print(f"Generated {count} CSV samples to {OUT_DIR}")


if __name__ == "__main__":
    main()
