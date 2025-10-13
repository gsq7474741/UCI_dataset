import os
import json
import random
import pandas as pd
from collections import defaultdict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(THIS_DIR, "processed", "v1", "ssl_samples")
SPLITS_PATH = os.path.join(ROOT, "splits.json")

RANDOM_SEED = int(os.environ.get("SPLIT_SEED", 42))
TRAIN_RATIO = float(os.environ.get("TRAIN_RATIO", 0.7))
VAL_RATIO = float(os.environ.get("VAL_RATIO", 0.15))
TEST_RATIO = float(os.environ.get("TEST_RATIO", 0.15))

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, "Ratios must sum to 1"


def read_label_tuple(csv_path: str) -> str:
    # labels are constant per file; read first row of each
    df = pd.read_csv(csv_path, nrows=1)
    e = float(df["ethylene_conc"].iloc[0]) if "ethylene_conc" in df.columns else 0.0
    c = float(df["co_conc"].iloc[0]) if "co_conc" in df.columns else 0.0
    m = float(df["methane_conc"].iloc[0]) if "methane_conc" in df.columns else 0.0
    return f"e{e:.3f}_c{c:.3f}_m{m:.3f}"


def main():
    random.seed(RANDOM_SEED)
    if not os.path.isdir(ROOT):
        raise FileNotFoundError(f"Processed dir not found: {ROOT}. Run process_data.py first")

    files = [f for f in os.listdir(ROOT) if f.endswith('.csv')]
    if not files:
        raise RuntimeError("No CSV files found. Run process_data.py first")

    per_bucket = defaultdict(list)
    for f in files:
        path = os.path.join(ROOT, f)
        bucket = read_label_tuple(path)
        per_bucket[bucket].append(f)

    train, val, test = [], [], []
    for bucket, fs in per_bucket.items():
        fs.sort()
        random.shuffle(fs)
        n = len(fs)
        n_train = round(n * TRAIN_RATIO)
        n_val = round(n * VAL_RATIO)
        n_test = n - n_train - n_val
        train.extend(fs[:n_train])
        val.extend(fs[n_train:n_train + n_val])
        test.extend(fs[n_train + n_val:])

    splits = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
    os.makedirs(ROOT, exist_ok=True)
    with open(SPLITS_PATH, 'w') as f:
        json.dump(splits, f, indent=2)

    print(f"Stratified (by concentration tuple) split saved to {SPLITS_PATH}")
    print(f"Train/Val/Test: {len(train)}/{len(val)}/{len(test)} (total {len(files)})")


if __name__ == "__main__":
    main()
