import os
import json
from typing import List, Tuple, Dict, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


SENSOR_COLS = [f"sensor_{i}" for i in range(8)]
ENV_COLS = ["temp", "humidity"]
FLOAT_COLS = SENSOR_COLS + ENV_COLS  # 10 channels total
LABEL_COL = "label_gas"


class SequenceCSVDataset(Dataset):
    """
    Sequence dataset for SSL/sequence tasks backed by per-sample CSV files.

    - Each CSV file contains columns:
      sensor_0..sensor_7, temp, humidity, date, label_gas
    - label_gas is mapped as: banana->0, wine->1, background->2
    - Returns (X, y, meta):
      X: FloatTensor [T, C] where C=len(FLOAT_COLS)=10
      y: LongTensor [] (scalar class id)
      meta: dict with keys {"id", "class", "path", "length"}
    """

    def __init__(
        self,
        root: str,
        file_list: List[str],
        float_cols: Optional[List[str]] = None,
        label_col: str = LABEL_COL,
        dtype: torch.dtype = torch.float32,
        ts_col: Optional[str] = "t_s",
    ) -> None:
        self.root = os.path.abspath(root)
        self.files = list(file_list)
        self.float_cols = list(float_cols or FLOAT_COLS)
        self.label_col = label_col
        self.dtype = dtype
        self.ts_col = ts_col

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        filename = self.files[idx]
        path = os.path.join(self.root, filename)
        # Read CSV; prefer selecting columns but fall back to reading full if ts_col must be detected
        try:
            usecols = self.float_cols + [self.label_col] + ([self.ts_col] if self.ts_col else [])
            df = pd.read_csv(path, usecols=usecols)
        except Exception:
            df = pd.read_csv(path)
        # Features [T, C]
        x = torch.tensor(df[self.float_cols].values, dtype=self.dtype)
        # Label: first row (constant per file)
        y_val = int(df[self.label_col].iloc[0])
        y = torch.tensor(y_val, dtype=torch.long)
        # Optional time (seconds from induction start)
        ts = None
        if self.ts_col and self.ts_col in df.columns:
            try:
                ts = torch.tensor(df[self.ts_col].values, dtype=torch.float32)
            except Exception:
                ts = None

        # Parse id and class from filename: "{id}_{class}.csv"
        stem = os.path.splitext(filename)[0]
        try:
            sample_id_str, cls_name = stem.split("_", 1)
        except ValueError:
            sample_id_str, cls_name = stem, "unknown"

        meta = {
            "id": int(sample_id_str) if sample_id_str.isdigit() else sample_id_str,
            "class": cls_name,
            "path": path,
            "length": x.shape[0],
        }
        if ts is not None:
            meta["t_s"] = ts  # [T]
        return x, y, meta


def _read_splits(root: str) -> Dict[str, List[str]]:
    splits_path = os.path.join(root, "splits.json")
    if not os.path.isfile(splits_path):
        raise FileNotFoundError(f"splits.json not found at {splits_path}. Run split_data.py first.")
    with open(splits_path, "r") as f:
        data = json.load(f)
    return {
        "train": data.get("train", []),
        "val": data.get("val", []),
        "test": data.get("test", []),
    }


@torch.no_grad()
def compute_channel_stats(
    root: str,
    files: List[str],
    float_cols: Optional[List[str]] = None,
    *,
    anchor_zero: bool = False,
    pre_s: int = 1800,
    post_s: int = 1800,
    baseline_subtract: bool = False,
    baseline_s: int = 300,
    ts_col: str = "t_s",
):
    """Compute per-channel mean and std across a list of CSV files after applying
    the same preprocessing pipeline used at training time (anchor windowing and
    optional baseline subtraction). Returns (mean[C], std[C]) as float32 tensors.
    """
    cols = list(float_cols or FLOAT_COLS)
    n_channels = len(cols)
    total_count = 0
    sum_vec = torch.zeros(n_channels, dtype=torch.float64)
    sumsq_vec = torch.zeros(n_channels, dtype=torch.float64)

    for fname in files:
        fpath = os.path.join(root, fname)
        # Load potentially with ts for window/baseline decisions
        try:
            df = pd.read_csv(fpath, usecols=cols + [ts_col])
        except Exception:
            df = pd.read_csv(fpath)
        x = torch.tensor(df[cols].values, dtype=torch.float64)  # [T, C]
        T = x.shape[0]

        ts = None
        if ts_col in df.columns:
            try:
                ts = torch.tensor(df[ts_col].values, dtype=torch.float64)
            except Exception:
                ts = None

        # Anchor-window to [-pre_s, +post_s] around t=0
        if anchor_zero and (ts is not None) and (ts.numel() == T):
            mask = (ts >= -float(pre_s)) & (ts <= float(post_s))
            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                x = x[idx]
                ts_w = ts[idx]
            else:
                ts_w = None
        else:
            ts_w = None

        # Baseline subtraction using pre-zero segment up to baseline_s seconds
        if baseline_subtract:
            x_base_src = x
            ts_for_base = ts_w if (ts_w is not None) else ts
            if (anchor_zero and (ts_for_base is not None) and (ts_for_base.numel() == x_base_src.shape[0])):
                base_mask = (ts_for_base < 0) & (ts_for_base >= -float(baseline_s))
                base_idx = torch.nonzero(base_mask, as_tuple=False).squeeze(-1)
                if base_idx.numel() > 0:
                    baseline = x_base_src[base_idx].mean(dim=0, keepdim=True)
                else:
                    # Fallback: use earliest portion before zero if available, else first min(100, T)
                    if ts_for_base is not None:
                        pre_zero_idx = torch.nonzero(ts_for_base < 0, as_tuple=False).squeeze(-1)
                    else:
                        pre_zero_idx = torch.arange(min(100, x_base_src.shape[0]))
                    if pre_zero_idx.numel() > 0:
                        baseline = x_base_src[pre_zero_idx].mean(dim=0, keepdim=True)
                    else:
                        baseline = x_base_src[: min(100, x_base_src.shape[0])].mean(dim=0, keepdim=True)
            else:
                baseline = x_base_src[: min(100, x_base_src.shape[0])].mean(dim=0, keepdim=True)
            x = x - baseline

        total_count += x.shape[0]
        sum_vec += x.sum(dim=0)
        sumsq_vec += (x * x).sum(dim=0)

    mean = (sum_vec / max(1, total_count)).to(torch.float32)
    var = (sumsq_vec / max(1, total_count)) - (mean.to(torch.float64) ** 2)
    var = torch.clamp(var, min=1e-12).to(torch.float32)
    std = torch.sqrt(var)
    # Avoid div by zero
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


def _build_pad_collate(mean: Optional[torch.Tensor], std: Optional[torch.Tensor],
                       pad_to: int, pad_value: float = 0.0, float_cols: Optional[List[str]] = None,
                       anchor_zero: bool = False, pre_s: int = 1800, post_s: int = 1800,
                       ts_key: str = "t_s", baseline_subtract: bool = False, baseline_s: int = 300):
    cols = list(float_cols or FLOAT_COLS)
    c = len(cols)

    def _center_crop_or_pad(x: torch.Tensor, ts: Optional[torch.Tensor]) -> torch.Tensor:
        # x: [T, C]
        T = x.shape[0]
        # 1) Anchor-based window to [-pre_s, +post_s]
        if anchor_zero and (ts is not None) and (ts.numel() == T):
            # Window around t=0 in seconds
            mask = (ts >= -float(pre_s)) & (ts <= float(post_s))
            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                x = x[idx]  # restrict to window
                ts = ts[idx]
                T = x.shape[0]

        # 2) Baseline subtraction using pre-zero segment up to baseline_s seconds
        if baseline_subtract:
            if (anchor_zero and (ts is not None) and (ts.numel() == T)):
                base_mask = (ts < 0) & (ts >= -float(baseline_s))
                base_idx = torch.nonzero(base_mask, as_tuple=False).squeeze(-1)
                if base_idx.numel() > 0:
                    baseline = x[base_idx].mean(dim=0, keepdim=True)
                else:
                    pre_zero_idx = torch.nonzero(ts < 0, as_tuple=False).squeeze(-1)
                    if pre_zero_idx.numel() > 0:
                        baseline = x[pre_zero_idx].mean(dim=0, keepdim=True)
                    else:
                        baseline = x[: min(100, T)].mean(dim=0, keepdim=True)
            else:
                baseline = x[: min(100, T)].mean(dim=0, keepdim=True)
            x = x - baseline

        # 3) Normalize using provided mean/std (computed with same preprocessing)
        if mean is not None and std is not None:
            x = (x - mean.view(1, c)) / std.view(1, c)

        # 4) Pad/Crop to fixed length
        if T == pad_to:
            out = x
        elif T > pad_to:
            # center crop the (possibly windowed) sequence
            start = (T - pad_to) // 2
            out = x[start:start + pad_to]
        else:
            # pad at the end
            pad_len = pad_to - T
            pad = torch.full((pad_len, c), pad_value, dtype=x.dtype)
            out = torch.cat([x, pad], dim=0)
        # [pad_to, C] -> [C, pad_to]
        return out.transpose(0, 1).contiguous()

    def collate(batch):
        # batch: list of (x[T,C], y, meta)
        xs, ys, metas = [], [], []
        for x, y, m in batch:
            ts = None
            if isinstance(m, dict) and ts_key in m:
                ts = m[ts_key]
            xs.append(_center_crop_or_pad(x, ts))
            ys.append(y)
            metas.append(m)
        X = torch.stack(xs, dim=0)  # [B, C, T]
        Y = torch.stack(ys, dim=0)  # [B]
        return X, Y, metas

    return collate


def load_sequence_datasets(
    root: str,
    float_cols: Optional[List[str]] = None,
) -> Tuple[SequenceCSVDataset, SequenceCSVDataset, SequenceCSVDataset]:
    """
    One-liner-friendly helper to build train/val/test datasets from CSV + splits.json.

    Example (idea seed code):
        from UCI_dataset.enose_uci_dataset.gas_sensors_for_home_activity_monitoring.sequence_loader import load_sequence_datasets
        train_ds, val_ds, test_ds = load_sequence_datasets(
            "UCI_dataset/enose_uci_dataset/gas_sensors_for_home_activity_monitoring/processed/v1/ssl_samples"
        )
    """
    root = os.path.abspath(root)
    splits = _read_splits(root)
    train_ds = SequenceCSVDataset(root, splits["train"], float_cols=float_cols)
    val_ds = SequenceCSVDataset(root, splits["val"], float_cols=float_cols)
    test_ds = SequenceCSVDataset(root, splits["test"], float_cols=float_cols)
    return train_ds, val_ds, test_ds


def _collate_list(batch):
    """Default collate: returns lists of tensors/meta for variable-length sequences.
    batch: list of (x[T,C], y[], meta)
    returns: xs(list[Tensor]), ys(LongTensor[N]), metas(list[dict])
    """
    xs, ys, metas = [], [], []
    for x, y, m in batch:
        xs.append(x)
        ys.append(y)
        metas.append(m)
    ys = torch.stack(ys, dim=0)  # [N]
    return xs, ys, metas


def load_sequence_dataloaders(
    root: str,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    float_cols: Optional[List[str]] = None,
    pad_to: Optional[int] = None,
    normalize: bool = False,
    anchor_zero: bool = False,
    pre_s: int = 1800,
    post_s: int = 1800,
    baseline_subtract: bool = False,
    baseline_s: int = 300,
):
    """
    Convenience helper to return train/val/test DataLoaders.
    - If pad_to is None: returns list-collate for variable-length batches (xs is list of [T,C]).
    - If pad_to is not None: returns tensors with shape [B, C, T], optionally normalized using train stats.
    """
    train_ds, val_ds, test_ds = load_sequence_datasets(root, float_cols=float_cols)

    if pad_to is None:
        collate_fn = _collate_list
    else:
        mean = std = None
        if normalize:
            mean, std = compute_channel_stats(
                root,
                train_ds.files,
                float_cols=float_cols,
                anchor_zero=anchor_zero,
                pre_s=pre_s,
                post_s=post_s,
                baseline_subtract=baseline_subtract,
                baseline_s=baseline_s,
            )
        collate_fn = _build_pad_collate(
            mean, std, pad_to=pad_to, pad_value=0.0, float_cols=float_cols,
            anchor_zero=anchor_zero, pre_s=pre_s, post_s=post_s,
            ts_key="t_s", baseline_subtract=baseline_subtract, baseline_s=baseline_s,
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader
