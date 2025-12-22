import os
import json
from typing import List, Tuple, Dict, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Columns
FLOAT_COLS = [f"sensor_{i}" for i in range(10)]  # 10 sensors
LABEL_COL = "label_gas"
TS_COL = "t_s"  # seconds since start (baseline start)


class SequenceCSVDataset(Dataset):
    def __init__(
        self,
        root: str,
        file_list: List[str],
        float_cols: Optional[List[str]] = None,
        label_col: str = LABEL_COL,
        dtype: torch.dtype = torch.float32,
        ts_col: Optional[str] = TS_COL,
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
        try:
            usecols = self.float_cols + [self.label_col] + ([self.ts_col] if self.ts_col else [])
            df = pd.read_csv(path, usecols=usecols)
        except Exception:
            df = pd.read_csv(path)
        x = torch.tensor(df[self.float_cols].values, dtype=self.dtype)
        y_val = int(df[self.label_col].iloc[0])
        y = torch.tensor(y_val, dtype=torch.long)
        ts = None
        if self.ts_col and self.ts_col in df.columns:
            try:
                ts = torch.tensor(df[self.ts_col].values, dtype=torch.float32)
            except Exception:
                ts = None

        stem = os.path.splitext(filename)[0]
        meta = {"id": stem, "path": path, "length": x.shape[0]}
        if ts is not None:
            meta[self.ts_col] = ts
        return x, y, meta


def _read_splits(root: str) -> Dict[str, List[str]]:
    splits_path = os.path.join(root, "splits.json")
    if not os.path.isfile(splits_path):
        raise FileNotFoundError(f"splits.json not found at {splits_path}. Run split_data.py first.")
    with open(splits_path, "r") as f:
        data = json.load(f)
    return {"train": data.get("train", []), "val": data.get("val", []), "test": data.get("test", [])}


@torch.no_grad()
def compute_channel_stats(
    root: str,
    files: List[str],
    float_cols: Optional[List[str]] = None,
    *,
    anchor_zero: bool = True,
    t0_offset_s: int = 300,
    pre_s: int = 600,
    post_s: int = 1200,
    baseline_subtract: bool = True,
    baseline_s: int = 300,
    ts_col: str = TS_COL,
):
    cols = list(float_cols or FLOAT_COLS)
    n_channels = len(cols)
    total_count = 0
    sum_vec = torch.zeros(n_channels, dtype=torch.float64)
    sumsq_vec = torch.zeros(n_channels, dtype=torch.float64)

    for fname in files:
        fpath = os.path.join(root, fname)
        try:
            df = pd.read_csv(fpath, usecols=cols + [ts_col])
        except Exception:
            df = pd.read_csv(fpath)
        x = torch.tensor(df[cols].values, dtype=torch.float64)
        T = x.shape[0]
        ts = None
        if ts_col in df.columns:
            try:
                ts = torch.tensor(df[ts_col].values, dtype=torch.float64) - float(t0_offset_s)
            except Exception:
                ts = None

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

        if baseline_subtract:
            x_src = x
            ts_for_base = ts_w if (ts_w is not None) else ts
            if (anchor_zero and (ts_for_base is not None) and (ts_for_base.numel() == x_src.shape[0])):
                base_mask = (ts_for_base < 0) & (ts_for_base >= -float(baseline_s))
                base_idx = torch.nonzero(base_mask, as_tuple=False).squeeze(-1)
                if base_idx.numel() > 0:
                    baseline = x_src[base_idx].mean(dim=0, keepdim=True)
                else:
                    pre_zero_idx = torch.nonzero(ts_for_base < 0, as_tuple=False).squeeze(-1)
                    baseline = x_src[pre_zero_idx].mean(dim=0, keepdim=True) if pre_zero_idx.numel() > 0 else x_src[: min(100, x_src.shape[0])].mean(dim=0, keepdim=True)
            else:
                baseline = x_src[: min(100, x_src.shape[0])].mean(dim=0, keepdim=True)
            x = x - baseline

        total_count += x.shape[0]
        sum_vec += x.sum(dim=0)
        sumsq_vec += (x * x).sum(dim=0)

    mean = (sum_vec / max(1, total_count)).to(torch.float32)
    var = (sumsq_vec / max(1, total_count)) - (mean.to(torch.float64) ** 2)
    var = torch.clamp(var, min=1e-12).to(torch.float32)
    std = torch.sqrt(var)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


def _build_pad_collate(
    mean: Optional[torch.Tensor], std: Optional[torch.Tensor],
    pad_to: int, pad_value: float = 0.0, float_cols: Optional[List[str]] = None,
    anchor_zero: bool = True, t0_offset_s: int = 300, pre_s: int = 600, post_s: int = 1200,
    ts_key: str = TS_COL, baseline_subtract: bool = True, baseline_s: int = 300,
):
    cols = list(float_cols or FLOAT_COLS)
    c = len(cols)

    def _window_baseline_norm(x: torch.Tensor, ts: Optional[torch.Tensor]) -> torch.Tensor:
        T = x.shape[0]
        if anchor_zero and (ts is not None) and (ts.numel() == T):
            ts = ts - float(t0_offset_s)
            mask = (ts >= -float(pre_s)) & (ts <= float(post_s))
            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                x = x[idx]
                ts = ts[idx]
                T = x.shape[0]
        if baseline_subtract:
            if (anchor_zero and (ts is not None) and (ts.numel() == T)):
                base_mask = (ts < 0) & (ts >= -float(baseline_s))
                base_idx = torch.nonzero(base_mask, as_tuple=False).squeeze(-1)
                if base_idx.numel() > 0:
                    baseline = x[base_idx].mean(dim=0, keepdim=True)
                else:
                    pre_zero_idx = torch.nonzero(ts < 0, as_tuple=False).squeeze(-1)
                    baseline = x[pre_zero_idx].mean(dim=0, keepdim=True) if pre_zero_idx.numel() > 0 else x[: min(100, T)].mean(dim=0, keepdim=True)
            else:
                baseline = x[: min(100, T)].mean(dim=0, keepdim=True)
            x = x - baseline
        if mean is not None and std is not None:
            x = (x - mean.view(1, c)) / std.view(1, c)
        if T == pad_to:
            out = x
        elif T > pad_to:
            start = (T - pad_to) // 2
            out = x[start:start + pad_to]
        else:
            pad_len = pad_to - T
            pad = torch.full((pad_len, c), pad_value, dtype=x.dtype)
            out = torch.cat([x, pad], dim=0)
        return out.transpose(0, 1).contiguous()

    def collate(batch):
        xs, ys, metas = [], [], []
        for x, y, m in batch:
            ts = None
            if isinstance(m, dict) and ts_key in m:
                ts = m[ts_key]
            xs.append(_window_baseline_norm(x, ts))
            ys.append(y)
            metas.append(m)
        X = torch.stack(xs, dim=0)
        Y = torch.stack(ys, dim=0)
        return X, Y, metas

    return collate


def load_sequence_datasets(root: str, float_cols: Optional[List[str]] = None) -> Tuple[SequenceCSVDataset, SequenceCSVDataset, SequenceCSVDataset]:
    root = os.path.abspath(root)
    splits = _read_splits(root)
    train_ds = SequenceCSVDataset(root, splits["train"], float_cols=float_cols)
    val_ds = SequenceCSVDataset(root, splits["val"], float_cols=float_cols)
    test_ds = SequenceCSVDataset(root, splits["test"], float_cols=float_cols)
    return train_ds, val_ds, test_ds


def _collate_list(batch):
    xs, ys, metas = [], [], []
    for x, y, m in batch:
        xs.append(x)
        ys.append(y)
        metas.append(m)
    ys = torch.stack(ys, dim=0)
    return xs, ys, metas


def load_sequence_dataloaders(
    root: str,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    float_cols: Optional[List[str]] = None,
    pad_to: Optional[int] = None,
    normalize: bool = True,
    anchor_zero: bool = True,
    t0_offset_s: int = 300,
    pre_s: int = 600,
    post_s: int = 1200,
    baseline_subtract: bool = True,
    baseline_s: int = 300,
):
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
                t0_offset_s=t0_offset_s,
                pre_s=pre_s,
                post_s=post_s,
                baseline_subtract=baseline_subtract,
                baseline_s=baseline_s,
            )
        collate_fn = _build_pad_collate(
            mean, std, pad_to=pad_to, pad_value=0.0, float_cols=float_cols,
            anchor_zero=anchor_zero, t0_offset_s=t0_offset_s, pre_s=pre_s, post_s=post_s,
            ts_key=TS_COL, baseline_subtract=baseline_subtract, baseline_s=baseline_s,
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
