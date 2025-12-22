from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_SCRIPT = _REPO_ROOT / "examples" / "train_twin_timeseries.py"


@dataclass(frozen=True)
class RunRecord:
    run_dir: Path
    zero_channels: Optional[int]
    zero_channel_mask: Optional[int]
    zero_channel_indices: Tuple[int, ...]


def _find_latest_version_dir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        m = re.match(r"^version_(\d+)$", p.name)
        if not m:
            continue
        candidates.append((int(m.group(1)), p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _parse_zero_from_path(p: Path) -> Optional[int]:
    for parent in [p] + list(p.parents):
        m = re.match(r"^zero_(\d+)$", parent.name)
        if m:
            return int(m.group(1))
    return None


def _parse_mask_from_path(p: Path) -> Optional[int]:
    for parent in [p] + list(p.parents):
        m = re.match(r"^mask_(\d+)$", parent.name)
        if m:
            return int(m.group(1))
        m = re.match(r"^mask_0x([0-9a-fA-F]+)$", parent.name)
        if m:
            return int(m.group(1), 16)
    return None


def _load_run_meta(run_dir: Path) -> Dict:
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def _load_epoch_metrics(run_dir: Path) -> pd.DataFrame:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.csv not found in {run_dir}")

    df = pd.read_csv(metrics_path)
    if "epoch" not in df.columns:
        raise RuntimeError(f"Unexpected metrics format: missing 'epoch' in {metrics_path}")

    # MTL metrics include both classification and regression
    metric_cols = [c for c in [
        "train/acc", "val/acc", 
        "train/r2", "val/r2",
        "train/cls_loss", "val/cls_loss",
        "train/reg_loss", "val/reg_loss",
        "train/loss", "val/loss"
    ] if c in df.columns]

    def _last_valid(s: pd.Series) -> float:
        s = s.dropna()
        if len(s) == 0:
            return float("nan")
        return float(s.iloc[-1])

    agg = {c: _last_valid for c in metric_cols}
    out = df.groupby("epoch", as_index=False).agg(agg)
    return out.sort_values("epoch").reset_index(drop=True)


def discover_runs(runs_root: Path) -> List[RunRecord]:
    records: List[RunRecord] = []

    for metrics_path in sorted(runs_root.rglob("metrics.csv")):
        run_dir = metrics_path.parent
        meta = _load_run_meta(run_dir)

        zero_channels: Optional[int]
        if "zero_channels" in meta:
            zero_channels = int(meta["zero_channels"])
        else:
            zero_channels = _parse_zero_from_path(run_dir)

        zero_channel_mask: Optional[int]
        if "zero_channel_mask" in meta:
            zero_channel_mask = int(meta["zero_channel_mask"])
        else:
            zero_channel_mask = _parse_mask_from_path(run_dir)

        zidx = tuple(int(i) for i in meta.get("zero_channel_indices", []))
        records.append(
            RunRecord(
                run_dir=run_dir,
                zero_channels=zero_channels,
                zero_channel_mask=zero_channel_mask,
                zero_channel_indices=zidx,
            )
        )

    unique: Dict[Path, RunRecord] = {r.run_dir: r for r in records}
    return sorted(unique.values(), key=lambda r: str(r.run_dir))


def run_sweep(args) -> List[Path]:
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    run_dirs: List[Path] = []

    def _has_metrics(dir_: Path) -> bool:
        for p in dir_.glob("version_*/metrics.csv"):
            if p.exists():
                return True
        return False

    def _ensure_disk_cache() -> None:
        root = Path(args.root).expanduser().resolve()
        ds_dir = root / "twin_gas_sensor_arrays"
        cache_dir = ds_dir / "cache" / "twin_gas_sensor_arrays_npy_v1"
        raw_dir = ds_dir / "raw"
        txt = list(raw_dir.glob("*.txt"))
        if not txt and (raw_dir / "data1").exists():
            txt = list((raw_dir / "data1").glob("*.txt"))
        expected = len(txt)
        existing = len(list(cache_dir.glob("*.npy"))) if cache_dir.exists() else 0

        if expected > 0 and existing >= expected:
            return

        cmd = [
            sys.executable,
            str(_TRAIN_SCRIPT),
            "--root",
            str(args.root),
            "--num-workers",
            "0",
            "--build-disk-cache",
            "--disk-cache",
            "--ram-cache-disk",
        ]
        if args.download:
            cmd.append("--download")

        if args.dry_run:
            print(" ".join(cmd))
            return

        p = subprocess.run(cmd)
        if p.returncode != 0:
            raise RuntimeError(f"disk cache build failed, returncode={p.returncode}")

    if args.ensure_disk_cache:
        _ensure_disk_cache()

    def _build_base_cmd() -> List[str]:
        """Build base command with MTL baseline parameters."""
        cmd = [
            sys.executable,
            str(_TRAIN_SCRIPT),
            "--root", str(args.root),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--seed", str(args.seed),
            "--num-workers", str(args.num_workers),
            "--test-size", str(args.test_size),
            "--seq-len", str(args.seq_len),
            "--disk-cache",
            # MTL specific
            "--task", str(args.task),
            "--reg-target", str(args.reg_target),
            "--reg-weight", str(float(args.reg_weight)),
            "--reg-loss-scale", str(float(args.reg_loss_scale)),
            # Backbone
            "--backbone", str(args.backbone),
            # LR scheduler
            "--lr-scheduler", str(args.lr_scheduler),
            "--lr-scheduler-t-max", str(args.lr_scheduler_t_max),
            # Truncation
            "--truncate-ratio", str(args.truncate_ratio),
            "--truncate-start-ratio", str(args.truncate_start_ratio),
            # Validation and checkpointing
            "--check-val-every-n-epoch", str(args.check_val_every_n_epoch),
            "--ckpt-every-n-epochs", str(args.ckpt_every_n_epochs),
        ]
        
        if args.early_stopping:
            cmd.append("--early-stopping")
            cmd.extend(["--early-stopping-patience", str(args.early_stopping_patience)])
        
        if args.train_ram_cache:
            cmd.append("--ram-cache")
            cmd.extend(["--ram-cache-max-samples", str(int(args.train_ram_cache_max_samples))])
            if bool(args.train_ram_cache_disk):
                cmd.append("--ram-cache-disk")
            else:
                cmd.append("--no-ram-cache-disk")
        
        if args.download:
            cmd.append("--download")
        
        return cmd

    if args.sweep == "count":
        for k in range(int(args.min_zero), int(args.max_zero) + 1):
            group_dir = out_root / f"zero_{k}"
            group_dir.mkdir(parents=True, exist_ok=True)
            if (not args.overwrite) and _has_metrics(group_dir):
                latest = _find_latest_version_dir(group_dir)
                if latest is not None:
                    run_dirs.append(latest)
                continue

            cmd = _build_base_cmd()
            cmd.extend(["--zero-channels", str(k)])
            cmd.extend(["--logdir", str(group_dir)])

            if args.dry_run:
                print(" ".join(cmd))
                continue

            p = subprocess.run(cmd)
            if p.returncode != 0:
                if args.continue_on_fail:
                    print(f"[warn] run failed (zero_channels={k}), returncode={p.returncode}")
                    continue
                raise RuntimeError(f"run failed (zero_channels={k}), returncode={p.returncode}")

            latest = _find_latest_version_dir(group_dir)
            if latest is None:
                raise RuntimeError(f"No version dir produced under {group_dir}")
            run_dirs.append(latest)

    elif args.sweep == "mask":
        for mask in range(int(args.min_mask), int(args.max_mask) + 1):
            group_dir = out_root / f"mask_{mask:03d}"
            group_dir.mkdir(parents=True, exist_ok=True)
            if (not args.overwrite) and _has_metrics(group_dir):
                latest = _find_latest_version_dir(group_dir)
                if latest is not None:
                    run_dirs.append(latest)
                continue

            cmd = _build_base_cmd()
            cmd.extend(["--zero-channel-mask", str(mask)])
            cmd.extend(["--logdir", str(group_dir)])

            if args.dry_run:
                print(" ".join(cmd))
                continue

            p = subprocess.run(cmd)
            if p.returncode != 0:
                if args.continue_on_fail:
                    print(f"[warn] run failed (mask={mask}), returncode={p.returncode}")
                    continue
                raise RuntimeError(f"run failed (mask={mask}), returncode={p.returncode}")

            latest = _find_latest_version_dir(group_dir)
            if latest is None:
                raise RuntimeError(f"No version dir produced under {group_dir}")
            run_dirs.append(latest)

    elif args.sweep == "hyperparam":
        # Sweep over hyperparameters with fixed zero_channel_mask=0
        param_grid = _build_hyperparam_grid(args)
        
        for i, params in enumerate(param_grid):
            param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
            group_dir = out_root / f"hp_{i:03d}_{param_str[:50]}"
            group_dir.mkdir(parents=True, exist_ok=True)
            
            if (not args.overwrite) and _has_metrics(group_dir):
                latest = _find_latest_version_dir(group_dir)
                if latest is not None:
                    run_dirs.append(latest)
                continue

            cmd = _build_base_cmd()
            cmd.extend(["--zero-channel-mask", str(args.zero_channel_mask)])
            cmd.extend(["--logdir", str(group_dir)])
            
            # Override with sweep params
            for param_name, param_value in params.items():
                cmd_param = f"--{param_name.replace('_', '-')}"
                # Find and replace existing param or append
                try:
                    idx = cmd.index(cmd_param)
                    cmd[idx + 1] = str(param_value)
                except ValueError:
                    cmd.extend([cmd_param, str(param_value)])

            if args.dry_run:
                print(" ".join(cmd))
                continue

            p = subprocess.run(cmd)
            if p.returncode != 0:
                if args.continue_on_fail:
                    print(f"[warn] run failed (params={params}), returncode={p.returncode}")
                    continue
                raise RuntimeError(f"run failed (params={params}), returncode={p.returncode}")

            latest = _find_latest_version_dir(group_dir)
            if latest is None:
                raise RuntimeError(f"No version dir produced under {group_dir}")
            run_dirs.append(latest)

    return run_dirs


def _build_hyperparam_grid(args) -> List[Dict]:
    """Build hyperparameter grid for sweep."""
    import itertools
    
    # Parse sweep ranges from args
    grid = {}
    
    if args.sweep_lr:
        grid["lr"] = [float(x) for x in args.sweep_lr.split(",")]
    if args.sweep_reg_weight:
        grid["reg_weight"] = [float(x) for x in args.sweep_reg_weight.split(",")]
    if args.sweep_seq_len:
        grid["seq_len"] = [int(x) for x in args.sweep_seq_len.split(",")]
    if args.sweep_truncate_ratio:
        grid["truncate_ratio"] = [float(x) for x in args.sweep_truncate_ratio.split(",")]
    if args.sweep_batch_size:
        grid["batch_size"] = [int(x) for x in args.sweep_batch_size.split(",")]
    
    if not grid:
        # Default grid if none specified
        grid = {
            "lr": [1e-4, 5e-4, 1e-3],
            "reg_weight": [0.5, 1.0, 2.0],
        }
    
    keys = list(grid.keys())
    values = list(grid.values())
    combinations = list(itertools.product(*values))
    
    return [dict(zip(keys, combo)) for combo in combinations]


def make_plots(records: List[RunRecord], *, out_dir: Path, title: str, task: str = "mtl") -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except Exception as e:
        raise RuntimeError("matplotlib is required for plotting. Please pip install matplotlib") from e

    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    curves_count: Dict[int, pd.DataFrame] = {}
    curves_mask: Dict[int, pd.DataFrame] = {}

    for r in records:
        mask: Optional[int] = None
        if r.zero_channel_mask is not None:
            mask = int(r.zero_channel_mask)
        elif r.zero_channel_indices:
            m = 0
            for i in r.zero_channel_indices:
                m |= 1 << int(i)
            mask = int(m)

        popcount = int(bin(mask).count("1")) if mask is not None else None

        m = _load_epoch_metrics(r.run_dir)
        if r.zero_channels is not None:
            curves_count[int(r.zero_channels)] = m
        if mask is not None:
            curves_mask[int(mask)] = m

        # Best epoch based on combined metric for MTL
        if task == "mtl":
            # Use combined score: acc + r2 (both higher is better)
            if "val/acc" in m.columns and "val/r2" in m.columns:
                combined = m["val/acc"].fillna(0) + m["val/r2"].fillna(0)
                best_epoch = int(combined.idxmax())
            elif "val/acc" in m.columns:
                best_epoch = int(m["val/acc"].idxmax())
            else:
                best_epoch = int(m.index[-1])
        else:
            best_epoch = int(m["val/acc"].idxmax()) if "val/acc" in m.columns else int(m.index[-1])
        
        best_val_acc = float(m.loc[best_epoch, "val/acc"]) if "val/acc" in m.columns else float("nan")
        best_train_acc = float(m.loc[best_epoch, "train/acc"]) if "train/acc" in m.columns else float("nan")
        best_val_r2 = float(m.loc[best_epoch, "val/r2"]) if "val/r2" in m.columns else float("nan")
        best_train_r2 = float(m.loc[best_epoch, "train/r2"]) if "train/r2" in m.columns else float("nan")
        best_val_loss = float(m.loc[best_epoch, "val/loss"]) if "val/loss" in m.columns else float("nan")
        best_val_cls_loss = float(m.loc[best_epoch, "val/cls_loss"]) if "val/cls_loss" in m.columns else float("nan")
        best_val_reg_loss = float(m.loc[best_epoch, "val/reg_loss"]) if "val/reg_loss" in m.columns else float("nan")

        rows.append(
            {
                "run_dir": str(r.run_dir),
                "zero_channels": None if r.zero_channels is None else int(r.zero_channels),
                "zero_channel_mask": None if mask is None else int(mask),
                "popcount": popcount,
                "zero_channel_indices": ",".join(map(str, r.zero_channel_indices)),
                "best_epoch": int(m.loc[best_epoch, "epoch"]) if "epoch" in m.columns else int(best_epoch),
                "best_val_acc": best_val_acc,
                "train_acc_at_best": best_train_acc,
                "best_val_r2": best_val_r2,
                "train_r2_at_best": best_train_r2,
                "best_val_loss": best_val_loss,
                "best_val_cls_loss": best_val_cls_loss,
                "best_val_reg_loss": best_val_reg_loss,
            }
        )

    if not rows:
        raise RuntimeError(f"No usable runs under {out_dir}")

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    has_mask = summary["zero_channel_mask"].notna().any()
    has_count = summary["zero_channels"].notna().any() and summary["zero_channels"].nunique(dropna=True) > 1
    has_r2 = summary["best_val_r2"].notna().any()

    # ========== Classification Plots ==========
    if has_count:
        sc = summary.dropna(subset=["zero_channels"]).copy()
        sc["zero_channels"] = sc["zero_channels"].astype(int)
        sc = sc.sort_values("zero_channels").reset_index(drop=True)
        ks = sc["zero_channels"].to_numpy()
        best_val = sc["best_val_acc"].to_numpy()

        plt.figure(figsize=(8, 4.5))
        plt.plot(ks, best_val, marker="o", color="tab:blue", label="val/acc")
        plt.xticks(ks)
        plt.xlabel("zero_channels")
        plt.ylabel("best val acc")
        plt.title(f"{title} - Classification")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "best_val_acc_vs_zero_channels.png", dpi=200)
        plt.close()

        baseline = float(sc.loc[sc["zero_channels"] == 0, "best_val_acc"].iloc[0]) if (sc["zero_channels"] == 0).any() else float("nan")
        rel = best_val - baseline if np.isfinite(baseline) else best_val

        plt.figure(figsize=(8, 4.5))
        plt.plot(ks, rel, marker="o", color="tab:blue")
        plt.axhline(0.0, color="black", linewidth=1)
        plt.xticks(ks)
        plt.xlabel("zero_channels")
        plt.ylabel("delta best val acc (vs zero=0)")
        plt.title(f"{title} - Classification Delta")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "delta_best_val_acc_vs_zero_channels.png", dpi=200)
        plt.close()

    # ========== R2 Regression Plots ==========
    if has_count and has_r2:
        sc = summary.dropna(subset=["zero_channels", "best_val_r2"]).copy()
        sc["zero_channels"] = sc["zero_channels"].astype(int)
        sc = sc.sort_values("zero_channels").reset_index(drop=True)
        ks = sc["zero_channels"].to_numpy()
        best_r2 = sc["best_val_r2"].to_numpy()

        plt.figure(figsize=(8, 4.5))
        plt.plot(ks, best_r2, marker="s", color="tab:orange", label="val/r2")
        plt.xticks(ks)
        plt.xlabel("zero_channels")
        plt.ylabel("best val R²")
        plt.title(f"{title} - Regression R²")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "best_val_r2_vs_zero_channels.png", dpi=200)
        plt.close()

        baseline_r2 = float(sc.loc[sc["zero_channels"] == 0, "best_val_r2"].iloc[0]) if (sc["zero_channels"] == 0).any() else float("nan")
        rel_r2 = best_r2 - baseline_r2 if np.isfinite(baseline_r2) else best_r2

        plt.figure(figsize=(8, 4.5))
        plt.plot(ks, rel_r2, marker="s", color="tab:orange")
        plt.axhline(0.0, color="black", linewidth=1)
        plt.xticks(ks)
        plt.xlabel("zero_channels")
        plt.ylabel("delta best val R² (vs zero=0)")
        plt.title(f"{title} - Regression R² Delta")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "delta_best_val_r2_vs_zero_channels.png", dpi=200)
        plt.close()

    # ========== Combined MTL Plot (Acc + R2) ==========
    if has_count and has_r2:
        sc = summary.dropna(subset=["zero_channels", "best_val_acc", "best_val_r2"]).copy()
        sc["zero_channels"] = sc["zero_channels"].astype(int)
        sc = sc.sort_values("zero_channels").reset_index(drop=True)
        ks = sc["zero_channels"].to_numpy()

        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        color1 = "tab:blue"
        ax1.set_xlabel("zero_channels")
        ax1.set_ylabel("Accuracy", color=color1)
        ax1.plot(ks, sc["best_val_acc"].to_numpy(), marker="o", color=color1, label="val/acc")
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_xticks(ks)
        
        ax2 = ax1.twinx()
        color2 = "tab:orange"
        ax2.set_ylabel("R²", color=color2)
        ax2.plot(ks, sc["best_val_r2"].to_numpy(), marker="s", color=color2, label="val/r2")
        ax2.tick_params(axis="y", labelcolor=color2)
        
        fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
        plt.title(f"{title} - MTL Combined (Acc & R²)")
        plt.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.savefig(out_dir / "mtl_combined_acc_r2_vs_zero_channels.png", dpi=200)
        plt.close()

    # ========== Mask-based plots ==========
    if has_mask:
        sm = summary.dropna(subset=["zero_channel_mask", "popcount"]).copy()
        sm["zero_channel_mask"] = sm["zero_channel_mask"].astype(int)
        sm["popcount"] = sm["popcount"].astype(int)

        # Accuracy vs popcount scatter
        plt.figure(figsize=(8, 4.5))
        plt.scatter(sm["popcount"].to_numpy(), sm["best_val_acc"].to_numpy(), s=12, alpha=0.7, color="tab:blue")
        plt.xticks(np.arange(0, 9))
        plt.xlabel("popcount (#zeroed channels)")
        plt.ylabel("best val acc")
        plt.title(f"{title} - Classification")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "best_val_acc_vs_popcount.png", dpi=200)
        plt.close()

        # Mean accuracy vs popcount
        by_pop = sm.groupby("popcount", as_index=False)["best_val_acc"].mean().sort_values("popcount")
        plt.figure(figsize=(8, 4.5))
        plt.plot(by_pop["popcount"], by_pop["best_val_acc"], marker="o", color="tab:blue")
        plt.xticks(np.arange(0, 9))
        plt.xlabel("popcount (#zeroed channels)")
        plt.ylabel("mean best val acc")
        plt.title(f"{title} - Mean Classification")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "mean_best_val_acc_vs_popcount.png", dpi=200)
        plt.close()

        # R2 vs popcount scatter (if available)
        if has_r2:
            sm_r2 = sm.dropna(subset=["best_val_r2"])
            if len(sm_r2) > 0:
                plt.figure(figsize=(8, 4.5))
                plt.scatter(sm_r2["popcount"].to_numpy(), sm_r2["best_val_r2"].to_numpy(), s=12, alpha=0.7, color="tab:orange")
                plt.xticks(np.arange(0, 9))
                plt.xlabel("popcount (#zeroed channels)")
                plt.ylabel("best val R²")
                plt.title(f"{title} - Regression R²")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(out_dir / "best_val_r2_vs_popcount.png", dpi=200)
                plt.close()

                # Mean R2 vs popcount
                by_pop_r2 = sm_r2.groupby("popcount", as_index=False)["best_val_r2"].mean().sort_values("popcount")
                plt.figure(figsize=(8, 4.5))
                plt.plot(by_pop_r2["popcount"], by_pop_r2["best_val_r2"], marker="s", color="tab:orange")
                plt.xticks(np.arange(0, 9))
                plt.xlabel("popcount (#zeroed channels)")
                plt.ylabel("mean best val R²")
                plt.title(f"{title} - Mean Regression R²")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(out_dir / "mean_best_val_r2_vs_popcount.png", dpi=200)
                plt.close()

        # Channel effects for classification
        effects = []
        for ch in range(8):
            in_mask = sm[sm["zero_channel_mask"].apply(lambda m: ((int(m) >> ch) & 1) == 1)]["best_val_acc"]
            out_mask = sm[sm["zero_channel_mask"].apply(lambda m: ((int(m) >> ch) & 1) == 0)]["best_val_acc"]
            a = float(in_mask.mean()) if len(in_mask) else float("nan")
            b = float(out_mask.mean()) if len(out_mask) else float("nan")
            
            # Also compute R2 effects if available
            if has_r2:
                in_mask_r2 = sm[sm["zero_channel_mask"].apply(lambda m: ((int(m) >> ch) & 1) == 1)]["best_val_r2"]
                out_mask_r2 = sm[sm["zero_channel_mask"].apply(lambda m: ((int(m) >> ch) & 1) == 0)]["best_val_r2"]
                a_r2 = float(in_mask_r2.mean()) if len(in_mask_r2) else float("nan")
                b_r2 = float(out_mask_r2.mean()) if len(out_mask_r2) else float("nan")
            else:
                a_r2, b_r2 = float("nan"), float("nan")
            
            effects.append({
                "channel": ch,
                "mean_acc_when_zeroed": a,
                "mean_acc_when_not_zeroed": b,
                "delta_acc": a - b,
                "mean_r2_when_zeroed": a_r2,
                "mean_r2_when_not_zeroed": b_r2,
                "delta_r2": a_r2 - b_r2,
            })
        eff = pd.DataFrame(effects)
        eff.to_csv(out_dir / "channel_effects.csv", index=False)

        # Channel effects bar plot - Classification
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(8)
        width = 0.35
        
        ax.bar(x - width/2, eff["delta_acc"].to_numpy(), width, label="Δ Accuracy", color="tab:blue", alpha=0.8)
        if has_r2:
            ax.bar(x + width/2, eff["delta_r2"].to_numpy(), width, label="Δ R²", color="tab:orange", alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels([f"Ch{i}" for i in range(8)])
        ax.set_xlabel("Channel")
        ax.set_ylabel("Delta (zeroed - not zeroed)")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend()
        ax.set_title(f"{title} - Channel Effects")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        plt.savefig(out_dir / "channel_effects_delta.png", dpi=200)
        plt.close()

    # ========== Loss plots ==========
    if has_count and "best_val_loss" in summary.columns:
        sc = summary.dropna(subset=["zero_channels", "best_val_loss"]).copy()
        if len(sc) > 0:
            sc["zero_channels"] = sc["zero_channels"].astype(int)
            sc = sc.sort_values("zero_channels").reset_index(drop=True)
            ks = sc["zero_channels"].to_numpy()

            plt.figure(figsize=(8, 4.5))
            plt.plot(ks, sc["best_val_loss"].to_numpy(), marker="^", color="tab:red", label="val/loss")
            plt.xticks(ks)
            plt.xlabel("zero_channels")
            plt.ylabel("best val loss")
            plt.title(f"{title} - Total Loss")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "best_val_loss_vs_zero_channels.png", dpi=200)
            plt.close()

    # ========== Scatter plot: Acc vs R2 ==========
    if has_r2:
        valid = summary.dropna(subset=["best_val_acc", "best_val_r2"])
        if len(valid) > 0:
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(
                valid["best_val_acc"].to_numpy(),
                valid["best_val_r2"].to_numpy(),
                c=valid["popcount"].to_numpy() if "popcount" in valid.columns else None,
                cmap="viridis",
                s=50,
                alpha=0.7,
            )
            if "popcount" in valid.columns:
                plt.colorbar(scatter, label="popcount (#zeroed)")
            plt.xlabel("Best Val Accuracy")
            plt.ylabel("Best Val R²")
            plt.title(f"{title} - Acc vs R² Trade-off")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "acc_vs_r2_scatter.png", dpi=200)
            plt.close()

    # ========== Training curves for selected runs ==========
    if curves_mask:
        # Plot training curves for mask=0 if available
        if 0 in curves_mask:
            _plot_training_curves(curves_mask[0], out_dir / "training_curves_mask0.png", "Training Curves (mask=0)")
        
        # Plot for a few representative masks
        for mask in [0, 85, 170, 255]:  # 0b00000000, 0b01010101, 0b10101010, 0b11111111
            if mask in curves_mask:
                _plot_training_curves(curves_mask[mask], out_dir / f"training_curves_mask{mask:03d}.png", f"Training Curves (mask={mask})")

    print(f"Plots saved to {out_dir}")


def _plot_training_curves(df: pd.DataFrame, save_path: Path, title: str) -> None:
    """Plot training curves for a single run."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Accuracy
    ax = axes[0, 0]
    if "train/acc" in df.columns:
        ax.plot(df["epoch"], df["train/acc"], label="train", alpha=0.8)
    if "val/acc" in df.columns:
        ax.plot(df["epoch"], df["val/acc"], label="val", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # R2
    ax = axes[0, 1]
    if "train/r2" in df.columns:
        ax.plot(df["epoch"], df["train/r2"], label="train", alpha=0.8)
    if "val/r2" in df.columns:
        ax.plot(df["epoch"], df["val/r2"], label="val", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R²")
    ax.set_title("R² Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Classification Loss
    ax = axes[1, 0]
    if "train/cls_loss" in df.columns:
        ax.plot(df["epoch"], df["train/cls_loss"], label="train", alpha=0.8)
    if "val/cls_loss" in df.columns:
        ax.plot(df["epoch"], df["val/cls_loss"], label="val", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cls Loss")
    ax.set_title("Classification Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Regression Loss
    ax = axes[1, 1]
    if "train/reg_loss" in df.columns:
        ax.plot(df["epoch"], df["train/reg_loss"], label="train", alpha=0.8)
    if "val/reg_loss" in df.columns:
        ax.plot(df["epoch"], df["val/reg_loss"], label="val", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reg Loss")
    ax.set_title("Regression Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="MTL Sweep for TwinGas Dataset")

    # Data args
    parser.add_argument("--root", type=str, default=str(Path.cwd() / ".cache" / "enose_uci_dataset"))
    parser.add_argument("--download", action="store_true")

    # Baseline MTL training args (from user's optimal config)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seq-len", type=int, default=1000)
    
    # MTL specific
    parser.add_argument("--task", type=str, default="mtl", choices=["cls", "reg", "mtl"])
    parser.add_argument("--reg-target", type=str, default="ppm", choices=["ppm", "log_ppm"])
    parser.add_argument("--reg-weight", type=float, default=1.0)
    parser.add_argument("--reg-loss-scale", type=float, default=250.0)
    
    # Model
    parser.add_argument("--backbone", type=str, default="tcn", choices=["tcn", "conv1d"])
    
    # LR scheduler
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=["none", "reduce_on_plateau", "cosine", "step"])
    parser.add_argument("--lr-scheduler-t-max", type=int, default=50)
    
    # Truncation
    parser.add_argument("--truncate-ratio", type=float, default=0.25)
    parser.add_argument("--truncate-start-ratio", type=float, default=0.0666)
    
    # Early stopping
    parser.add_argument("--early-stopping", action="store_true", default=True)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    
    # Validation/checkpoint
    parser.add_argument("--check-val-every-n-epoch", type=int, default=5)
    parser.add_argument("--ckpt-every-n-epochs", type=int, default=5)

    # Sweep type
    parser.add_argument("--sweep", type=str, default="mask", choices=["count", "mask", "hyperparam"])
    parser.add_argument("--zero-channel-mask", type=int, default=0, help="Fixed mask for hyperparam sweep")
    
    # Count sweep args
    parser.add_argument("--min-zero", type=int, default=0)
    parser.add_argument("--max-zero", type=int, default=8)
    
    # Mask sweep args
    parser.add_argument("--min-mask", type=int, default=0)
    parser.add_argument("--max-mask", type=int, default=255)
    
    # Hyperparam sweep args
    parser.add_argument("--sweep-lr", type=str, default="", help="Comma-separated LR values to sweep")
    parser.add_argument("--sweep-reg-weight", type=str, default="", help="Comma-separated reg_weight values")
    parser.add_argument("--sweep-seq-len", type=str, default="", help="Comma-separated seq_len values")
    parser.add_argument("--sweep-truncate-ratio", type=str, default="", help="Comma-separated truncate_ratio values")
    parser.add_argument("--sweep-batch-size", type=str, default="", help="Comma-separated batch_size values")

    # Output
    parser.add_argument(
        "--out-root",
        type=str,
        default=str(_REPO_ROOT / "runs" / "sweep_mtl"),
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="",
        help="Where to save plots. Defaults to <out-root>/plots",
    )

    # Control
    parser.add_argument("--only-plot", action="store_true")
    parser.add_argument("--runs-dir", type=str, default="", help="Existing runs root to plot")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-fail", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ensure-disk-cache", action="store_true")
    parser.add_argument("--train-ram-cache", action="store_true")
    parser.add_argument("--train-ram-cache-max-samples", type=int, default=0)
    parser.add_argument("--train-ram-cache-disk", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()
    args.root = str(Path(args.root).expanduser().resolve())

    if not args.only_plot:
        run_sweep(args)

    # Skip plotting in dry-run mode
    if args.dry_run and not args.only_plot:
        print("\n[dry-run] Skipping plotting. Use --only-plot to generate plots from existing runs.")
        return

    runs_root = Path(args.runs_dir).expanduser().resolve() if args.runs_dir else Path(args.out_root).expanduser().resolve()
    records = discover_runs(runs_root)
    
    if not records:
        print(f"No runs found under {runs_root}")
        return

    plots_dir = Path(args.plots_dir).expanduser().resolve() if args.plots_dir else (Path(args.out_root).expanduser().resolve() / "plots")

    title = f"TwinGas MTL sweep (seq_len={args.seq_len}, seed={args.seed})"
    make_plots(records, out_dir=plots_dir, title=title, task=args.task)


if __name__ == "__main__":
    main()
