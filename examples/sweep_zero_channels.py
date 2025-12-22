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

    metric_cols = [c for c in ["train/acc", "val/acc", "train/loss", "val/loss"] if c in df.columns]

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

    def _append_train_cache_flags(cmd: List[str]) -> None:
        if args.train_ram_cache:
            cmd.append("--ram-cache")
            cmd.extend(["--ram-cache-max-samples", str(int(args.train_ram_cache_max_samples))])
            if bool(args.train_ram_cache_disk):
                cmd.append("--ram-cache-disk")
            else:
                cmd.append("--no-ram-cache-disk")

        cmd.extend(["--task", str(args.task)])
        cmd.extend(["--reg-target", str(args.reg_target)])
        cmd.extend(["--reg-weight", str(float(args.reg_weight))])
        cmd.extend(["--reg-loss-scale", str(float(args.reg_loss_scale))])

        cmd.extend(["--check-val-every-n-epoch", str(int(args.check_val_every_n_epoch))])
        cmd.extend(["--ckpt-every-n-epochs", str(int(args.ckpt_every_n_epochs))])

    if args.sweep == "count":
        for k in range(int(args.min_zero), int(args.max_zero) + 1):
            group_dir = out_root / f"zero_{k}"
            group_dir.mkdir(parents=True, exist_ok=True)
            if (not args.overwrite) and _has_metrics(group_dir):
                latest = _find_latest_version_dir(group_dir)
                if latest is not None:
                    run_dirs.append(latest)
                continue

            cmd = [
                sys.executable,
                str(_TRAIN_SCRIPT),
                "--root",
                str(args.root),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--seed",
                str(args.seed),
                "--num-workers",
                str(args.num_workers),
                "--test-size",
                str(args.test_size),
                "--seq-len",
                str(args.seq_len),
                "--zero-channels",
                str(k),
                "--disk-cache",
                "--logdir",
                str(group_dir),
            ]
            _append_train_cache_flags(cmd)
            if args.download:
                cmd.append("--download")

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

            cmd = [
                sys.executable,
                str(_TRAIN_SCRIPT),
                "--root",
                str(args.root),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--seed",
                str(args.seed),
                "--num-workers",
                str(args.num_workers),
                "--test-size",
                str(args.test_size),
                "--seq-len",
                str(args.seq_len),
                "--zero-channel-mask",
                str(mask),
                "--disk-cache",
                "--logdir",
                str(group_dir),
            ]
            _append_train_cache_flags(cmd)
            if args.download:
                cmd.append("--download")

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

    return run_dirs


def make_plots(records: List[RunRecord], *, out_dir: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
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

        best_epoch = int(m["val/acc"].idxmax()) if "val/acc" in m.columns else int(m.index[-1])
        best_val = float(m.loc[best_epoch, "val/acc"]) if "val/acc" in m.columns else float("nan")
        best_train = float(m.loc[best_epoch, "train/acc"]) if "train/acc" in m.columns else float("nan")

        rows.append(
            {
                "run_dir": str(r.run_dir),
                "zero_channels": None if r.zero_channels is None else int(r.zero_channels),
                "zero_channel_mask": None if mask is None else int(mask),
                "popcount": popcount,
                "zero_channel_indices": ",".join(map(str, r.zero_channel_indices)),
                "best_epoch": int(m.loc[best_epoch, "epoch"]) if "epoch" in m.columns else int(best_epoch),
                "best_val_acc": best_val,
                "train_acc_at_best": best_train,
            }
        )

    if not rows:
        raise RuntimeError(f"No usable runs under {out_dir}")

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    has_mask = summary["zero_channel_mask"].notna().any()
    has_count = summary["zero_channels"].notna().any() and summary["zero_channels"].nunique(dropna=True) > 1

    if has_count:
        sc = summary.dropna(subset=["zero_channels"]).copy()
        sc["zero_channels"] = sc["zero_channels"].astype(int)
        sc = sc.sort_values("zero_channels").reset_index(drop=True)
        ks = sc["zero_channels"].to_numpy()
        best_val = sc["best_val_acc"].to_numpy()

        plt.figure(figsize=(8, 4.5))
        plt.plot(ks, best_val, marker="o")
        plt.xticks(ks)
        plt.xlabel("zero_channels")
        plt.ylabel("best val acc")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "best_val_acc_vs_zero_channels.png", dpi=200)
        plt.close()

        baseline = float(sc.loc[sc["zero_channels"] == 0, "best_val_acc"].iloc[0]) if (sc["zero_channels"] == 0).any() else float("nan")
        rel = best_val - baseline if np.isfinite(baseline) else best_val

        plt.figure(figsize=(8, 4.5))
        plt.plot(ks, rel, marker="o")
        plt.axhline(0.0, color="black", linewidth=1)
        plt.xticks(ks)
        plt.xlabel("zero_channels")
        plt.ylabel("delta best val acc (vs zero=0)")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "delta_best_val_acc_vs_zero_channels.png", dpi=200)
        plt.close()

    if has_mask:
        sm = summary.dropna(subset=["zero_channel_mask", "popcount"]).copy()
        sm["zero_channel_mask"] = sm["zero_channel_mask"].astype(int)
        sm["popcount"] = sm["popcount"].astype(int)

        plt.figure(figsize=(8, 4.5))
        plt.scatter(sm["popcount"].to_numpy(), sm["best_val_acc"].to_numpy(), s=12, alpha=0.7)
        plt.xticks(np.arange(0, 9))
        plt.xlabel("popcount (#zeroed channels)")
        plt.ylabel("best val acc")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "best_val_acc_vs_popcount.png", dpi=200)
        plt.close()

        by_pop = sm.groupby("popcount", as_index=False)["best_val_acc"].mean().sort_values("popcount")
        plt.figure(figsize=(8, 4.5))
        plt.plot(by_pop["popcount"], by_pop["best_val_acc"], marker="o")
        plt.xticks(np.arange(0, 9))
        plt.xlabel("popcount (#zeroed channels)")
        plt.ylabel("mean best val acc")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "mean_best_val_acc_vs_popcount.png", dpi=200)
        plt.close()

        effects = []
        for ch in range(8):
            in_mask = sm[sm["zero_channel_mask"].apply(lambda m: ((int(m) >> ch) & 1) == 1)]["best_val_acc"]
            out_mask = sm[sm["zero_channel_mask"].apply(lambda m: ((int(m) >> ch) & 1) == 0)]["best_val_acc"]
            a = float(in_mask.mean()) if len(in_mask) else float("nan")
            b = float(out_mask.mean()) if len(out_mask) else float("nan")
            effects.append({"channel": ch, "mean_when_zeroed": a, "mean_when_not_zeroed": b, "delta": a - b})
        eff = pd.DataFrame(effects)
        eff.to_csv(out_dir / "channel_effects.csv", index=False)

        plt.figure(figsize=(8, 4.5))
        plt.bar(eff["channel"].to_numpy(), eff["delta"].to_numpy())
        plt.xticks(np.arange(0, 8))
        plt.xlabel("channel")
        plt.ylabel("mean(best_val_acc | zeroed) - mean(best_val_acc | not_zeroed)")
        plt.title(title)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "channel_effects_delta.png", dpi=200)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default=str(Path.cwd() / ".cache" / "enose_uci_dataset"))
    parser.add_argument("--download", action="store_true")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seq-len", type=int, default=2000)

    parser.add_argument("--task", type=str, default="cls", choices=["cls", "reg", "mtl"])
    parser.add_argument("--reg-target", type=str, default="ppm", choices=["ppm", "log_ppm"])
    parser.add_argument("--reg-weight", type=float, default=1.0)
    parser.add_argument("--reg-loss-scale", type=float, default=250.0)

    parser.add_argument("--sweep", type=str, default="count", choices=["count", "mask"])

    parser.add_argument("--min-zero", type=int, default=0)
    parser.add_argument("--max-zero", type=int, default=8)

    parser.add_argument("--min-mask", type=int, default=0)
    parser.add_argument("--max-mask", type=int, default=255)

    parser.add_argument(
        "--out-root",
        type=str,
        default=str(_REPO_ROOT / "runs" / "sweep_zero_channels"),
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="",
        help="Where to save plots. Defaults to <out-root>/plots",
    )

    parser.add_argument("--only-plot", action="store_true")
    parser.add_argument("--runs-dir", type=str, default="", help="Existing runs root to plot (will rglob metrics.csv)")

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-fail", action="store_true")

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ensure-disk-cache", action="store_true")
    parser.add_argument("--train-ram-cache", action="store_true")
    parser.add_argument("--train-ram-cache-max-samples", type=int, default=0)
    parser.add_argument("--train-ram-cache-disk", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--check-val-every-n-epoch", type=int, default=1)
    parser.add_argument("--ckpt-every-n-epochs", type=int, default=1)

    args = parser.parse_args()

    args.root = str(Path(args.root).expanduser().resolve())

    if not args.only_plot:
        run_sweep(args)

    runs_root = Path(args.runs_dir).expanduser().resolve() if args.runs_dir else Path(args.out_root).expanduser().resolve()
    records = discover_runs(runs_root)

    plots_dir = Path(args.plots_dir).expanduser().resolve() if args.plots_dir else (Path(args.out_root).expanduser().resolve() / "plots")

    title = f"TwinGas Conv1D sweep (seq_len={args.seq_len}, seed={args.seed})"
    make_plots(records, out_dir=plots_dir, title=title)


if __name__ == "__main__":
    main()
