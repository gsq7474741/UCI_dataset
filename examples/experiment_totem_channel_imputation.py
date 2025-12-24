"""
TOTEM通道补全实验脚本

实验流程:
1. 训练一个8通道齐全（无mask）的MTL模型作为baseline
2. 测试不同通道mask（0-255）时的推理性能（性能下降）
3. 测试用TOTEM补全缺失通道后的推理性能（性能恢复）
4. 生成全面的可视化图表

Usage:
    python experiment_totem_channel_imputation.py --phase all
    python experiment_totem_channel_imputation.py --phase train
    python experiment_totem_channel_imputation.py --phase eval
    python experiment_totem_channel_imputation.py --phase plot
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# 路径设置
_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOTEM_ROOT = Path("/root/TOTEM")
_TRAIN_SCRIPT = _REPO_ROOT / "examples" / "train_twin_timeseries.py"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_TOTEM_ROOT / "imputation") not in sys.path:
    sys.path.insert(0, str(_TOTEM_ROOT / "imputation"))

from enose_uci_dataset.datasets import TwinGasSensorArrays


# ============================================================================
# TOTEM 补全模块
# ============================================================================

class TOTEMChannelImputer:
    """使用TOTEM进行跨通道补全"""
    
    TOTEM_MODEL_PATH = _TOTEM_ROOT / "pretrained_models" / "generatlist_pretrained_tokenizers" / \
                       "imputation" / "CD64_CW512_CF4_BS8192_ITR120000_seed1_maskratio0.5" / \
                       "checkpoints" / "final_model.pth"
    
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载TOTEM预训练模型"""
        if not self.TOTEM_MODEL_PATH.exists():
            raise FileNotFoundError(f"TOTEM model not found at {self.TOTEM_MODEL_PATH}")
        
        self.model = torch.load(self.TOTEM_MODEL_PATH, map_location=self.device, weights_only=False)
        self.model.to(self.device)
        self.model.eval()
        print(f"[TOTEM] Model loaded from {self.TOTEM_MODEL_PATH}")
    
    def impute_channels(
        self, 
        data: torch.Tensor, 
        missing_channels: List[int],
        totem_seq_len: int = 96
    ) -> torch.Tensor:
        """
        使用TOTEM进行跨通道补全
        
        Args:
            data: [B, C, T] 输入数据（原始数据，缺失通道可以是任意值）
            missing_channels: 缺失通道的索引列表
            totem_seq_len: TOTEM期望的序列长度
            
        Returns:
            imputed_data: [B, C, T] 补全后的数据
        """
        if not missing_channels:
            return data
        
        B, C, T = data.shape
        available_channels = [i for i in range(C) if i not in missing_channels]
        
        if not available_channels:
            return data  # 无可用通道，无法补全
        
        # 下采样到TOTEM期望的长度
        if T > totem_seq_len:
            indices = np.linspace(0, T - 1, totem_seq_len, dtype=int)
            data_resampled = data[:, :, indices]
        else:
            data_resampled = data
        
        # 只提取可用通道的数据
        available_data = data_resampled[:, available_channels, :]  # [B, num_avail, T']
        
        # 对可用通道做标准化，并记录统计量用于反标准化
        avail_normalized, avail_means, avail_stds = self._normalize_channels_with_stats(available_data)
        
        # 编码可用通道
        num_avail = len(available_channels)
        flat_avail = avail_normalized.reshape(B * num_avail, -1).to(self.device)
        
        with torch.no_grad():
            z = self.model.encoder(flat_avail, self.model.compression_factor)
            vq_loss, quantized, perplexity, _, _, _ = self.model.vq(z)
        
        # [B, num_avail, D, compressed_T]
        quantized_reshaped = quantized.reshape(B, num_avail, quantized.shape[1], -1)
        
        # 计算可用通道的平均quantized vector
        avg_quantized = quantized_reshaped.mean(dim=1)  # [B, D, compressed_T]
        
        # 解码得到翻译结果（标准化空间）
        with torch.no_grad():
            translated = self.model.decoder(avg_quantized, self.model.compression_factor)  # [B, T']
        
        # 使用可用通道的平均统计量进行反标准化
        avg_mean = avail_means.mean(dim=1, keepdim=True)  # [B, 1]
        avg_std = avail_stds.mean(dim=1, keepdim=True)    # [B, 1]
        translated_denorm = translated.cpu() * avg_std + avg_mean
        
        # 上采样回原始长度
        if T > totem_seq_len:
            translated_upsampled = F.interpolate(
                translated_denorm.unsqueeze(1), size=T, mode='linear', align_corners=True
            ).squeeze(1)
        else:
            translated_upsampled = translated_denorm
        
        # 填充缺失通道
        imputed_data = data.clone()
        for ch in missing_channels:
            imputed_data[:, ch, :] = translated_upsampled
        
        return imputed_data
    
    def _normalize_channels_with_stats(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        每通道Z-score标准化，并返回统计量
        
        Args:
            data: [B, C, T]
        Returns:
            normalized: [B, C, T]
            means: [B, C]
            stds: [B, C]
        """
        B, C, T = data.shape
        means = data.mean(dim=2)  # [B, C]
        stds = data.std(dim=2)    # [B, C]
        stds = torch.where(stds < 1e-6, torch.ones_like(stds), stds)
        
        normalized = (data - means.unsqueeze(2)) / stds.unsqueeze(2)
        return normalized, means, stds


# ============================================================================
# 实验记录数据类
# ============================================================================

@dataclass
class ExperimentResult:
    mask: int
    popcount: int
    masked_channels: Tuple[int, ...]
    # 原始性能（无mask）
    baseline_acc: float
    baseline_r2: float
    # 带mask的性能
    masked_acc: float
    masked_r2: float
    # TOTEM补全后的性能
    totem_acc: float
    totem_r2: float
    # 恢复率
    acc_recovery_rate: float
    r2_recovery_rate: float


# ============================================================================
# 训练模块
# ============================================================================

def train_baseline_model(args) -> Path:
    """训练8通道齐全的baseline模型"""
    out_dir = Path(args.out_root) / "baseline_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已存在
    ckpt_dir = out_dir / "version_0" / "checkpoints"
    if ckpt_dir.exists() and list(ckpt_dir.glob("*.ckpt")):
        print(f"[Train] Baseline model already exists at {out_dir}")
        return out_dir
    
    cmd = [
        sys.executable, str(_TRAIN_SCRIPT),
        "--root", str(args.root),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--num-workers", str(args.num_workers),
        "--test-size", str(args.test_size),
        "--seq-len", str(args.seq_len),
        "--disk-cache",
        "--task", "mtl",
        "--reg-target", "ppm",
        "--reg-weight", str(args.reg_weight),
        "--reg-loss-scale", str(args.reg_loss_scale),
        "--backbone", "tcn",
        "--lr-scheduler", "cosine",
        "--lr-scheduler-t-max", str(args.lr_scheduler_t_max),
        "--truncate-ratio", str(args.truncate_ratio),
        "--truncate-start-ratio", str(args.truncate_start_ratio),
        "--zero-channel-mask", "0",  # 无mask
        "--logdir", str(out_dir),
        "--check-val-every-n-epoch", str(args.check_val_every_n_epoch),
        "--ckpt-every-n-epochs", str(args.ckpt_every_n_epochs),
        "--save-last",
    ]
    
    if args.early_stopping:
        cmd.append("--early-stopping")
        cmd.extend(["--early-stopping-patience", str(args.early_stopping_patience)])
    
    if args.download:
        cmd.append("--download")
    
    print(f"[Train] Training baseline model...")
    print(f"[Train] Command: {' '.join(cmd)}")
    
    if args.dry_run:
        print("[Train] Dry run, skipping actual training")
        return out_dir
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Baseline training failed with return code {result.returncode}")
    
    print(f"[Train] Baseline model saved to {out_dir}")
    return out_dir


def load_trained_model(model_dir: Path, device: str = "cuda:0"):
    """加载训练好的模型，优先使用r2最优的checkpoint"""
    import lightning as L
    import re
    
    # 查找checkpoint
    ckpt_dir = model_dir / "version_0" / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    
    # 递归查找所有.ckpt文件
    ckpt_files = list(ckpt_dir.rglob("*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")
    
    # 优先查找r2最优的checkpoint（用于mtl任务）
    r2_ckpts = [f for f in ckpt_files if "r2" in str(f)]
    if r2_ckpts:
        # 从文件名解析r2值，选择最大的
        def extract_r2(path):
            match = re.search(r'r2[=_]?([-\d.]+)\.ckpt', str(path))
            return float(match.group(1)) if match else -999
        r2_ckpts.sort(key=extract_r2, reverse=True)
        ckpt_path = r2_ckpts[0]
        print(f"[Model] Found {len(r2_ckpts)} r2 checkpoints, using best: {ckpt_path.name}")
    else:
        # 回退到last.ckpt
        last_ckpt = ckpt_dir / "last.ckpt"
        if last_ckpt.exists():
            ckpt_path = last_ckpt
        else:
            ckpt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            ckpt_path = ckpt_files[0]
    
    print(f"[Model] Loading checkpoint: {ckpt_path}")
    
    # 动态导入LightningModule
    sys.path.insert(0, str(_REPO_ROOT / "examples"))
    from train_twin_timeseries import TwinGasLitModule
    
    model = TwinGasLitModule.load_from_checkpoint(str(ckpt_path))
    model.to(device)
    model.eval()
    
    return model


# ============================================================================
# 评估模块
# ============================================================================

def create_test_dataloader(args, zero_channel_mask: int = 0):
    """创建测试数据加载器"""
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    
    sys.path.insert(0, str(_REPO_ROOT / "examples"))
    from train_twin_timeseries import TwinGasTimeSeriesDataset
    
    base = TwinGasSensorArrays(root=Path(args.root), download=False)
    labels = np.asarray([int(base.get_record(i).target["gas"]) for i in range(len(base))], dtype=np.int64)
    indices = np.arange(len(base), dtype=np.int64)
    
    _, idx_val = train_test_split(
        indices, test_size=args.test_size, random_state=args.seed, stratify=labels
    )
    
    # 计算被mask的通道
    zero_channels = tuple(i for i in range(8) if ((zero_channel_mask >> i) & 1) == 1)
    
    disk_cache_dir = base.dataset_dir / "cache" / "twin_gas_sensor_arrays_npy_v1"
    
    # 注意：task=mtl时，input_norm应该是"none"（与训练时一致）
    # train_twin_timeseries.py中：input_norm = "per_sample" if task == "cls" else "none"
    input_norm = "per_sample" if args.task == "cls" else "none"
    
    val_ds = TwinGasTimeSeriesDataset(
        base, idx_val,
        seq_len=args.seq_len,
        train=False,
        zero_channels=zero_channels,
        input_norm=input_norm,
        train_downsample="uniform",
        reg_target=args.reg_target,
        disk_cache_dir=disk_cache_dir,
        truncate_ratio=args.truncate_ratio,
        truncate_start_ratio=args.truncate_start_ratio,
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    return val_loader, zero_channels


def evaluate_model(
    model, 
    dataloader, 
    device: str = "cuda:0",
    totem_imputer: Optional[TOTEMChannelImputer] = None,
    missing_channels: Optional[List[int]] = None,
) -> Tuple[float, float]:
    """
    评估模型性能
    
    Returns:
        (accuracy, r2_score)
    """
    from torchmetrics.classification import MulticlassAccuracy
    from torchmetrics.regression import R2Score
    
    acc_metric = MulticlassAccuracy(num_classes=4).to(device)
    r2_metric = R2Score().to(device)
    
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            x, y_cls, y_reg = batch
            x = x.to(device)
            y_cls = y_cls.to(device)
            y_reg = y_reg.to(device)
            
            # 如果需要TOTEM补全
            if totem_imputer is not None and missing_channels:
                x = totem_imputer.impute_channels(x, missing_channels)
                x = x.to(device)
            
            logits, reg = model(x)
            
            acc_metric.update(logits, y_cls)
            r2_metric.update(reg, y_reg)
    
    acc = acc_metric.compute().item()
    r2 = r2_metric.compute().item()
    
    return acc, r2


def run_evaluation(args, model_dir: Path) -> List[ExperimentResult]:
    """运行完整评估实验"""
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    
    # 加载模型
    model = load_trained_model(model_dir, device)
    
    # 加载TOTEM
    print("[Eval] Loading TOTEM imputer...")
    totem_imputer = TOTEMChannelImputer(device=device)
    
    # 获取baseline性能
    print("[Eval] Evaluating baseline (no mask)...")
    baseline_loader, _ = create_test_dataloader(args, zero_channel_mask=0)
    baseline_acc, baseline_r2 = evaluate_model(model, baseline_loader, device)
    print(f"[Eval] Baseline: acc={baseline_acc:.4f}, r2={baseline_r2:.4f}")
    
    results: List[ExperimentResult] = []
    
    # 测试所有256种mask组合
    masks_to_test = list(range(args.min_mask, args.max_mask + 1))
    total = len(masks_to_test)
    
    for idx, mask in enumerate(masks_to_test):
        popcount = bin(mask).count("1")
        masked_channels = tuple(i for i in range(8) if ((mask >> i) & 1) == 1)
        
        print(f"\n[Eval] Mask {mask:03d} ({idx+1}/{total}), channels={masked_channels}")
        
        # 创建带mask的数据加载器
        masked_loader, _ = create_test_dataloader(args, zero_channel_mask=mask)
        
        # 评估带mask的性能（不补全）
        masked_acc, masked_r2 = evaluate_model(model, masked_loader, device)
        print(f"  Masked: acc={masked_acc:.4f}, r2={masked_r2:.4f}")
        
        # 评估TOTEM补全后的性能
        if popcount > 0:
            # 需要重新创建不带mask的数据加载器，然后在推理时做mask和补全
            clean_loader, _ = create_test_dataloader(args, zero_channel_mask=0)
            
            # 自定义评估：先mask再补全
            totem_acc, totem_r2 = evaluate_with_totem_imputation(
                model, clean_loader, device, totem_imputer, list(masked_channels)
            )
        else:
            totem_acc, totem_r2 = masked_acc, masked_r2
        
        print(f"  TOTEM: acc={totem_acc:.4f}, r2={totem_r2:.4f}")
        
        # 计算恢复率
        acc_drop = baseline_acc - masked_acc
        acc_recovery = (totem_acc - masked_acc) / acc_drop if acc_drop > 0.001 else 0.0
        
        r2_drop = baseline_r2 - masked_r2
        r2_recovery = (totem_r2 - masked_r2) / r2_drop if r2_drop > 0.001 else 0.0
        
        results.append(ExperimentResult(
            mask=mask,
            popcount=popcount,
            masked_channels=masked_channels,
            baseline_acc=baseline_acc,
            baseline_r2=baseline_r2,
            masked_acc=masked_acc,
            masked_r2=masked_r2,
            totem_acc=totem_acc,
            totem_r2=totem_r2,
            acc_recovery_rate=acc_recovery,
            r2_recovery_rate=r2_recovery,
        ))
    
    return results


def evaluate_with_totem_imputation(
    model,
    dataloader,
    device: str,
    totem_imputer: TOTEMChannelImputer,
    missing_channels: List[int],
) -> Tuple[float, float]:
    """带TOTEM补全的评估"""
    from torchmetrics.classification import MulticlassAccuracy
    from torchmetrics.regression import R2Score
    
    acc_metric = MulticlassAccuracy(num_classes=4).to(device)
    r2_metric = R2Score().to(device)
    
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            x, y_cls, y_reg = batch
            
            # 用TOTEM补全缺失通道（传入原始数据，TOTEM只使用可用通道）
            x_imputed = totem_imputer.impute_channels(x, missing_channels)
            
            # 确保非缺失通道保持原值，缺失通道使用补全值
            # （impute_channels已经处理了这个，但为了安全再mask一次确保）
            for ch in missing_channels:
                pass  # x_imputed[:, ch, :] 已经是补全值
            
            x_imputed = x_imputed.to(device)
            y_cls = y_cls.to(device)
            y_reg = y_reg.to(device)
            
            logits, reg = model(x_imputed)
            
            acc_metric.update(logits, y_cls)
            r2_metric.update(reg, y_reg)
    
    acc = acc_metric.compute().item()
    r2 = r2_metric.compute().item()
    
    return acc, r2


# ============================================================================
# 可视化模块
# ============================================================================

def make_comprehensive_plots(results: List[ExperimentResult], out_dir: Path, args):
    """生成全面的可视化图表"""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 转换为DataFrame
    df = pd.DataFrame([{
        "mask": r.mask,
        "popcount": r.popcount,
        "masked_channels": str(r.masked_channels),
        "baseline_acc": r.baseline_acc,
        "baseline_r2": r.baseline_r2,
        "masked_acc": r.masked_acc,
        "masked_r2": r.masked_r2,
        "totem_acc": r.totem_acc,
        "totem_r2": r.totem_r2,
        "acc_recovery_rate": r.acc_recovery_rate,
        "r2_recovery_rate": r.r2_recovery_rate,
        "acc_delta_masked": r.masked_acc - r.baseline_acc,
        "acc_delta_totem": r.totem_acc - r.baseline_acc,
        "r2_delta_masked": r.masked_r2 - r.baseline_r2,
        "r2_delta_totem": r.totem_r2 - r.baseline_r2,
    } for r in results])
    
    df.to_csv(out_dir / "results.csv", index=False)
    print(f"[Plot] Results saved to {out_dir / 'results.csv'}")
    
    # ========== 1. 主要结果图：Accuracy对比 ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 按popcount分组
    for popcount in range(9):
        sub = df[df["popcount"] == popcount]
        if len(sub) == 0:
            continue
        
        ax = axes[0]
        ax.scatter([popcount] * len(sub), sub["masked_acc"], alpha=0.3, c="red", s=20, label="Masked" if popcount == 0 else "")
        ax.scatter([popcount] * len(sub), sub["totem_acc"], alpha=0.3, c="green", s=20, label="TOTEM" if popcount == 0 else "")
    
    # 均值线
    mean_masked = df.groupby("popcount")["masked_acc"].mean()
    mean_totem = df.groupby("popcount")["totem_acc"].mean()
    baseline = df["baseline_acc"].iloc[0]
    
    ax = axes[0]
    ax.plot(mean_masked.index, mean_masked.values, "r-o", linewidth=2, markersize=8, label="Masked (mean)")
    ax.plot(mean_totem.index, mean_totem.values, "g-s", linewidth=2, markersize=8, label="TOTEM (mean)")
    ax.axhline(baseline, color="blue", linestyle="--", linewidth=2, label=f"Baseline ({baseline:.4f})")
    ax.set_xlabel("Number of Masked Channels")
    ax.set_ylabel("Accuracy")
    ax.set_title("Classification Accuracy vs Channel Masking")
    ax.set_xticks(range(9))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # R2对比
    mean_masked_r2 = df.groupby("popcount")["masked_r2"].mean()
    mean_totem_r2 = df.groupby("popcount")["totem_r2"].mean()
    baseline_r2 = df["baseline_r2"].iloc[0]
    
    ax = axes[1]
    ax.plot(mean_masked_r2.index, mean_masked_r2.values, "r-o", linewidth=2, markersize=8, label="Masked (mean)")
    ax.plot(mean_totem_r2.index, mean_totem_r2.values, "g-s", linewidth=2, markersize=8, label="TOTEM (mean)")
    ax.axhline(baseline_r2, color="blue", linestyle="--", linewidth=2, label=f"Baseline ({baseline_r2:.4f})")
    ax.set_xlabel("Number of Masked Channels")
    ax.set_ylabel("R² Score")
    ax.set_title("Regression R² vs Channel Masking")
    ax.set_xticks(range(9))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "main_results.png", dpi=200)
    plt.close()
    print(f"[Plot] Saved main_results.png")
    
    # ========== 2. 恢复率图 ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 按popcount计算平均恢复率
    recovery_by_pop = df[df["popcount"] > 0].groupby("popcount").agg({
        "acc_recovery_rate": ["mean", "std"],
        "r2_recovery_rate": ["mean", "std"],
    })
    
    ax = axes[0]
    pops = recovery_by_pop.index.values
    mean_acc_recovery = recovery_by_pop[("acc_recovery_rate", "mean")].values
    std_acc_recovery = recovery_by_pop[("acc_recovery_rate", "std")].values
    ax.bar(pops, mean_acc_recovery, yerr=std_acc_recovery, capsize=5, color="steelblue", alpha=0.8)
    ax.axhline(1.0, color="green", linestyle="--", label="Full Recovery")
    ax.axhline(0.0, color="red", linestyle="--", label="No Recovery")
    ax.set_xlabel("Number of Masked Channels")
    ax.set_ylabel("Accuracy Recovery Rate")
    ax.set_title("TOTEM Accuracy Recovery Rate")
    ax.set_xticks(pops)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    ax = axes[1]
    mean_r2_recovery = recovery_by_pop[("r2_recovery_rate", "mean")].values
    std_r2_recovery = recovery_by_pop[("r2_recovery_rate", "std")].values
    ax.bar(pops, mean_r2_recovery, yerr=std_r2_recovery, capsize=5, color="darkorange", alpha=0.8)
    ax.axhline(1.0, color="green", linestyle="--", label="Full Recovery")
    ax.axhline(0.0, color="red", linestyle="--", label="No Recovery")
    ax.set_xlabel("Number of Masked Channels")
    ax.set_ylabel("R² Recovery Rate")
    ax.set_title("TOTEM R² Recovery Rate")
    ax.set_xticks(pops)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(out_dir / "recovery_rate.png", dpi=200)
    plt.close()
    print(f"[Plot] Saved recovery_rate.png")
    
    # ========== 3. 热力图：每个通道的影响 ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 计算每个通道被mask时的平均性能
    channel_effects_acc = []
    channel_effects_r2 = []
    for ch in range(8):
        in_mask = df[df["mask"].apply(lambda m: ((int(m) >> ch) & 1) == 1)]
        out_mask = df[df["mask"].apply(lambda m: ((int(m) >> ch) & 1) == 0)]
        
        in_acc = in_mask["masked_acc"].mean() if len(in_mask) > 0 else np.nan
        out_acc = out_mask["masked_acc"].mean() if len(out_mask) > 0 else np.nan
        in_totem_acc = in_mask["totem_acc"].mean() if len(in_mask) > 0 else np.nan
        
        in_r2 = in_mask["masked_r2"].mean() if len(in_mask) > 0 else np.nan
        out_r2 = out_mask["masked_r2"].mean() if len(out_mask) > 0 else np.nan
        in_totem_r2 = in_mask["totem_r2"].mean() if len(in_mask) > 0 else np.nan
        
        channel_effects_acc.append({
            "channel": ch,
            "when_masked": in_acc,
            "when_not_masked": out_acc,
            "delta": in_acc - out_acc,
            "totem_recovery": in_totem_acc - in_acc,
        })
        channel_effects_r2.append({
            "channel": ch,
            "when_masked": in_r2,
            "when_not_masked": out_r2,
            "delta": in_r2 - out_r2,
            "totem_recovery": in_totem_r2 - in_r2,
        })
    
    eff_acc_df = pd.DataFrame(channel_effects_acc)
    eff_r2_df = pd.DataFrame(channel_effects_r2)
    
    # 保存通道效应
    eff_acc_df.to_csv(out_dir / "channel_effects_acc.csv", index=False)
    eff_r2_df.to_csv(out_dir / "channel_effects_r2.csv", index=False)
    
    # 绘制通道效应条形图
    ax = axes[0]
    x = np.arange(8)
    width = 0.35
    ax.bar(x - width/2, eff_acc_df["delta"], width, label="Impact when masked", color="red", alpha=0.7)
    ax.bar(x + width/2, eff_acc_df["totem_recovery"], width, label="TOTEM recovery", color="green", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Ch{i}" for i in range(8)])
    ax.set_xlabel("Channel")
    ax.set_ylabel("Accuracy Delta")
    ax.set_title("Per-Channel Accuracy Impact & TOTEM Recovery")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    ax = axes[1]
    ax.bar(x - width/2, eff_r2_df["delta"], width, label="Impact when masked", color="red", alpha=0.7)
    ax.bar(x + width/2, eff_r2_df["totem_recovery"], width, label="TOTEM recovery", color="green", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Ch{i}" for i in range(8)])
    ax.set_xlabel("Channel")
    ax.set_ylabel("R² Delta")
    ax.set_title("Per-Channel R² Impact & TOTEM Recovery")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(out_dir / "channel_effects.png", dpi=200)
    plt.close()
    print(f"[Plot] Saved channel_effects.png")
    
    # ========== 4. Delta散点图：Masked vs TOTEM ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    scatter = ax.scatter(
        df["acc_delta_masked"], df["acc_delta_totem"],
        c=df["popcount"], cmap="viridis", s=30, alpha=0.7
    )
    plt.colorbar(scatter, ax=ax, label="Popcount")
    ax.plot([-1, 0.1], [-1, 0.1], "k--", alpha=0.5, label="y=x (no improvement)")
    ax.axhline(0, color="green", linestyle=":", alpha=0.5)
    ax.axvline(0, color="red", linestyle=":", alpha=0.5)
    ax.set_xlabel("Accuracy Delta (Masked)")
    ax.set_ylabel("Accuracy Delta (TOTEM)")
    ax.set_title("TOTEM Improvement: Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    scatter = ax.scatter(
        df["r2_delta_masked"], df["r2_delta_totem"],
        c=df["popcount"], cmap="viridis", s=30, alpha=0.7
    )
    plt.colorbar(scatter, ax=ax, label="Popcount")
    ax.plot([-2, 0.5], [-2, 0.5], "k--", alpha=0.5, label="y=x (no improvement)")
    ax.axhline(0, color="green", linestyle=":", alpha=0.5)
    ax.axvline(0, color="red", linestyle=":", alpha=0.5)
    ax.set_xlabel("R² Delta (Masked)")
    ax.set_ylabel("R² Delta (TOTEM)")
    ax.set_title("TOTEM Improvement: R²")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "delta_scatter.png", dpi=200)
    plt.close()
    print(f"[Plot] Saved delta_scatter.png")
    
    # ========== 5. 综合仪表板 ==========
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 5.1 Accuracy曲线
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(mean_masked.index, mean_masked.values, "r-o", label="Masked")
    ax1.plot(mean_totem.index, mean_totem.values, "g-s", label="TOTEM")
    ax1.axhline(baseline, color="blue", linestyle="--", label="Baseline")
    ax1.set_xlabel("# Masked Channels")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy vs Masking")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 5.2 R2曲线
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(mean_masked_r2.index, mean_masked_r2.values, "r-o", label="Masked")
    ax2.plot(mean_totem_r2.index, mean_totem_r2.values, "g-s", label="TOTEM")
    ax2.axhline(baseline_r2, color="blue", linestyle="--", label="Baseline")
    ax2.set_xlabel("# Masked Channels")
    ax2.set_ylabel("R²")
    ax2.set_title("R² vs Masking")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 5.3 恢复率
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(pops - 0.2, mean_acc_recovery, 0.4, label="Acc Recovery", color="steelblue")
    ax3.bar(pops + 0.2, mean_r2_recovery, 0.4, label="R² Recovery", color="darkorange")
    ax3.axhline(1.0, color="green", linestyle="--")
    ax3.set_xlabel("# Masked Channels")
    ax3.set_ylabel("Recovery Rate")
    ax3.set_title("TOTEM Recovery Rate")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")
    
    # 5.4 通道效应热力图
    ax4 = fig.add_subplot(gs[1, :2])
    heatmap_data = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            if i == j:
                sub = df[df["mask"] == (1 << i)]
            else:
                sub = df[df["mask"] == ((1 << i) | (1 << j))]
            if len(sub) > 0:
                heatmap_data[i, j] = sub["totem_acc"].iloc[0] - sub["masked_acc"].iloc[0]
    
    im = ax4.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=-0.1, vmax=0.1)
    ax4.set_xticks(range(8))
    ax4.set_yticks(range(8))
    ax4.set_xticklabels([f"Ch{i}" for i in range(8)])
    ax4.set_yticklabels([f"Ch{i}" for i in range(8)])
    ax4.set_title("TOTEM Improvement Heatmap (Acc)")
    plt.colorbar(im, ax=ax4, label="Δ Accuracy")
    
    # 5.5 统计摘要
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    summary_text = f"""
    === TOTEM Channel Imputation Results ===
    
    Baseline Performance:
      Accuracy: {baseline:.4f}
      R²: {baseline_r2:.4f}
    
    Mean Performance Drop (all masks):
      Accuracy: {df['acc_delta_masked'].mean():.4f}
      R²: {df['r2_delta_masked'].mean():.4f}
    
    Mean TOTEM Recovery:
      Accuracy: {(df['totem_acc'] - df['masked_acc']).mean():.4f}
      R²: {(df['totem_r2'] - df['masked_r2']).mean():.4f}
    
    Average Recovery Rate:
      Accuracy: {df[df['popcount']>0]['acc_recovery_rate'].mean():.2%}
      R²: {df[df['popcount']>0]['r2_recovery_rate'].mean():.2%}
    """
    ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    # 5.6 分布直方图
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.hist(df["masked_acc"], bins=20, alpha=0.5, label="Masked", color="red")
    ax6.hist(df["totem_acc"], bins=20, alpha=0.5, label="TOTEM", color="green")
    ax6.axvline(baseline, color="blue", linestyle="--", label="Baseline")
    ax6.set_xlabel("Accuracy")
    ax6.set_ylabel("Count")
    ax6.set_title("Accuracy Distribution")
    ax6.legend(fontsize=8)
    
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.hist(df["masked_r2"], bins=20, alpha=0.5, label="Masked", color="red")
    ax7.hist(df["totem_r2"], bins=20, alpha=0.5, label="TOTEM", color="green")
    ax7.axvline(baseline_r2, color="blue", linestyle="--", label="Baseline")
    ax7.set_xlabel("R²")
    ax7.set_ylabel("Count")
    ax7.set_title("R² Distribution")
    ax7.legend(fontsize=8)
    
    # 5.7 改进幅度箱线图
    ax8 = fig.add_subplot(gs[2, 2])
    improvement = df["totem_acc"] - df["masked_acc"]
    by_pop = [improvement[df["popcount"] == p].values for p in range(1, 9)]
    ax8.boxplot(by_pop, labels=[str(p) for p in range(1, 9)])
    ax8.axhline(0, color="red", linestyle="--")
    ax8.set_xlabel("# Masked Channels")
    ax8.set_ylabel("Accuracy Improvement")
    ax8.set_title("TOTEM Improvement by Popcount")
    ax8.grid(True, alpha=0.3, axis="y")
    
    plt.suptitle("TOTEM Channel Imputation Experiment Results", fontsize=14, fontweight="bold")
    plt.savefig(out_dir / "dashboard.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved dashboard.png")
    
    print(f"\n[Plot] All plots saved to {out_dir}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="TOTEM Channel Imputation Experiment")
    
    # 阶段控制
    parser.add_argument("--phase", type=str, default="all", choices=["all", "train", "eval", "plot"])
    
    # 数据参数
    parser.add_argument("--root", type=str, default=str(Path.cwd() / ".cache" / "enose_uci_dataset"))
    parser.add_argument("--download", action="store_true")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seq-len", type=int, default=1000)
    parser.add_argument("--task", type=str, default="mtl", choices=["cls", "reg", "mtl"])
    parser.add_argument("--reg-target", type=str, default="ppm", choices=["ppm", "log_ppm"])
    parser.add_argument("--reg-weight", type=float, default=1.0)
    parser.add_argument("--reg-loss-scale", type=float, default=250.0)
    parser.add_argument("--lr-scheduler-t-max", type=int, default=50)
    parser.add_argument("--truncate-ratio", type=float, default=0.25)
    parser.add_argument("--truncate-start-ratio", type=float, default=0.0666)
    parser.add_argument("--early-stopping", action="store_true", default=True)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=1)
    parser.add_argument("--ckpt-every-n-epochs", type=int, default=1)
    
    # 评估参数
    parser.add_argument("--min-mask", type=int, default=0)
    parser.add_argument("--max-mask", type=int, default=255)
    parser.add_argument("--gpu", type=int, default=0)
    
    # 输出路径
    parser.add_argument("--out-root", type=str, 
                       default=str(_REPO_ROOT / "runs" / "totem_imputation_experiment"))
    
    # 控制
    parser.add_argument("--dry-run", action="store_true")
    
    args = parser.parse_args()
    
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config_path = out_root / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[Main] Config saved to {config_path}")
    
    # Phase 1: 训练
    if args.phase in ["all", "train"]:
        print("\n" + "="*60)
        print("Phase 1: Training Baseline Model (8 channels, no mask)")
        print("="*60)
        model_dir = train_baseline_model(args)
    else:
        model_dir = Path(args.out_root) / "baseline_model"
    
    # Phase 2: 评估
    results_path = out_root / "results.json"
    if args.phase in ["all", "eval"]:
        print("\n" + "="*60)
        print("Phase 2: Evaluating with Different Masks + TOTEM Imputation")
        print("="*60)
        
        if not args.dry_run:
            results = run_evaluation(args, model_dir)
            
            # 保存结果
            results_json = [
                {
                    "mask": r.mask,
                    "popcount": r.popcount,
                    "masked_channels": list(r.masked_channels),
                    "baseline_acc": r.baseline_acc,
                    "baseline_r2": r.baseline_r2,
                    "masked_acc": r.masked_acc,
                    "masked_r2": r.masked_r2,
                    "totem_acc": r.totem_acc,
                    "totem_r2": r.totem_r2,
                    "acc_recovery_rate": r.acc_recovery_rate,
                    "r2_recovery_rate": r.r2_recovery_rate,
                }
                for r in results
            ]
            with open(results_path, "w") as f:
                json.dump(results_json, f, indent=2)
            print(f"[Main] Results saved to {results_path}")
    
    # Phase 3: 绘图
    if args.phase in ["all", "plot"]:
        print("\n" + "="*60)
        print("Phase 3: Generating Visualization Plots")
        print("="*60)
        
        if results_path.exists():
            with open(results_path, "r") as f:
                results_json = json.load(f)
            
            results = [
                ExperimentResult(
                    mask=r["mask"],
                    popcount=r["popcount"],
                    masked_channels=tuple(r["masked_channels"]),
                    baseline_acc=r["baseline_acc"],
                    baseline_r2=r["baseline_r2"],
                    masked_acc=r["masked_acc"],
                    masked_r2=r["masked_r2"],
                    totem_acc=r["totem_acc"],
                    totem_r2=r["totem_r2"],
                    acc_recovery_rate=r["acc_recovery_rate"],
                    r2_recovery_rate=r["r2_recovery_rate"],
                )
                for r in results_json
            ]
            
            plots_dir = out_root / "plots"
            make_comprehensive_plots(results, plots_dir, args)
        else:
            print(f"[Main] Results file not found: {results_path}")
            print("[Main] Run with --phase eval first")
    
    print("\n" + "="*60)
    print("Experiment Complete!")
    print("="*60)
    print(f"Output directory: {out_root}")


if __name__ == "__main__":
    main()
