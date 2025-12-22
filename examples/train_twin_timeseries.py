from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.regression import R2Score

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from enose_uci_dataset.datasets import TwinGasSensorArrays


class TwinGasRamCache:
    def __init__(self, *, cache_dir: Optional[Path] = None, save_to_disk: bool = True):
        self._data_by_base_index: Dict[int, np.ndarray] = {}
        self.cache_dir = cache_dir
        self.save_to_disk = bool(save_to_disk)
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_file(self, sample_id: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{sample_id}.npy"

    def preload(self, base: TwinGasSensorArrays, indices: np.ndarray) -> None:
        idx = np.asarray(indices, dtype=np.int64)
        uniq = np.unique(idx)
        n = int(uniq.shape[0])
        if n == 0:
            return

        for j, base_i in enumerate(uniq.tolist()):
            rec = base.get_record(int(base_i))
            cache_file = self._cache_file(rec.sample_id)
            if cache_file is not None and cache_file.exists():
                x = np.load(cache_file, allow_pickle=False)
            else:
                x = np.loadtxt(rec.path, dtype=np.float32, usecols=range(1, 9))
                if cache_file is not None and self.save_to_disk:
                    tmp = cache_file.with_suffix(".tmp.npy")
                    try:
                        np.save(tmp, x)
                        tmp.replace(cache_file)
                    finally:
                        if tmp.exists():
                            tmp.unlink()
            self._data_by_base_index[int(base_i)] = x
            if (j + 1) % 50 == 0 or (j + 1) == n:
                print(f"RAM cache preload: {j + 1}/{n}")

    def get(self, base_index: int) -> np.ndarray:
        return self._data_by_base_index[int(base_index)]


class TwinGasTimeSeriesDataset(Dataset):
    def __init__(
        self,
        base: TwinGasSensorArrays,
        indices: np.ndarray,
        *,
        seq_len: int,
        train: bool,
        zero_channels: Optional[Tuple[int, ...]] = None,
        input_norm: str = "per_sample",
        train_downsample: str = "random",
        reg_target: str = "ppm",
        ram_cache: Optional[TwinGasRamCache] = None,
        disk_cache_dir: Optional[Path] = None,
        truncate_ratio: float = 1.0,
        truncate_start_ratio: float = 0.0,
    ):
        self.base = base
        self.indices = indices.astype(np.int64)
        self.seq_len = int(seq_len)
        self.train = bool(train)
        self.zero_channels = tuple(int(i) for i in (zero_channels or ()))
        self.input_norm = str(input_norm)
        self.train_downsample = str(train_downsample)
        self.reg_target = str(reg_target)
        self.ram_cache = ram_cache
        self.disk_cache_dir = disk_cache_dir
        self.truncate_ratio = float(truncate_ratio)
        self.truncate_start_ratio = float(truncate_start_ratio)
        self._truncate_warned = False

    def _disk_cache_file(self, sample_id: str) -> Optional[Path]:
        if self.disk_cache_dir is None:
            return None
        return self.disk_cache_dir / f"{sample_id}.npy"

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _train_random_downsample(self, x: np.ndarray) -> np.ndarray:
        t, _ = x.shape
        if t <= self.seq_len:
            return x

        edges = np.linspace(0, t, num=self.seq_len + 1, dtype=np.float32)
        edges = np.floor(edges).astype(np.int64)
        edges = np.clip(edges, 0, t)

        starts = edges[:-1]
        ends = edges[1:]
        sizes = ends - starts
        sizes = np.where(sizes <= 0, 1, sizes)

        offsets = (np.random.random(self.seq_len) * sizes).astype(np.int64)
        idx = starts + offsets
        idx = np.clip(idx, 0, t - 1)
        return x[idx]

    def _eval_downsample(self, x: np.ndarray) -> np.ndarray:
        t, _ = x.shape
        if t <= self.seq_len:
            return x
        idx = np.linspace(0, t - 1, num=self.seq_len, dtype=np.float32)
        idx = np.rint(idx).astype(np.int64)
        idx = np.clip(idx, 0, t - 1)
        return x[idx]

    def _pad_to_len(self, x: np.ndarray) -> np.ndarray:
        t, c = x.shape
        if t >= self.seq_len:
            return x
        pad = np.zeros((self.seq_len - t, c), dtype=x.dtype)
        return np.concatenate([x, pad], axis=0)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_i = int(self.indices[i])
        rec = self.base.get_record(base_i)
        target = rec.target

        if self.ram_cache is not None:
            x = self.ram_cache.get(base_i)
        else:
            cache_file = self._disk_cache_file(rec.sample_id)
            if cache_file is not None and cache_file.exists():
                x = np.load(cache_file, allow_pickle=False)
            else:
                x = np.loadtxt(rec.path, dtype=np.float32, usecols=range(1, 9))

        # Apply truncation if specified
        if self.truncate_ratio < 1.0 or self.truncate_start_ratio > 0.0:
            orig_len = x.shape[0]
            start_idx = int(orig_len * self.truncate_start_ratio)
            end_idx = int(orig_len * self.truncate_ratio)

            # Ensure valid range
            start_idx = max(0, min(start_idx, orig_len - 1))
            end_idx = max(start_idx + 1, min(end_idx, orig_len))

            x = x[start_idx:end_idx]
            current_len = x.shape[0]
            
            # Warn if truncated length is less than seq_len
            if current_len < self.seq_len and not self._truncate_warned:
                import warnings
                warnings.warn(
                    f"Truncated length ({current_len}) is less than seq_len ({self.seq_len}). "
                    f"Original length was {orig_len}. "
                    f"Start ratio: {self.truncate_start_ratio}, End ratio: {self.truncate_ratio}",
                    UserWarning
                )
                self._truncate_warned = True

        if self.train:
            if self.train_downsample == "uniform":
                x = self._eval_downsample(x)
            else:
                x = self._train_random_downsample(x)
        else:
            x = self._eval_downsample(x)

        if self.input_norm == "per_sample":
            mean = x.mean(axis=0, keepdims=True)
            std = x.std(axis=0, keepdims=True)
            std = np.where(std < 1e-6, 1.0, std)
            x = (x - mean) / std

        x = self._pad_to_len(x)
        x_t = torch.from_numpy(x.T.copy())

        if self.zero_channels:
            x_t[list(self.zero_channels), :] = 0.0

        y_cls = torch.tensor(int(target["gas"]), dtype=torch.long)
        ppm = float(target["ppm"])
        if self.reg_target == "log_ppm":
            y_reg = torch.tensor(float(np.log(ppm)), dtype=torch.float32)
        else:
            y_reg = torch.tensor(ppm, dtype=torch.float32)
        return x_t, y_cls, y_reg


class TwinGasTimeSeriesDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        root: Path,
        download: bool,
        batch_size: int,
        num_workers: int,
        prefetch_factor: int,
        seed: int,
        test_size: float,
        seq_len: int,
        input_norm: str,
        train_downsample: str,
        reg_target: str,
        zero_channels: int,
        zero_channel_mask: Optional[int],
        zero_channel_indices: Optional[Tuple[int, ...]],
        disk_cache: bool,
        ram_cache: bool,
        ram_cache_max_samples: int,
        ram_cache_disk: bool,
        truncate_ratio: float = 1.0,
        truncate_start_ratio: float = 0.0,
    ):
        super().__init__()
        self.root = root
        self.download = download
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.prefetch_factor = int(prefetch_factor)
        self.seed = int(seed)
        self.test_size = float(test_size)
        self.seq_len = int(seq_len)
        self.input_norm = str(input_norm)
        self.train_downsample = str(train_downsample)
        self.reg_target = str(reg_target)
        self.zero_channels = int(zero_channels)
        self.zero_channel_mask = None if zero_channel_mask is None else int(zero_channel_mask)
        self._zero_channel_indices_arg = None if zero_channel_indices is None else tuple(int(i) for i in zero_channel_indices)
        self.disk_cache = bool(disk_cache)
        self.ram_cache = bool(ram_cache)
        self.ram_cache_max_samples = int(ram_cache_max_samples)
        self.ram_cache_disk = bool(ram_cache_disk)
        self.truncate_ratio = float(truncate_ratio)
        self.truncate_start_ratio = float(truncate_start_ratio)

        self._base: Optional[TwinGasSensorArrays] = None
        self.train_ds: Optional[TwinGasTimeSeriesDataset] = None
        self.val_ds: Optional[TwinGasTimeSeriesDataset] = None
        self._zero_channel_indices: Tuple[int, ...] = ()
        self._ram_cache: Optional[TwinGasRamCache] = None

    def prepare_data(self) -> None:
        TwinGasSensorArrays(root=self.root, download=self.download)

    def setup(self, stage: Optional[str] = None) -> None:
        if self._base is not None and self.train_ds is not None and self.val_ds is not None:
            return

        base = TwinGasSensorArrays(root=self.root, download=False)
        labels = np.asarray([int(base.get_record(i).target["gas"]) for i in range(len(base))], dtype=np.int64)
        indices = np.arange(len(base), dtype=np.int64)

        num_channels = 8
        if self._zero_channel_indices_arg is not None:
            uniq = sorted(set(int(i) for i in self._zero_channel_indices_arg))
            if any(i < 0 or i >= num_channels for i in uniq):
                raise ValueError(f"zero_channel_indices must be in [0, {num_channels - 1}], got {uniq}")
            zero_idx = tuple(uniq)
        elif self.zero_channel_mask is not None:
            mask = int(self.zero_channel_mask)
            if mask < 0 or mask > (1 << num_channels) - 1:
                raise ValueError(f"zero_channel_mask must be in [0, {2 ** num_channels - 1}], got {mask}")
            zero_idx = tuple(i for i in range(num_channels) if ((mask >> i) & 1) == 1)
        else:
            if self.zero_channels < 0 or self.zero_channels > num_channels:
                raise ValueError(f"zero_channels must be in [0, {num_channels}], got {self.zero_channels}")
            if self.zero_channels == 0:
                zero_idx = ()
            else:
                rng = np.random.default_rng(self.seed)
                chosen = rng.choice(num_channels, size=self.zero_channels, replace=False)
                zero_idx = tuple(int(i) for i in np.sort(chosen))
        self._zero_channel_indices = zero_idx

        idx_train, idx_val = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=labels,
        )

        disk_cache_dir: Optional[Path] = None
        if self.disk_cache:
            disk_cache_dir = base.dataset_dir / "cache" / "twin_gas_sensor_arrays_npy_v1"

        if self.ram_cache:
            start_method = mp.get_start_method(allow_none=True)
            if start_method is None:
                start_method = mp.get_start_method()
            if self.num_workers > 0 and start_method != "fork":
                print("RAM cache requires multiprocessing start method 'fork' when num_workers>0; forcing num_workers=0")
                self.num_workers = 0
            cache_indices = np.unique(np.concatenate([idx_train, idx_val]))
            if self.ram_cache_max_samples > 0 and cache_indices.shape[0] > self.ram_cache_max_samples:
                rng = np.random.default_rng(self.seed)
                cache_indices = rng.choice(cache_indices, size=self.ram_cache_max_samples, replace=False)
            cache_dir = disk_cache_dir if self.ram_cache_disk else None
            ram_cache = TwinGasRamCache(cache_dir=cache_dir, save_to_disk=self.ram_cache_disk)
            ram_cache.preload(base, cache_indices)
            self._ram_cache = ram_cache

        self._base = base
        self.train_ds = TwinGasTimeSeriesDataset(
            base,
            idx_train,
            seq_len=self.seq_len,
            train=True,
            zero_channels=self._zero_channel_indices,
            input_norm=self.input_norm,
            train_downsample=self.train_downsample,
            reg_target=self.reg_target,
            ram_cache=self._ram_cache,
            disk_cache_dir=disk_cache_dir,
            truncate_ratio=self.truncate_ratio,
            truncate_start_ratio=self.truncate_start_ratio,
        )
        self.val_ds = TwinGasTimeSeriesDataset(
            base,
            idx_val,
            seq_len=self.seq_len,
            train=False,
            zero_channels=self._zero_channel_indices,
            input_norm=self.input_norm,
            train_downsample="uniform",
            reg_target=self.reg_target,
            ram_cache=self._ram_cache,
            disk_cache_dir=disk_cache_dir,
            truncate_ratio=self.truncate_ratio,
            truncate_start_ratio=self.truncate_start_ratio,
        )

    @property
    def zero_channel_indices(self) -> Tuple[int, ...]:
        return tuple(self._zero_channel_indices)

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "persistent_workers": self.num_workers > 0,
        }
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(self.train_ds, **kwargs)

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "persistent_workers": self.num_workers > 0,
        }
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(self.val_ds, **kwargs)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNBackbone(nn.Module):
    def __init__(self, in_channels: int = 8, num_channels: list = None, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 128, 256]
        
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = in_channels if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_ch, out_ch, kernel_size, stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        x = self.pool(x)
        x = x.flatten(1)
        return x


class Conv1DBackbone(nn.Module):
    def __init__(self, in_channels: int = 8):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        x = x.flatten(1)
        return x


class TwinGasMultiTaskNet(nn.Module):
    def __init__(self, *, in_channels: int = 8, num_classes: int = 4, backbone: str = "tcn"):
        super().__init__()
        if backbone == "tcn":
            self.backbone = TCNBackbone(in_channels=in_channels)
        else:
            self.backbone = Conv1DBackbone(in_channels=in_channels)
        self.dropout = nn.Dropout(p=0.2)
        self.cls_head = nn.Linear(256, num_classes)
        self.reg_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        feat = self.dropout(feat)
        logits = self.cls_head(feat)
        reg = self.reg_head(feat).squeeze(-1)
        return logits, reg


class TwinGasLitModule(L.LightningModule):
    def __init__(
        self,
        *,
        lr: float,
        task: str,
        reg_weight: float,
        reg_loss_scale: float,
        num_classes: int = 4,
        backbone: str = "tcn",
        lr_scheduler: str = "none",
        lr_scheduler_patience: int = 10,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_t_max: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = TwinGasMultiTaskNet(in_channels=8, num_classes=num_classes, backbone=backbone)
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_reg = nn.MSELoss()

        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.train_r2 = R2Score()
        self.val_r2 = R2Score()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y_cls, y_reg = batch
        logits, reg = self(x)

        task = str(self.hparams.task)
        loss = 0.0

        if task in {"cls", "mtl"}:
            loss_cls = self.criterion_cls(logits, y_cls)
            acc = self.train_acc(logits, y_cls)
            self.log("train/cls_loss", loss_cls, on_step=False, on_epoch=True, prog_bar=False)
            self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
            loss = loss + loss_cls

        if task in {"reg", "mtl"}:
            scale = float(self.hparams.reg_loss_scale)
            loss_reg = self.criterion_reg(reg / scale, y_reg / scale)
            r2 = self.train_r2(reg, y_reg)
            self.log("train/reg_loss", loss_reg, on_step=False, on_epoch=True, prog_bar=False)
            self.log("train/r2", r2, on_step=False, on_epoch=True, prog_bar=True)
            if task == "mtl":
                loss = loss + float(self.hparams.reg_weight) * loss_reg
            else:
                loss = loss + loss_reg

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_cls, y_reg = batch
        logits, reg = self(x)

        task = str(self.hparams.task)
        loss = 0.0

        out = {"val_loss": None, "val_acc": None, "val_r2": None}

        if task in {"cls", "mtl"}:
            loss_cls = self.criterion_cls(logits, y_cls)
            acc = self.val_acc(logits, y_cls)
            self.log("val/cls_loss", loss_cls, on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
            loss = loss + loss_cls
            out["val_acc"] = acc

        if task in {"reg", "mtl"}:
            scale = float(self.hparams.reg_loss_scale)
            loss_reg = self.criterion_reg(reg / scale, y_reg / scale)
            r2 = self.val_r2(reg, y_reg)
            self.log("val/reg_loss", loss_reg, on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/r2", r2, on_step=False, on_epoch=True, prog_bar=True)
            if task == "mtl":
                loss = loss + float(self.hparams.reg_weight) * loss_reg
            else:
                loss = loss + loss_reg
            out["val_r2"] = r2

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        out["val_loss"] = loss
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=float(self.hparams.lr))
        
        lr_scheduler = str(self.hparams.lr_scheduler)
        if lr_scheduler == "none":
            return optimizer
        
        scheduler_config = {"optimizer": optimizer}
        
        if lr_scheduler == "reduce_on_plateau":
            task = str(self.hparams.task)
            monitor = "val/r2" if task == "reg" else "val/acc"
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=float(self.hparams.lr_scheduler_factor),
                patience=int(self.hparams.lr_scheduler_patience),
                verbose=True,
            )
            scheduler_config["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": monitor,
                "interval": "epoch",
                "frequency": 1,
            }
        elif lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.hparams.lr_scheduler_t_max),
                eta_min=0,
            )
            scheduler_config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        elif lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(self.hparams.lr_scheduler_patience),
                gamma=float(self.hparams.lr_scheduler_factor),
            )
            scheduler_config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        
        return scheduler_config


class DualMonitorEarlyStopping(L.Callback):
    def __init__(
        self,
        monitor1: str,
        monitor2: str,
        min_delta: float = 0.001,
        patience: int = 10,
        verbose: bool = False,
        mode: str = "max",
    ):
        super().__init__()
        self.monitor1 = monitor1
        self.monitor2 = monitor2
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        
        self.wait_count = 0
        self.best_score1 = None
        self.best_score2 = None

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        if self.monitor1 not in metrics or self.monitor2 not in metrics:
            return

        score1 = metrics[self.monitor1].item()
        score2 = metrics[self.monitor2].item()

        if self.best_score1 is None:
            self.best_score1 = score1
            self.best_score2 = score2
            return

        # Check improvements
        improved1 = False
        improved2 = False

        if self.mode == "max":
            if score1 > self.best_score1 + self.min_delta:
                improved1 = True
            if score2 > self.best_score2 + self.min_delta:
                improved2 = True
        else:
            if score1 < self.best_score1 - self.min_delta:
                improved1 = True
            if score2 < self.best_score2 - self.min_delta:
                improved2 = True

        if improved1:
            self.best_score1 = score1
        if improved2:
            self.best_score2 = score2

        if improved1 or improved2:
            if self.wait_count > 0 and self.verbose:
                print(f"EarlyStopping counter reset (imp1={improved1}, imp2={improved2})")
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.wait_count} out of {self.patience}. Scores: {self.monitor1}={score1:.4f}, {self.monitor2}={score2:.4f}")
            
            if self.wait_count >= self.patience:
                trainer.should_stop = True
                if self.verbose:
                    print(f"Stopping because both {self.monitor1} and {self.monitor2} did not improve for {self.patience} epochs.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(Path.cwd() / ".cache" / "enose_uci_dataset"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--task", type=str, default="cls", choices=["cls", "reg", "mtl"])
    parser.add_argument("--input-norm", type=str, default="auto", choices=["auto", "per_sample", "none"])
    parser.add_argument("--train-downsample", type=str, default="auto", choices=["auto", "random", "uniform"])
    parser.add_argument("--reg-target", type=str, default="ppm", choices=["ppm", "log_ppm"])
    parser.add_argument("--reg-weight", type=float, default=1.0)
    parser.add_argument("--reg-loss-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seq-len", type=int, default=2000)
    parser.add_argument("--zero-channels", type=int, default=0)
    parser.add_argument("--zero-channel-mask", type=int, default=None)
    parser.add_argument("--zero-channel-indices", type=str, default=None)
    parser.add_argument("--disk-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ram-cache", action="store_true")
    parser.add_argument("--ram-cache-max-samples", type=int, default=0)
    parser.add_argument("--ram-cache-disk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--build-disk-cache", action="store_true")
    parser.add_argument("--check-val-every-n-epoch", type=int, default=1)
    parser.add_argument("--ckpt-every-n-epochs", type=int, default=1)
    parser.add_argument("--save-last", action="store_true")
    parser.add_argument("--backbone", type=str, default="tcn", choices=["tcn", "conv1d"])
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--lr-scheduler", type=str, default="none", choices=["none", "reduce_on_plateau", "cosine", "step"])
    parser.add_argument("--lr-scheduler-patience", type=int, default=10)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--lr-scheduler-t-max", type=int, default=50)
    parser.add_argument("--truncate-ratio", type=float, default=1.0, help="Truncate to this ratio of original length (0.0-1.0)")
    parser.add_argument("--truncate-start-ratio", type=float, default=0.0, help="Start truncation from this ratio (0.0-1.0)")
    parser.add_argument(
        "--logdir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "runs" / "twin_timeseries_conv1d"),
    )
    args = parser.parse_args()

    def _parse_indices(s: Optional[str]) -> Optional[Tuple[int, ...]]:
        if s is None:
            return None
        s = str(s).strip()
        if not s:
            return ()
        parts = [p.strip() for p in s.split(",")]
        out = [int(p) for p in parts if p]
        return tuple(out)

    zero_channel_indices = _parse_indices(args.zero_channel_indices)

    input_norm = str(args.input_norm)
    if input_norm == "auto":
        input_norm = "per_sample" if str(args.task) == "cls" else "none"

    train_downsample = str(args.train_downsample)
    if train_downsample == "auto":
        train_downsample = "random"

    reg_loss_scale = float(args.reg_loss_scale)
    if reg_loss_scale <= 0:
        reg_loss_scale = 250.0 if str(args.reg_target) == "ppm" else 5.0

    L.seed_everything(args.seed, workers=True)

    root = Path(args.root).expanduser()

    dm = TwinGasTimeSeriesDataModule(
        root=root,
        download=args.download,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed,
        test_size=args.test_size,
        seq_len=args.seq_len,
        input_norm=input_norm,
        train_downsample=train_downsample,
        reg_target=args.reg_target,
        zero_channels=args.zero_channels,
        zero_channel_mask=args.zero_channel_mask,
        zero_channel_indices=zero_channel_indices,
        disk_cache=args.disk_cache,
        ram_cache=(True if args.build_disk_cache else args.ram_cache),
        ram_cache_max_samples=(0 if args.build_disk_cache else args.ram_cache_max_samples),
        ram_cache_disk=(True if args.build_disk_cache else args.ram_cache_disk),
        truncate_ratio=args.truncate_ratio,
        truncate_start_ratio=args.truncate_start_ratio,
    )

    dm.prepare_data()

    dm.setup(stage="fit")

    # DEBUG: Print dataset sample
    if dm.train_ds is not None and len(dm.train_ds) > 0:
        print("\n" + "=" * 40)
        print("DEBUG: Dataset Sample Info")
        print(f"Train dataset length: {len(dm.train_ds)}")
        print(f"Val dataset length: {len(dm.val_ds)}")

        # Calculate length statistics
        print("Calculating length statistics for first 100 samples...")
        lengths = []
        for i in range(min(100, len(dm.train_ds))):
            # Bypass __getitem__ truncation to get original length
            base_i = int(dm.train_ds.indices[i])
            if dm.train_ds.ram_cache is not None:
                x_orig = dm.train_ds.ram_cache.get(base_i)
            else:
                rec = dm.train_ds.base.get_record(base_i)
                cache_file = dm.train_ds._disk_cache_file(rec.sample_id)
                if cache_file is not None and cache_file.exists():
                    x_orig = np.load(cache_file, allow_pickle=False)
                else:
                    x_orig = np.loadtxt(rec.path, dtype=np.float32, usecols=range(1, 9))
            lengths.append(x_orig.shape[0])
        
        lengths = np.array(lengths)
        print(f"Original Lengths Stats (First {len(lengths)} samples):")
        print(f"  Min: {lengths.min()}")
        print(f"  Max: {lengths.max()}")
        print(f"  Mean: {lengths.mean():.2f}")
        print(f"  Median: {np.median(lengths):.2f}")
        print(f"  Truncate Ratio: {args.truncate_ratio}")
        print(f"  Truncate Start Ratio: {args.truncate_start_ratio}")
        print(f"  Expected Truncated Length (Mean): {lengths.mean() * (args.truncate_ratio - args.truncate_start_ratio):.2f}")
        
        # Get one sample
        x, y_cls, y_reg = dm.train_ds[0]
        print(f"\nSample 0:")
        print(f"  Input shape: {x.shape}")
        print(f"  Class label: {y_cls} (type: {y_cls.dtype})")
        print(f"  Reg target:  {y_reg} (type: {y_reg.dtype})")
        print(f"  Input stats: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}, std={x.std():.4f}")
        print(f"  First 8 channels, first 10 timesteps:\n{x[:, :10]}")
        print("=" * 40 + "\n")

    if args.build_disk_cache:
        return

    lit = TwinGasLitModule(
        lr=args.lr,
        task=args.task,
        reg_weight=args.reg_weight,
        reg_loss_scale=reg_loss_scale,
        num_classes=4,
        backbone=args.backbone,
        lr_scheduler=args.lr_scheduler,
        lr_scheduler_patience=args.lr_scheduler_patience,
        lr_scheduler_factor=args.lr_scheduler_factor,
        lr_scheduler_t_max=args.lr_scheduler_t_max,
    )

    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    logdir = Path(args.logdir)
    logger = CSVLogger(save_dir=str(logdir), name="")

    effective_mask = 0
    for i in dm.zero_channel_indices:
        effective_mask |= 1 << int(i)

    run_meta = {
        "seed": int(args.seed),
        "task": str(args.task),
        "reg_target": str(args.reg_target),
        "reg_weight": float(args.reg_weight),
        "reg_loss_scale": float(reg_loss_scale),
        "input_norm": str(input_norm),
        "train_downsample": str(train_downsample),
        "zero_channels": int(args.zero_channels),
        "zero_channel_mask_arg": None if args.zero_channel_mask is None else int(args.zero_channel_mask),
        "zero_channel_indices_arg": list(zero_channel_indices) if zero_channel_indices is not None else None,
        "zero_channel_mask": int(effective_mask),
        "zero_channel_indices": list(dm.zero_channel_indices),
        "seq_len": int(args.seq_len),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "epochs": int(args.epochs),
        "test_size": float(args.test_size),
        "num_workers": int(dm.num_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "disk_cache": bool(args.disk_cache),
        "ram_cache": bool(args.ram_cache),
        "ram_cache_max_samples": int(args.ram_cache_max_samples),
        "ram_cache_disk": bool(args.ram_cache_disk),
        "check_val_every_n_epoch": int(args.check_val_every_n_epoch),
        "ckpt_every_n_epochs": int(args.ckpt_every_n_epochs),
        "save_last": bool(args.save_last),
        "root": str(root),
    }
    run_dir = Path(logger.log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if str(args.task) == "reg":
        monitor = "val/r2"
        ckpt_filename = "epoch{epoch:03d}-val_r2{val/r2:.4f}"
    else:
        monitor = "val/acc"
        ckpt_filename = "epoch{epoch:03d}-val_acc{val/acc:.4f}"

    ckpt = ModelCheckpoint(
        monitor=monitor,
        mode="max",
        save_last=bool(args.save_last),
        every_n_epochs=int(args.ckpt_every_n_epochs),
        filename=ckpt_filename,
    )

    callbacks = [ckpt]
    if args.early_stopping:
        if str(args.task) == "mtl":
            print(f"Using DualMonitorEarlyStopping for task mtl (acc & r2)")
            early_stop = DualMonitorEarlyStopping(
                monitor1="val/acc",
                monitor2="val/r2",
                min_delta=0.001,
                patience=int(args.early_stopping_patience),
                mode="max",
                verbose=True,
            )
            callbacks.append(early_stop)
        else:
            early_stop = EarlyStopping(
                monitor=monitor,
                min_delta=0.001,
                patience=int(args.early_stopping_patience),
                mode="max",
                verbose=True,
            )
            callbacks.append(early_stop)

    torch.set_float32_matmul_precision('medium')

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        check_val_every_n_epoch=int(args.check_val_every_n_epoch),
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    trainer.fit(lit, datamodule=dm)


if __name__ == "__main__":
    main()
