from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import lightning as L
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from enose_uci_dataset.datasets import TwinGasSensorArrays


def extract_features(df) -> np.ndarray:
    """把一个样本的时序 DataFrame -> 固定维度特征向量。

    这里用每个传感器列的统计量：mean/std/min/max（共 8*4=32 维）。
    """

    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    x = df[sensor_cols].to_numpy(dtype=np.float32)
    feats = np.concatenate(
        [
            np.nanmean(x, axis=0),
            np.nanstd(x, axis=0),
            np.nanmin(x, axis=0),
            np.nanmax(x, axis=0),
        ],
        axis=0,
    )
    return feats.astype(np.float32)


@dataclass
class XY:
    x: np.ndarray
    y: np.ndarray


def build_xy(root: Path, *, download: bool) -> XY:
    ds = TwinGasSensorArrays(root=root, download=download)

    xs: List[np.ndarray] = []
    ys: List[int] = []

    for i in tqdm(range(len(ds)), desc="Featurizing"):
        df, target = ds[i]
        xs.append(extract_features(df))
        ys.append(int(target["gas"]))

    x = np.stack(xs, axis=0)
    y = np.asarray(ys, dtype=np.int64)
    return XY(x=x, y=y)


class NumpyDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class TwinGasDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        root: Path,
        download: bool,
        batch_size: int,
        num_workers: int,
        seed: int,
        test_size: float,
    ):
        super().__init__()
        self.root = root
        self.download = download
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.test_size = test_size

        self._scaler: Optional[StandardScaler] = None
        self.train_ds: Optional[NumpyDataset] = None
        self.val_ds: Optional[NumpyDataset] = None

    def prepare_data(self) -> None:
        TwinGasSensorArrays(root=self.root, download=self.download)

    def setup(self, stage: Optional[str] = None) -> None:
        xy = build_xy(self.root, download=False)

        x_train, x_val, y_train, y_val = train_test_split(
            xy.x,
            xy.y,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=xy.y,
        )

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train).astype(np.float32)
        x_val = scaler.transform(x_val).astype(np.float32)

        self._scaler = scaler
        self.train_ds = NumpyDataset(x_train, y_train)
        self.val_ds = NumpyDataset(x_val, y_val)

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class TwinGasLitModule(L.LightningModule):
    def __init__(self, *, in_dim: int, lr: float, num_classes: int = 4):
        super().__init__()
        self.save_hyperparameters()

        self.model = MLP(in_dim=in_dim, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.train_acc(logits, y)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.val_acc(logits, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(Path.cwd() / ".cache" / "enose_uci_dataset"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--logdir", type=str, default=str(Path(__file__).resolve().parents[1] / "runs" / "twin_mlp"))
    args = parser.parse_args()

    L.seed_everything(args.seed, workers=True)

    root = Path(args.root).expanduser()

    dm = TwinGasDataModule(
        root=root,
        download=args.download,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        test_size=args.test_size,
    )

    dm.prepare_data()
    dm.setup(stage="fit")
    assert dm.train_ds is not None
    in_dim = int(dm.train_ds.x.shape[1])

    lit = TwinGasLitModule(in_dim=in_dim, lr=args.lr, num_classes=4)

    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    logdir = Path(args.logdir)
    logger = CSVLogger(save_dir=str(logdir), name="")
    ckpt = ModelCheckpoint(
        monitor="val/acc",
        mode="max",
        save_last=True,
        filename="epoch{epoch:03d}-val_acc{val/acc:.4f}",
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[ckpt],
        log_every_n_steps=10,
    )

    trainer.fit(lit, datamodule=dm)


if __name__ == "__main__":
    main()
