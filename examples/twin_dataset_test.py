from torch.utils.data import Dataset, DataLoader

import sys
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from enose_uci_dataset.datasets import TwinGasSensorArrays



def test():
    dataset = TwinGasSensorArrays(root=str(Path.cwd() / ".cache" / "enose_uci_dataset"))
    dataloader = DataLoader(dataset, batch_size=4)
    for batch in dataloader:
        print(batch)
        exit()


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

if __name__ == "__main__":
    test()