from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


try:
    from torch.utils.data import Dataset as _TorchDataset  # type: ignore
except Exception:  # pragma: no cover
    class _TorchDataset:  # type: ignore
        pass


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    path: Path
    target: Any
    meta: Dict[str, Any]


class DatasetWithTransforms(_TorchDataset):
    def __init__(
        self,
        root: Union[str, Path],
        *,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        root_path = Path(root).expanduser()
        self.root = root_path

        has_transforms = transforms is not None
        has_separate = transform is not None or target_transform is not None
        if has_transforms and has_separate:
            raise ValueError("Only transforms or transform/target_transform can be passed, not both")

        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform

    def __repr__(self) -> str:
        head = f"{self.__class__.__name__}({self.root})"
        extra = self.extra_repr()
        if not extra:
            return head
        lines = [head]
        for line in extra.split("\n"):
            lines.append(f"  {line}")
        return "\n".join(lines)

    def extra_repr(self) -> str:
        return ""


class BaseEnoseDataset(DatasetWithTransforms):
    name: str = ""

    def __init__(
        self,
        root: Union[str, Path],
        *,
        split: Optional[str] = None,
        download: bool = False,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)

        if not self.name:
            raise ValueError("Dataset must define class attribute 'name'")

        self.split = split
        from ._info import get_dataset_info  # local import to avoid cycles

        self.info = get_dataset_info(self.name)
        self._samples: List[SampleRecord] = []

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                f"Dataset not found or corrupted: {self.name}. "
                f"You can use download=True to download it."
            )

        self._samples = self._make_dataset()

    @property
    def dataset_dir(self) -> Path:
        return self.root / self.name

    @property
    def uci_id(self) -> Optional[int]:
        return self.info.uci_id

    @property
    def tasks(self) -> List[str]:
        return list(self.info.tasks)

    @property
    def sensor_type(self) -> str:
        return self.info.sensors.type

    @property
    def num_sensors(self) -> int:
        return self.info.sensors.count

    @property
    def sample_rate_hz(self) -> Optional[int]:
        if self.info.time_series is None:
            return None
        return self.info.time_series.sample_rate_hz

    @property
    def raw_dir(self) -> Path:
        return self.dataset_dir / "raw"

    def download(self) -> None:
        raise NotImplementedError

    def _check_exists(self) -> bool:
        return self.dataset_dir.exists()

    def _make_dataset(self) -> List[SampleRecord]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        rec = self._samples[index]
        data, target = self._load_sample(rec)

        if self.transforms is not None:
            data, target = self.transforms(data, target)
        else:
            if self.transform is not None:
                data = self.transform(data)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return data, target

    def _load_sample(self, record: SampleRecord) -> Tuple[Any, Any]:
        raise NotImplementedError

    def get_record(self, index: int) -> SampleRecord:
        return self._samples[index]

    def extra_repr(self) -> str:
        parts = []
        if self.split is not None:
            parts.append(f"split={self.split}")
        parts.append(f"samples={len(self)}")
        return "\n".join(parts)
