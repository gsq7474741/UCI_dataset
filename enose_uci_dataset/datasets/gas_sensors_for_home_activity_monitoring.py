from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from ._base import BaseEnoseDataset, SampleRecord
from ._info import get_dataset_info
from ._utils import download_and_extract


class GasSensorsForHomeActivityMonitoring(BaseEnoseDataset):
    name = "gas_sensors_for_home_activity_monitoring"

    classes = ["banana", "wine", "background"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    def __init__(
        self,
        root: Union[str, Path],
        *,
        split: Optional[str] = None,
        download: bool = False,
        transforms=None,
        transform=None,
        target_transform=None,
    ):
        super().__init__(
            root,
            split=split,
            download=download,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )

    @property
    def processed_dir(self) -> Path:
        return self.dataset_dir / "processed" / "v1" / "ssl_samples"

    @property
    def schema_path(self) -> Path:
        # 复用 legacy 中人工整理过的列级元数据
        return Path(__file__).resolve().parents[1] / "legacy" / self.name / "metadata.json"

    def download(self) -> None:
        info = get_dataset_info(self.name)
        download_and_extract(info, self.dataset_dir, force=False, verify=True)

    def _check_exists(self) -> bool:
        if self.processed_dir.exists() and any(self.processed_dir.glob("*.csv")):
            return True
        if self.raw_dir.exists() and any(self.raw_dir.iterdir()):
            return True
        return False

    def _ensure_processed(self) -> None:
        if self.processed_dir.exists() and any(self.processed_dir.glob("*.csv")):
            return

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self._process_raw_to_processed()

    def _process_raw_to_processed(self) -> None:
        meta_path = self.raw_dir / "HT_Sensor_metadata.dat"
        data_path = self.raw_dir / "HT_Sensor_dataset.dat"

        if not meta_path.exists() or not data_path.exists():
            raise RuntimeError(
                f"raw 文件不存在，无法处理: {self.raw_dir}\n"
                f"  期望: HT_Sensor_metadata.dat, HT_Sensor_dataset.dat"
            )

        metadata = pd.read_csv(meta_path, sep="\t+", engine="python")
        sensor_data = pd.read_csv(data_path, sep="\\s+", engine="python")

        gas_mapping = self.class_to_idx

        for _, meta_row in metadata.iterrows():
            sample_id = int(meta_row["id"])
            gas_type = str(meta_row["class"])

            if sample_id == 95:
                # legacy 中记录该样本存在缺失值
                continue

            if gas_type not in gas_mapping:
                continue

            date = pd.to_datetime(meta_row["date"], format="%m-%d-%y")

            sample_data = sensor_data[sensor_data["id"] == sample_id].copy()
            if sample_data.empty:
                continue

            sample_data["date"] = date + pd.to_timedelta(sample_data["time"], unit="h")
            sample_data["date"] = sample_data["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
            sample_data["t_s"] = sample_data["time"] * 3600.0

            sample_data = sample_data.rename(
                columns={
                    "R1": "sensor_0",
                    "R2": "sensor_1",
                    "R3": "sensor_2",
                    "R4": "sensor_3",
                    "R5": "sensor_4",
                    "R6": "sensor_5",
                    "R7": "sensor_6",
                    "R8": "sensor_7",
                    "Temp.": "temp",
                    "Humidity": "humidity",
                }
            )

            sample_data["label_gas"] = int(gas_mapping[gas_type])

            cols = [
                "sensor_0",
                "sensor_1",
                "sensor_2",
                "sensor_3",
                "sensor_4",
                "sensor_5",
                "sensor_6",
                "sensor_7",
                "temp",
                "humidity",
                "t_s",
                "date",
                "label_gas",
            ]

            missing = [c for c in cols if c not in sample_data.columns]
            if missing:
                raise RuntimeError(f"处理后列缺失: {missing}")

            sample_data = sample_data[cols]

            out = self.processed_dir / f"{sample_id}_{gas_type}.csv"
            sample_data.to_csv(out, index=False)

    def _load_splits(self) -> Optional[Dict[str, List[str]]]:
        path = self.processed_dir / "splits.json"
        if not path.exists():
            return None
        import json

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if k in {"train", "val", "test"}}

    def _make_dataset(self) -> List[SampleRecord]:
        self._ensure_processed()

        split_filter: Optional[set] = None
        if self.split is not None:
            splits = self._load_splits()
            if splits and self.split in splits:
                split_filter = set(splits[self.split])

        samples: List[SampleRecord] = []
        for p in sorted(self.processed_dir.glob("*.csv")):
            fname = p.name
            if split_filter is not None and fname not in split_filter:
                continue

            head = pd.read_csv(p, nrows=1)
            if "label_gas" not in head.columns:
                continue
            target = int(head["label_gas"].iloc[0])

            sample_id = p.stem
            meta: Dict[str, Any] = {"file": fname}

            samples.append(SampleRecord(sample_id=sample_id, path=p, target=target, meta=meta))

        return samples

    # Columns that are sensor readings (to be kept for pretraining)
    _sensor_columns = [f"sensor_{i}" for i in range(8)]
    
    def _load_sample(self, record: SampleRecord) -> Tuple[pd.DataFrame, int]:
        df = pd.read_csv(record.path)
        # Keep only sensor columns for pretraining
        # (drop temp, humidity, t_s, date, label_gas)
        sensor_cols = [c for c in self._sensor_columns if c in df.columns]
        df = df[sensor_cols]
        return df, int(record.target)

    @property
    def targets(self) -> List[int]:
        return [int(r.target) for r in self._samples]

    def extra_repr(self) -> str:
        base = super().extra_repr()
        return "\n".join([base, f"classes={len(self.classes)}"]) if base else f"classes={len(self.classes)}"
