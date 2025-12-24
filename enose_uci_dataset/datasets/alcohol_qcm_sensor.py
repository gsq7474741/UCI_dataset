"""Alcohol QCM Sensor Dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ._base import BaseEnoseDataset, SampleRecord
from ._info import get_dataset_info
from ._utils import download_and_extract


class AlcoholQCMSensor(BaseEnoseDataset):
    """Alcohol QCM Sensor Dataset (UCI ML Repository, id=496).

    Source:
        https://archive.ics.uci.edu/dataset/496/alcohol+qcm+sensor+dataset

    Dataset summary (from the UCI page):
        Classification of 5 types of alcohol using 5 different QCM (Quartz Crystal Microbalance) sensors.
        125 instances, 8 features.

    Dataset Information (UCI page):
        In the dataset there are 5 types of QCM sensors: QCM3, QCM6, QCM7, QCM10, QCM12.
        Each sensor has different MIP (Molecularly Imprinted Polymer) and NP (Non-imprinted Polymer) ratios:
        - QCM3:  MIP ratio=1, NP ratio=1
        - QCM6:  MIP ratio=1, NP ratio=0
        - QCM7:  MIP ratio=1, NP ratio=0.5
        - QCM10: MIP ratio=1, NP ratio=2
        - QCM12: MIP ratio=0, NP ratio=1

        5 types of alcohols are classified:
        - 1-octanol
        - 1-propanol
        - 2-butanol
        - 2-propanol
        - 1-isobutanol

    Variable Information (UCI page):
        The gas sample is passed through the sensor in five different concentrations.
        Concentration ratios (Air ratio : Gas ratio in ml):
        - Concentration 1: 0.799 : 0.201
        - Concentration 2: 0.700 : 0.300
        - Concentration 3: 0.600 : 0.400
        - Concentration 4: 0.501 : 0.499
        - Concentration 5: 0.400 : 0.600

        Each sensor has two channels (Channel 1 and Channel 2).

    Introductory Paper:
        Adak et al. "Classification of alcohols obtained by QCM sensors with different
        characteristics using ABC based neural network", Engineering Science and Technology,
        an International Journal, 2020.

    DOI:
        https://doi.org/10.24432/C5KC7M

    License:
        CC BY 4.0

    Args:
        root: Root directory where the dataset is stored or will be downloaded.
        split: Optional split name (not used currently).
        download: If True, download the dataset if not found.
        cache: If True, cache processed data as .npy files for faster loading.
        transforms: Optional transforms to apply to (data, target) pairs.
        transform: Optional transform to apply to data only.
        target_transform: Optional transform to apply to target only.
    """

    name = "alcohol_qcm_sensor_dataset"

    classes = ["1-octanol", "1-propanol", "2-butanol", "2-propanol", "1-isobutanol"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    sensors = ["QCM3", "QCM6", "QCM7", "QCM10", "QCM12"]
    sensor_mip_np_ratio = {
        "QCM3": (1, 1),
        "QCM6": (1, 0),
        "QCM7": (1, 0.5),
        "QCM10": (1, 2),
        "QCM12": (0, 1),
    }

    concentration_ratios = {
        1: (0.799, 0.201),
        2: (0.700, 0.300),
        3: (0.600, 0.400),
        4: (0.501, 0.499),
        5: (0.400, 0.600),
    }

    def __init__(
        self,
        root: Union[str, Path],
        *,
        split: Optional[str] = None,
        download: bool = False,
        cache: bool = True,
        transforms=None,
        transform=None,
        target_transform=None,
    ):
        self._cache = cache
        super().__init__(
            root,
            split=split,
            download=download,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )

    @property
    def cache_dir(self) -> Path:
        return self.dataset_dir / "cache"

    def download(self) -> None:
        info = get_dataset_info(self.name)
        download_and_extract(info, self.dataset_dir, force=False, verify=True)

    def _check_exists(self) -> bool:
        raw = self.raw_dir
        if not raw.exists():
            return False
        # Check for CSV files in raw or subdirectory
        csv_files = list(raw.glob("*.csv")) + list(raw.glob("**/*.csv"))
        return len(csv_files) > 0

    def _find_csv_files(self) -> List[Path]:
        """Find all CSV files in the dataset."""
        csv_files = []
        # Check in raw directory and subdirectories
        for pattern in ["*.csv", "**/*.csv"]:
            csv_files.extend(self.raw_dir.glob(pattern))
        return sorted(set(csv_files))

    def _parse_qcm_csv(self, csv_path: Path) -> List[Dict[str, Any]]:
        """Parse a QCM sensor CSV file with the specific format.

        Format: First row is header with concentration ratios and alcohol names.
        Data rows: 10 feature values (2 channels × 5 concentrations) + 5 one-hot alcohol labels.
        """
        samples = []

        # Read with semicolon separator
        df = pd.read_csv(csv_path, sep=";", header=0)

        # The header contains concentration info and alcohol names
        # Data columns: first 10 are features (2 channels × 5 concentrations)
        # Last 5 are one-hot encoded alcohol labels

        sensor_name = csv_path.stem.upper()

        for idx, row in df.iterrows():
            # Features are the first 10 columns
            features = row.iloc[:10].values.astype(np.float32)

            # One-hot encoded labels are the last 5 columns
            one_hot = row.iloc[10:15].values.astype(int)

            # Find which alcohol (the one with 1 in one-hot)
            alcohol_idx = np.argmax(one_hot)
            if one_hot[alcohol_idx] != 1:
                continue  # Skip if no valid label

            samples.append({
                "features": features,
                "alcohol_idx": int(alcohol_idx),
                "sensor_name": sensor_name,
                "row_idx": int(idx),
            })

        return samples

    def _make_dataset(self) -> List[SampleRecord]:
        # Try to load from cache
        if self._cache and self.cache_dir.exists():
            cache_file = self.cache_dir / "features.npy"
            meta_file = self.cache_dir / "metadata.npy"
            if cache_file.exists() and meta_file.exists():
                return self._load_from_cache()

        csv_files = self._find_csv_files()
        samples: List[SampleRecord] = []

        for csv_path in csv_files:
            sensor_name = csv_path.stem.upper()
            if sensor_name not in self.sensors:
                # Try to extract sensor name from filename
                for s in self.sensors:
                    if s.lower() in csv_path.stem.lower():
                        sensor_name = s
                        break
                else:
                    continue

            try:
                parsed_samples = self._parse_qcm_csv(csv_path)
            except Exception:
                continue

            for parsed in parsed_samples:
                alcohol_idx = parsed["alcohol_idx"]

                target = {
                    "alcohol": alcohol_idx,
                    "concentration": 0,  # Not directly available per sample
                    "sensor": self.sensors.index(sensor_name),
                }
                meta = {
                    "alcohol_name": self.idx_to_class.get(alcohol_idx, str(alcohol_idx)),
                    "concentration": 0,
                    "sensor_name": sensor_name,
                    "features": parsed["features"],
                }

                sample_id = f"{sensor_name}_{parsed['row_idx']}_{meta['alcohol_name']}"
                samples.append(
                    SampleRecord(
                        sample_id=sample_id,
                        path=csv_path,
                        target=target,
                        meta=meta,
                    )
                )

        if self._cache and samples:
            self._save_to_cache(samples)

        return samples

    def _save_to_cache(self, samples: List[SampleRecord]) -> None:
        """Save processed samples to cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Find max feature length
        max_len = max(len(s.meta["features"]) for s in samples)

        # Pad features to same length
        features = np.zeros((len(samples), max_len), dtype=np.float32)
        for i, s in enumerate(samples):
            f = s.meta["features"]
            features[i, :len(f)] = f

        np.save(self.cache_dir / "features.npy", features)

        metadata = []
        for s in samples:
            metadata.append({
                "sample_id": s.sample_id,
                "alcohol": s.target["alcohol"],
                "concentration": s.target["concentration"],
                "sensor": s.target["sensor"],
                "alcohol_name": s.meta["alcohol_name"],
                "sensor_name": s.meta["sensor_name"],
                "feature_len": len(s.meta["features"]),
            })
        np.save(self.cache_dir / "metadata.npy", np.array(metadata, dtype=object))

    def _load_from_cache(self) -> List[SampleRecord]:
        """Load samples from cache."""
        features = np.load(self.cache_dir / "features.npy")
        metadata = np.load(self.cache_dir / "metadata.npy", allow_pickle=True)

        samples = []
        for i, meta in enumerate(metadata):
            feat_len = meta.get("feature_len", features.shape[1])
            target = {
                "alcohol": meta["alcohol"],
                "concentration": meta["concentration"],
                "sensor": meta["sensor"],
            }
            sample_meta = {
                "alcohol_name": meta["alcohol_name"],
                "concentration": meta["concentration"],
                "sensor_name": meta["sensor_name"],
                "features": features[i, :feat_len],
            }
            samples.append(
                SampleRecord(
                    sample_id=meta["sample_id"],
                    path=self.cache_dir / "features.npy",
                    target=target,
                    meta=sample_meta,
                )
            )
        return samples

    def _load_sample(self, record: SampleRecord) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load a single sample.

        Returns:
            Tuple of (features, target) where:
            - features: numpy array of sensor readings
            - target: dict with keys 'alcohol', 'concentration', 'sensor'
        """
        features = record.meta["features"]
        if not isinstance(features, np.ndarray):
            features = np.array(features, dtype=np.float32)
        return features, dict(record.target)

    @property
    def targets(self) -> List[int]:
        return [r.target["alcohol"] for r in self._samples]

    def extra_repr(self) -> str:
        base = super().extra_repr()
        parts = [base] if base else []
        parts.append(f"classes={len(self.classes)}")
        parts.append(f"sensors={len(self.sensors)}")
        parts.append(f"cache={self._cache}")
        return "\n".join(parts)
