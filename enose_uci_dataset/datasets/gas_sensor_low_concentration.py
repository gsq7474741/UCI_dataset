"""Gas Sensor Array Low-Concentration Dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ._base import BaseEnoseDataset, SampleRecord
from ._info import get_dataset_info
from ._utils import download_and_extract


class GasSensorLowConcentration(BaseEnoseDataset):
    """Gas Sensor Array Low-Concentration Dataset (UCI ML Repository, id=1081).

    Source:
        https://archive.ics.uci.edu/dataset/1081/gas+sensor+array+low-concentration

    Dataset summary (from the UCI page):
        Response signals from a 10-sensor array for 6 gases at low concentrations (ppb level).
        90 instances, each with 9000 data points (900 points × 10 sensors).

    Dataset Information (UCI page):
        6 gases collected at 3 concentration levels:
        - Ethanol
        - Acetone
        - Toluene
        - Ethyl acetate
        - Isopropanol
        - n-Hexane

        Concentration levels: 50 ppb, 100 ppb, 200 ppb
        5 samples per gas and concentration = 90 total samples

        Collection process:
        - Baseline: 5 minutes
        - Injection: 10 minutes
        - Cleaning: 15 minutes
        - Sampling frequency: 1 Hz
        - Dataset provides baseline + injection stages (15 min = 900 points per sensor)

    Sensor Array (10 sensors):
        1. TGS2603
        2. TGS2630
        3. TGS813
        4. TGS822
        5. MQ-135
        6. MQ-137
        7. MQ-138
        8. 2M012
        9. VOCS-P
        10. 2SH12

    Variable Information (UCI page):
        In gsalc.csv:
        - Column 1: Gas label (1-6)
        - Column 2: Concentration label (1-3 for 50/100/200 ppb)
        - Columns 3+: Sensor array responses (9000 points total)

        Data is concatenated: first 900 points = TGS2603, next 900 = TGS2630, etc.

    Introductory Paper:
        Zhao et al. "Feature Ensemble Learning for Sensor Array Data Classification
        Under Low-Concentration Gas", IEEE Transactions on Instrumentation and Measurement, 2023.

    DOI:
        https://doi.org/10.24432/C5CK6F

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

    name = "gas_sensor_array_low_concentration"

    classes = ["ethanol", "acetone", "toluene", "ethyl_acetate", "isopropanol", "n_hexane"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    # Map string gas names to indices (case insensitive)
    _gas_name_map = {
        "ethanol": 0, "acetone": 1, "toluene": 2,
        "ethyl_acetate": 3, "ethyl acetate": 3,
        "isopropanol": 4, "n_hexane": 5, "n-hexane": 5, "hexane": 5,
    }

    # Map concentration strings to ppb values
    _conc_map = {
        "50ppb": 50, "100ppb": 100, "200ppb": 200,
        "50": 50, "100": 100, "200": 200,
    }

    sensor_types = [
        "TGS2603", "TGS2630", "TGS813", "TGS822", "MQ-135",
        "MQ-137", "MQ-138", "2M012", "VOCS-P", "2SH12"
    ]
    num_sensors = 10
    points_per_sensor = 900

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
        csv_files = list(raw.glob("*.csv")) + list(raw.glob("**/*.csv"))
        return len(csv_files) > 0

    def _find_data_file(self) -> Path:
        """Find the main data CSV file."""
        # Look for gsalc.csv or similar
        for pattern in ["gsalc.csv", "*.csv"]:
            files = list(self.raw_dir.glob(pattern))
            if files:
                return files[0]
            # Check subdirectories
            for subdir in self.raw_dir.iterdir():
                if subdir.is_dir():
                    files = list(subdir.glob(pattern))
                    if files:
                        return files[0]
        raise FileNotFoundError(f"No CSV file found in {self.raw_dir}")

    def _make_dataset(self) -> List[SampleRecord]:
        # Try to load from cache
        if self._cache and self.cache_dir.exists():
            cache_file = self.cache_dir / "data.npy"
            meta_file = self.cache_dir / "metadata.npy"
            if cache_file.exists() and meta_file.exists():
                return self._load_from_cache()

        data_path = self._find_data_file()
        df = pd.read_csv(data_path, header=None)

        samples: List[SampleRecord] = []
        all_data = []

        for idx, row in df.iterrows():
            # First column is gas name (string), second is concentration
            gas_name = str(row.iloc[0]).strip().lower()
            conc_str = str(row.iloc[1]).strip().lower()

            # Map gas name to index
            gas_idx = self._gas_name_map.get(gas_name)
            if gas_idx is None:
                continue

            # Map concentration string to ppb value
            conc_ppb = self._conc_map.get(conc_str, 0)
            if conc_ppb == 0:
                # Try to extract number from string
                import re
                match = re.search(r'(\d+)', conc_str)
                if match:
                    conc_ppb = int(match.group(1))

            # Get sensor data (columns 2 onwards)
            sensor_data = row.iloc[2:].values.astype(np.float32)

            # Reshape to (10 sensors, 900 time points)
            expected_len = self.num_sensors * self.points_per_sensor
            if len(sensor_data) >= expected_len:
                sensor_data = sensor_data[:expected_len].reshape(self.num_sensors, self.points_per_sensor)
            else:
                # Pad if necessary
                padded = np.zeros((self.num_sensors, self.points_per_sensor), dtype=np.float32)
                actual_per_sensor = len(sensor_data) // self.num_sensors
                for s in range(self.num_sensors):
                    start = s * actual_per_sensor
                    end = start + actual_per_sensor
                    padded[s, :actual_per_sensor] = sensor_data[start:end]
                sensor_data = padded

            target = {
                "gas": gas_idx,
                "concentration_ppb": conc_ppb,
            }
            meta = {
                "gas_name": self.idx_to_class[gas_idx],
                "concentration_ppb": conc_ppb,
                "data_idx": len(all_data),
            }

            sample_id = f"{idx}_{meta['gas_name']}_{conc_ppb}ppb"
            samples.append(
                SampleRecord(
                    sample_id=sample_id,
                    path=data_path,
                    target=target,
                    meta=meta,
                )
            )
            all_data.append(sensor_data)

        if self._cache and samples:
            self._save_to_cache(samples, np.stack(all_data, axis=0))

        return samples

    def _save_to_cache(self, samples: List[SampleRecord], data: np.ndarray) -> None:
        """Save processed samples to cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        np.save(self.cache_dir / "data.npy", data)

        metadata = []
        for s in samples:
            metadata.append({
                "sample_id": s.sample_id,
                "gas": s.target["gas"],
                "concentration_ppb": s.target["concentration_ppb"],
                "gas_name": s.meta["gas_name"],
                "data_idx": s.meta["data_idx"],
            })
        np.save(self.cache_dir / "metadata.npy", np.array(metadata, dtype=object))

    def _load_from_cache(self) -> List[SampleRecord]:
        """Load samples from cache."""
        data = np.load(self.cache_dir / "data.npy")
        metadata = np.load(self.cache_dir / "metadata.npy", allow_pickle=True)

        samples = []
        for meta in metadata:
            target = {
                "gas": meta["gas"],
                "concentration_ppb": meta["concentration_ppb"],
            }
            sample_meta = {
                "gas_name": meta["gas_name"],
                "concentration_ppb": meta["concentration_ppb"],
                "data_idx": meta["data_idx"],
                "_cached_data": data[meta["data_idx"]],
            }
            samples.append(
                SampleRecord(
                    sample_id=meta["sample_id"],
                    path=self.cache_dir / "data.npy",
                    target=target,
                    meta=sample_meta,
                )
            )
        return samples

    def _load_sample(self, record: SampleRecord) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load a single sample.

        Returns:
            Tuple of (data, target) where:
            - data: numpy array of shape (10, 900) - 10 sensors, 900 time points each
            - target: dict with gas class and concentration
        """
        if "_cached_data" in record.meta:
            data = record.meta["_cached_data"]
        else:
            # Load from original file
            data_path = self._find_data_file()
            df = pd.read_csv(data_path, header=None)
            row = df.iloc[record.meta["data_idx"]]
            sensor_data = row.iloc[2:].values.astype(np.float32)
            expected_len = self.num_sensors * self.points_per_sensor
            if len(sensor_data) >= expected_len:
                data = sensor_data[:expected_len].reshape(self.num_sensors, self.points_per_sensor)
            else:
                data = np.zeros((self.num_sensors, self.points_per_sensor), dtype=np.float32)

        return data, dict(record.target)

    @property
    def targets(self) -> List[int]:
        return [r.target["gas"] for r in self._samples]

    def extra_repr(self) -> str:
        base = super().extra_repr()
        parts = [base] if base else []
        parts.append(f"classes={len(self.classes)}")
        parts.append(f"sensors={self.num_sensors}")
        parts.append(f"cache={self._cache}")
        return "\n".join(parts)
