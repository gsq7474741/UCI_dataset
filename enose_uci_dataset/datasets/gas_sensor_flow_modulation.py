"""Gas Sensor Array under Flow Modulation Dataset."""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ._base import BaseEnoseDataset, SampleRecord
from ._info import get_dataset_info
from ._utils import download_and_extract


class GasSensorFlowModulation(BaseEnoseDataset):
    """Gas Sensor Array under Flow Modulation (UCI ML Repository, id=308).

    Source:
        https://archive.ics.uci.edu/dataset/308/gas+sensor+array+under+flow+modulation

    Dataset summary (from the UCI page):
        Raw time series and extracted features from 16 MOX gas sensors exposed to
        acetone and ethanol mixtures under flow modulation conditions.
        58 instances with 120,000 raw data points each.

    Dataset Information (UCI page):
        The dataset is organized in two CSV files:
        - rawdata.csv.gz (4.5 MB): Raw time series data
        - features.csv (200 kB): Extracted features

        Raw data:
        - Each sample contains 16 time-series (one per sensor)
        - Recording duration: 5 minutes at 25 Hz (7500 data points per time-series)
        - Total attributes per sample: 120,000 (16 sensors × 7500 points)

        Feature data:
        - 3 types of features extracted from each time-series:
          1. Maximum features: 1 per sensor
          2. High-frequency features: 13 per sensor (first 13 respiration cycles)
          3. Low-frequency features: 13 per sensor
        - Total attributes per sample: 432 (16 sensors × 27 features)

        Gas classes:
        - Pure acetone
        - Pure ethanol
        - Mixture of acetone and ethanol
        - Air (background)

        Concentration range: 0-1 vol.% for both acetone and ethanol.

    Variable Information (UCI page):
        Common attributes in both files:
        - exp: experiment number (100-181)
        - batch: batch identifier (5 values)
        - ace_conc: acetone concentration (vol.%, 0-1)
        - eth_conc: ethanol concentration (vol.%, 0-1)
        - lab: class label (12 values)
        - gas: simplified class (4 values: pure analytes, mixture, or air)
        - col: color code for plotting

        Raw data specific:
        - sensor: sensor number (1-16)
        - sample: sample number (1-58)
        - dR_t<m>: time series value at time instant m (1-7500)

        Feature data specific:
        - S<j>_max: maximum feature from sensor j
        - S<j>_r<k>_Alf: low-frequency feature from sensor j at respiration k
        - S<j>_r<k>_Ahf: high-frequency feature from sensor j at respiration k

    DOI:
        https://doi.org/10.24432/C5BG7G

    License:
        CC BY 4.0

    Args:
        root: Root directory where the dataset is stored or will be downloaded.
        split: Optional split name (not used currently).
        download: If True, download the dataset if not found.
        cache: If True, cache processed data as .npy files for faster loading.
        use_features: If True, load extracted features; otherwise load raw time series.
        transforms: Optional transforms to apply to (data, target) pairs.
        transform: Optional transform to apply to data only.
        target_transform: Optional transform to apply to target only.
    """

    name = "gas_sensor_array_under_flow_modulation"

    gas_classes = ["air", "acetone", "ethanol", "mixture"]
    class_to_idx = {c: i for i, c in enumerate(gas_classes)}

    def __init__(
        self,
        root: Union[str, Path],
        *,
        split: Optional[str] = None,
        download: bool = False,
        cache: bool = True,
        use_features: bool = False,
        transforms=None,
        transform=None,
        target_transform=None,
    ):
        self._cache = cache
        self._use_features = use_features
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
        suffix = "features" if self._use_features else "raw"
        return self.dataset_dir / "cache" / suffix

    def download(self) -> None:
        info = get_dataset_info(self.name)
        download_and_extract(info, self.dataset_dir, force=False, verify=True)

    def _check_exists(self) -> bool:
        raw = self.raw_dir
        if not raw.exists():
            return False
        # Check for the data files
        rawdata = raw / "rawdata.csv.gz"
        features = raw / "features.csv"
        return rawdata.exists() or features.exists()

    def _load_features_file(self) -> pd.DataFrame:
        """Load features.csv file."""
        features_path = self.raw_dir / "features.csv"
        if features_path.exists():
            return pd.read_csv(features_path)
        raise FileNotFoundError(f"features.csv not found in {self.raw_dir}")

    def _load_rawdata_file(self) -> pd.DataFrame:
        """Load rawdata.csv.gz file."""
        rawdata_path = self.raw_dir / "rawdata.csv.gz"
        if rawdata_path.exists():
            return pd.read_csv(rawdata_path, compression="gzip")
        # Try uncompressed version
        rawdata_path = self.raw_dir / "rawdata.csv"
        if rawdata_path.exists():
            return pd.read_csv(rawdata_path)
        raise FileNotFoundError(f"rawdata.csv[.gz] not found in {self.raw_dir}")

    def _make_dataset(self) -> List[SampleRecord]:
        # Try to load from cache
        if self._cache and self.cache_dir.exists():
            meta_file = self.cache_dir / "metadata.npy"
            if meta_file.exists():
                return self._load_from_cache()

        samples: List[SampleRecord] = []

        if self._use_features:
            df = self._load_features_file()
            data_path = self.raw_dir / "features.csv"
        else:
            df = self._load_rawdata_file()
            data_path = self.raw_dir / "rawdata.csv.gz"

        # Group by sample/experiment
        if self._use_features:
            # Features file has one row per sample
            for idx, row in df.iterrows():
                exp = row.get("exp", idx)
                batch = row.get("batch", "unknown")
                ace_conc = row.get("ace_conc", 0.0)
                eth_conc = row.get("eth_conc", 0.0)
                gas = str(row.get("gas", "unknown")).lower()
                lab = row.get("lab", "unknown")

                # Determine gas class
                gas_idx = self.class_to_idx.get(gas, -1)
                if gas_idx < 0:
                    if ace_conc > 0 and eth_conc > 0:
                        gas_idx = self.class_to_idx["mixture"]
                    elif ace_conc > 0:
                        gas_idx = self.class_to_idx["acetone"]
                    elif eth_conc > 0:
                        gas_idx = self.class_to_idx["ethanol"]
                    else:
                        gas_idx = self.class_to_idx["air"]

                target = {
                    "gas": gas_idx,
                    "ace_conc": float(ace_conc),
                    "eth_conc": float(eth_conc),
                }
                meta = {
                    "exp": exp,
                    "batch": batch,
                    "lab": lab,
                    "row_idx": int(idx),
                }

                sample_id = f"exp{exp}_{batch}_{lab}"
                samples.append(
                    SampleRecord(
                        sample_id=sample_id,
                        path=data_path,
                        target=target,
                        meta=meta,
                    )
                )
        else:
            # Raw data file has 16 rows per sample (one per sensor)
            unique_samples = df.groupby("sample").first().reset_index()
            for _, row in unique_samples.iterrows():
                sample_num = int(row["sample"])
                exp = row.get("exp", sample_num)
                batch = row.get("batch", "unknown")
                ace_conc = row.get("ace_conc", 0.0)
                eth_conc = row.get("eth_conc", 0.0)
                gas = str(row.get("gas", "unknown")).lower()
                lab = row.get("lab", "unknown")

                gas_idx = self.class_to_idx.get(gas, -1)
                if gas_idx < 0:
                    if ace_conc > 0 and eth_conc > 0:
                        gas_idx = self.class_to_idx["mixture"]
                    elif ace_conc > 0:
                        gas_idx = self.class_to_idx["acetone"]
                    elif eth_conc > 0:
                        gas_idx = self.class_to_idx["ethanol"]
                    else:
                        gas_idx = self.class_to_idx["air"]

                target = {
                    "gas": gas_idx,
                    "ace_conc": float(ace_conc),
                    "eth_conc": float(eth_conc),
                }
                meta = {
                    "exp": exp,
                    "batch": batch,
                    "lab": lab,
                    "sample_num": sample_num,
                }

                sample_id = f"sample{sample_num}_exp{exp}_{batch}"
                samples.append(
                    SampleRecord(
                        sample_id=sample_id,
                        path=data_path,
                        target=target,
                        meta=meta,
                    )
                )

        if self._cache and samples:
            self._save_to_cache(samples)

        return samples

    def _save_to_cache(self, samples: List[SampleRecord]) -> None:
        """Save metadata to cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        metadata = []
        for s in samples:
            metadata.append({
                "sample_id": s.sample_id,
                "path": str(s.path),
                "target": s.target,
                "meta": s.meta,
            })
        np.save(self.cache_dir / "metadata.npy", np.array(metadata, dtype=object))

    def _load_from_cache(self) -> List[SampleRecord]:
        """Load samples from cache."""
        metadata = np.load(self.cache_dir / "metadata.npy", allow_pickle=True)

        samples = []
        for meta in metadata:
            samples.append(
                SampleRecord(
                    sample_id=meta["sample_id"],
                    path=Path(meta["path"]),
                    target=meta["target"],
                    meta=meta["meta"],
                )
            )
        return samples

    def _load_sample(self, record: SampleRecord) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load a single sample.

        Returns:
            Tuple of (data, target) where:
            - data: numpy array of features or raw time series (16 sensors × time points)
            - target: dict with gas class and concentrations
        """
        if self._use_features:
            df = self._load_features_file()
            row_idx = record.meta["row_idx"]
            row = df.iloc[row_idx]

            # Extract feature columns (S<j>_max, S<j>_r<k>_Alf, S<j>_r<k>_Ahf)
            feature_cols = [c for c in df.columns if c.startswith("S") and ("_max" in c or "_r" in c)]
            features = row[feature_cols].values.astype(np.float32)

            return features, dict(record.target)
        else:
            df = self._load_rawdata_file()
            sample_num = record.meta["sample_num"]

            # Get all 16 sensor rows for this sample
            sample_data = df[df["sample"] == sample_num].sort_values("sensor")

            # Extract time series columns (dR_t1, dR_t2, ..., dR_t7500)
            ts_cols = [c for c in df.columns if c.startswith("dR_t")]
            ts_cols = sorted(ts_cols, key=lambda x: int(x.split("dR_t")[1]))

            # Create array of shape (16, 7500)
            data = sample_data[ts_cols].values.astype(np.float32)

            return data, dict(record.target)

    @property
    def targets(self) -> List[int]:
        return [r.target["gas"] for r in self._samples]

    def extra_repr(self) -> str:
        base = super().extra_repr()
        parts = [base] if base else []
        parts.append(f"use_features={self._use_features}")
        parts.append(f"cache={self._cache}")
        return "\n".join(parts)
