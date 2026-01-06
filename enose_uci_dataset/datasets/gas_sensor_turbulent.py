"""Gas Sensor Array Exposed to Turbulent Gas Mixtures Dataset."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ._base import BaseEnoseDataset, SampleRecord
from ._info import get_dataset_info
from ._utils import download_and_extract


class GasSensorTurbulent(BaseEnoseDataset):
    """Gas Sensor Array Exposed to Turbulent Gas Mixtures (UCI ML Repository, id=309).

    Source:
        https://archive.ics.uci.edu/dataset/309/gas+sensor+array+exposed+to+turbulent+gas+mixtures

    Dataset summary (from the UCI page):
        A chemical detection platform composed of 8 chemo-resistive gas sensors was exposed to
        turbulent gas mixtures generated naturally in a wind tunnel. 180 instances with time series.

    Dataset Information (UCI page):
        The experimental setup was designed to test gas sensors in realistic environments.
        A wind tunnel with two independent gas sources generates two gas plumes that get
        naturally mixed along a turbulent flow.

        Chemical detection platform:
        - 8 MOX gas sensors from Figaro: TGS2611, TGS2612, TGS2610, TGS2600, TGS2602, TGS2620
        - Operating temperature controlled by built-in heater at constant 5V
        - Includes Temperature and Relative Humidity sensors
        - Sampling rate: 20 ms (50 Hz), also provided downsampled at 100 ms (10 Hz)

        Wind tunnel: 2.5m x 1.2m x 0.4m with two gas sources (source1 and source2).

        Gas mixtures:
        - Ethylene with Methane
        - Ethylene with Carbon Monoxide

        Each volatile released at 4 flow levels: zero (z), low (l), medium (m), high (h)
        - 15 mixtures of Ethylene with CO
        - 15 mixtures of Ethylene with Methane
        - Each configuration repeated 6 times → 180 measurements total

        Mean concentration levels at sensors' location:
        - Ethylene: l=31 ppm, m=46 ppm, h=96 ppm
        - CO: l=270 ppm, m=397 ppm, h=460 ppm
        - Methane: l=51 ppm, m=115 ppm, h=131 ppm

        Each measurement duration: 300 seconds
        - 0-60s: no gas (baseline)
        - 60-240s: gas release
        - 240-300s: recovery

    Variable Information (UCI page):
        180 text files, one per measurement.
        Filename format: XXX_Et_Y_GG_Z where:
        - XXX: local identifier
        - Y: Ethylene concentration (n=zero, L=Low, M=Medium, H=High)
        - GG: second gas (Me=Methane, CO=Carbon Monoxide)
        - Z: concentration level

        Columns: Time(s), Temperature(°C), Relative Humidity(%), 8 sensor readings
        Sensor order: TGS2600, TGS2602, TGS2602, TGS2620, TGS2612, TGS2620, TGS2611, TGS2610

        Conversion to resistance: Rs(KOhm) = 10 * (3110 - A) / A, where A is acquired value.

    DOI:
        https://doi.org/10.24432/C5JS5P

    License:
        CC BY 4.0

    Args:
        root: Root directory where the dataset is stored or will be downloaded.
        split: Optional split name (not used currently).
        download: If True, download the dataset if not found.
        cache: If True, cache processed data as .npy files for faster loading.
        use_downsampled: If True, use 100ms downsampled data instead of 20ms raw data.
        transforms: Optional transforms to apply to (data, target) pairs.
        transform: Optional transform to apply to data only.
        target_transform: Optional transform to apply to target only.
    """

    name = "gas_sensor_array_exposed_to_turbulent_gas_mixtures"

    sensor_types = ["TGS2600", "TGS2602", "TGS2602", "TGS2620", "TGS2612", "TGS2620", "TGS2611", "TGS2610"]

    concentration_levels = {"n": 0, "z": 0, "l": 1, "m": 2, "h": 3, "L": 1, "M": 2, "H": 3}
    level_names = {0: "zero", 1: "low", 2: "medium", 3: "high"}

    # Mean ppm values for each gas at each level
    gas_ppm = {
        "Ethylene": {"l": 31, "m": 46, "h": 96, "z": 0, "n": 0},
        "CO": {"l": 270, "m": 397, "h": 460, "z": 0, "n": 0},
        "Methane": {"l": 51, "m": 115, "h": 131, "z": 0, "n": 0},
    }

    _filename_re = re.compile(
        r"^(\d+)_Et_([nzlmhLMH])_(Me|CO)_([nzlmhLMH])(\.txt)?$",
        re.IGNORECASE
    )

    def __init__(
        self,
        root: Union[str, Path],
        *,
        split: Optional[str] = None,
        download: bool = False,
        cache: bool = True,
        use_downsampled: bool = True,
        transforms=None,
        transform=None,
        target_transform=None,
    ):
        self._cache = cache
        self._use_downsampled = use_downsampled
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
        suffix = "downsampled" if self._use_downsampled else "raw"
        return self.dataset_dir / "cache" / suffix

    def download(self) -> None:
        info = get_dataset_info(self.name)
        download_and_extract(info, self.dataset_dir, force=False, verify=True)

    def _check_exists(self) -> bool:
        raw = self.raw_dir
        if not raw.exists():
            return False
        # Check for data files (with or without .txt extension)
        data_files = list(raw.glob("**/???_Et_*"))
        return len(data_files) > 0

    def _find_data_dir(self) -> Path:
        """Find directory with data files, preferring downsampled if requested."""
        if self._use_downsampled:
            ds_dir = self.raw_dir / "dataset_twosources_downsampled"
            if ds_dir.exists() and list(ds_dir.glob("*_Et_*")):
                return ds_dir
        # Try raw data directory
        raw_data_dir = self.raw_dir / "dataset_twosources_raw"
        if raw_data_dir.exists() and list(raw_data_dir.glob("*_Et_*")):
            return raw_data_dir
        # Check subdirectories for any matching pattern
        for subdir in self.raw_dir.iterdir():
            if subdir.is_dir() and list(subdir.glob("*_Et_*")):
                return subdir
        return self.raw_dir

    def _parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Parse filename to extract gas configuration."""
        m = self._filename_re.match(filename)
        if not m:
            return None

        local_id = m.group(1)
        et_level = m.group(2).lower()
        second_gas = m.group(3)
        second_level = m.group(4).lower()

        return {
            "local_id": local_id,
            "ethylene_level": self.concentration_levels.get(et_level, 0),
            "ethylene_ppm": self.gas_ppm["Ethylene"].get(et_level, 0),
            "second_gas": second_gas,
            "second_level": self.concentration_levels.get(second_level, 0),
            "second_ppm": self.gas_ppm.get(second_gas, {}).get(second_level, 0),
        }

    def _make_dataset(self) -> List[SampleRecord]:
        data_dir = self._find_data_dir()

        # Try to load from cache
        if self._cache and self.cache_dir.exists():
            meta_file = self.cache_dir / "metadata.npy"
            if meta_file.exists():
                return self._load_from_cache()

        samples: List[SampleRecord] = []
        # Find all data files (with or without .txt extension)
        data_files = sorted([f for f in data_dir.iterdir() if f.is_file() and "_Et_" in f.name])

        for data_path in data_files:
            parsed = self._parse_filename(data_path.name)
            if parsed is None:
                continue

            target = {
                "ethylene_level": parsed["ethylene_level"],
                "ethylene_ppm": parsed["ethylene_ppm"],
                "second_gas": 0 if parsed["second_gas"] == "Me" else 1,  # 0=Methane, 1=CO
                "second_level": parsed["second_level"],
                "second_ppm": parsed["second_ppm"],
            }
            meta = {
                "local_id": parsed["local_id"],
                "ethylene_level_name": self.level_names[parsed["ethylene_level"]],
                "second_gas_name": "Methane" if parsed["second_gas"] == "Me" else "CO",
                "second_level_name": self.level_names[parsed["second_level"]],
                "file": data_path.name,
            }

            sample_id = data_path.stem
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
        """Save metadata to cache (data is loaded on demand)."""
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

    def _load_sample(self, record: SampleRecord) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load a single sample.

        Returns:
            Tuple of (data, target) where:
            - data: DataFrame with columns [time_s, temp_c, humidity_pct, sensor_0..sensor_7]
            - target: dict with gas configuration info
        """
        # Try comma separator first (downsampled files use comma)
        # Fall back to whitespace for raw files
        try:
            df = pd.read_csv(record.path, sep=",", header=None)
            if df.shape[1] == 1:
                # Single column means wrong separator, try whitespace
                df = pd.read_csv(record.path, sep=r"\s+", header=None)
        except Exception:
            df = pd.read_csv(record.path, sep=r"\s+", header=None)

        # Expected: time, temp, humidity, 8 sensors
        if df.shape[1] >= 11:
            df = df.iloc[:, :11]
            df.columns = ["time_s", "temp_c", "humidity_pct"] + [f"sensor_{i}" for i in range(8)]
        else:
            # Handle variable column counts
            cols = ["time_s", "temp_c", "humidity_pct"] + [f"sensor_{i}" for i in range(df.shape[1] - 3)]
            df.columns = cols[:df.shape[1]]

        return df, dict(record.target)

    @staticmethod
    def convert_to_resistance(value: float) -> float:
        """Convert raw sensor value to resistance in KOhm."""
        if value == 0:
            return float("inf")
        return 10.0 * (3110.0 - value) / value

    def extra_repr(self) -> str:
        base = super().extra_repr()
        parts = [base] if base else []
        parts.append(f"use_downsampled={self._use_downsampled}")
        parts.append(f"cache={self._cache}")
        return "\n".join(parts)
