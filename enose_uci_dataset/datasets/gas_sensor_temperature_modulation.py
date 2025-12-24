"""Gas Sensor Array Temperature Modulation Dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ._base import BaseEnoseDataset, SampleRecord
from ._info import get_dataset_info
from ._utils import download_and_extract


class GasSensorTemperatureModulation(BaseEnoseDataset):
    """Gas Sensor Array Temperature Modulation (UCI ML Repository, id=487).

    Source:
        https://archive.ics.uci.edu/dataset/487/gas+sensor+array+temperature+modulation

    Dataset summary (from the UCI page):
        A chemical detection platform composed of 14 temperature-modulated MOX gas sensors
        was exposed to dynamic mixtures of CO and humid synthetic air.
        ~4.1 million instances.

    Dataset Information (UCI page):
        Chemical detection platform:
        - 14 MOX gas sensors: 7 units of Figaro TGS 3870-A04, 7 units of FIS SB-500-12
        - Temperature modulated via heater voltage (0.2-0.9V) in cycles of 20 and 25 seconds
        - Sensors pre-heated for one week before experiments
        - MOX read-out circuits: voltage dividers with 1 MOhm load resistors, powered at 5V
        - Sampling rate: 3.5 Hz, 15 bits precision

        Gas delivery:
        - Dynamic mixtures of CO and humid synthetic air
        - PTFE test chamber: 250 cm³ internal volume
        - Mass flow controllers (MFC): EL-FLOW Select, Bronkhorst
        - Full scale flow rates: 1000 mln/min (air), 3 mln/min (CO)
        - CO bottle: 1600 ppm CO in synthetic air
        - Humidity: 15-75% r.h.

        Temperature/humidity sensor: SHT75 (Sensirion), sampled every 5s
        Temperature variations: < 3°C per experiment

        Experimental protocol:
        - 100 measurements per experiment: 10 concentrations (0-20 ppm) × 10 replicates
        - Each replicate: random humidity (15-75% r.h.)
        - Gas chamber cleaning: 15 min with synthetic air (240 mln/min)
        - Each measurement: 15 min at 240 mln/min
        - Total experiment duration: ~25 hours
        - 13 experiment days over 17 natural days

    Variable Information (UCI page):
        13 text files, one per measurement day.
        Filename: timestamp (yyyymmdd_HHMMSS) of measurement start.

        20 columns:
        1. Time (s)
        2. CO concentration (ppm)
        3. Humidity (%r.h.)
        4. Temperature (°C)
        5. Flow rate (mL/min)
        6. Heater voltage (V)
        7-20. Sensor resistances R1-R14 (MOhm)

        R1-R7: Figaro TGS 3870-A04 sensors
        R8-R14: FIS SB-500-12 sensors

        Sampling rate: 3.5 Hz

    DOI:
        https://doi.org/10.24432/C5PP56

    License:
        CC BY 4.0

    Args:
        root: Root directory where the dataset is stored or will be downloaded.
        split: Optional split name (not used currently).
        download: If True, download the dataset if not found.
        cache: If True, cache processed data as .npy files for faster loading.
        segment_by_concentration: If True, segment data by concentration changes.
        transforms: Optional transforms to apply to (data, target) pairs.
        transform: Optional transform to apply to data only.
        target_transform: Optional transform to apply to target only.
    """

    name = "gas_sensor_array_temperature_modulation"

    sensor_types = (
        ["TGS3870-A04"] * 7 +  # R1-R7
        ["SB-500-12"] * 7       # R8-R14
    )

    def __init__(
        self,
        root: Union[str, Path],
        *,
        split: Optional[str] = None,
        download: bool = False,
        cache: bool = True,
        segment_by_concentration: bool = False,
        transforms=None,
        transform=None,
        target_transform=None,
    ):
        self._cache = cache
        self._segment_by_concentration = segment_by_concentration
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
        # Check for CSV files (actual data format)
        csv_files = list(raw.glob("*.csv")) + list(raw.glob("**/*.csv"))
        return len(csv_files) > 0

    def _find_data_files(self) -> List[Path]:
        """Find all CSV data files."""
        files = list(self.raw_dir.glob("*.csv"))
        if not files:
            # Check subdirectories
            for subdir in self.raw_dir.iterdir():
                if subdir.is_dir():
                    files.extend(subdir.glob("*.csv"))
        # Filter out non-data files (keep only timestamp-named files like 20161005_140846.csv)
        data_files = []
        for f in files:
            # Check if filename looks like a timestamp (starts with digits, YYYYMMDD format)
            if f.stem[0].isdigit() and len(f.stem) >= 8 and '_' in f.stem:
                data_files.append(f)
        return sorted(data_files)

    def _segment_by_co_concentration(self, df: pd.DataFrame, day_idx: int) -> List[Dict[str, Any]]:
        """Segment data based on CO concentration changes."""
        segments = []

        co_col = df.columns[1]  # CO concentration column
        co_vals = df[co_col].values

        # Find change points (threshold for change)
        diffs = np.abs(np.diff(co_vals))
        change_threshold = 0.5  # ppm
        change_points = np.where(diffs > change_threshold)[0] + 1
        change_points = np.concatenate([[0], change_points, [len(df)]])

        for i in range(len(change_points) - 1):
            start_idx = change_points[i]
            end_idx = change_points[i + 1]

            if end_idx - start_idx < 100:  # Skip very short segments
                continue

            segment_df = df.iloc[start_idx:end_idx]
            co_ppm = float(segment_df[co_col].mean())
            humidity = float(segment_df[df.columns[2]].mean())

            segments.append({
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "day": day_idx,
                "co_ppm": co_ppm,
                "humidity": humidity,
            })

        return segments

    def _make_dataset(self) -> List[SampleRecord]:
        # Try to load from cache
        if self._cache and self.cache_dir.exists():
            meta_file = self.cache_dir / "metadata.npy"
            if meta_file.exists():
                return self._load_from_cache()

        data_files = self._find_data_files()
        samples: List[SampleRecord] = []

        for day_idx, data_path in enumerate(data_files):
            if self._segment_by_concentration:
                try:
                    df = pd.read_csv(data_path)  # CSV with header
                    segments = self._segment_by_co_concentration(df, day_idx)

                    for seg_idx, seg in enumerate(segments):
                        target = {
                            "day": seg["day"],
                            "co_ppm": seg["co_ppm"],
                            "humidity": seg["humidity"],
                        }
                        meta = {
                            "file": data_path.name,
                            "start_idx": seg["start_idx"],
                            "end_idx": seg["end_idx"],
                        }

                        sample_id = f"day{day_idx}_{seg_idx}_{seg['co_ppm']:.1f}ppm"
                        samples.append(
                            SampleRecord(
                                sample_id=sample_id,
                                path=data_path,
                                target=target,
                                meta=meta,
                            )
                        )
                except Exception:
                    continue
            else:
                # Whole file as single sample
                target = {
                    "day": day_idx,
                    "co_ppm": None,
                    "humidity": None,
                }
                meta = {
                    "file": data_path.name,
                    "start_idx": None,
                    "end_idx": None,
                }
                sample_id = f"day{day_idx}_{data_path.stem}"
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

    def _load_sample(self, record: SampleRecord) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load a single sample.

        Returns:
            Tuple of (data, target) where:
            - data: DataFrame with columns [time_s, co_ppm, humidity_pct, temp_c, flow_rate,
                    heater_v, sensor_0..sensor_13]
            - target: dict with day and concentration info
        """
        df = pd.read_csv(record.path)  # CSV with header

        # Rename columns to standardized names
        if len(df.columns) >= 20:
            df.columns = ["time_s", "co_ppm", "humidity_pct", "temp_c", "flow_rate", "heater_v"] + \
                         [f"sensor_{i}" for i in range(14)]

        # Slice if segment indices provided
        start_idx = record.meta.get("start_idx")
        end_idx = record.meta.get("end_idx")
        if start_idx is not None and end_idx is not None:
            df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        return df, dict(record.target)

    def extra_repr(self) -> str:
        base = super().extra_repr()
        parts = [base] if base else []
        parts.append(f"segment_by_concentration={self._segment_by_concentration}")
        parts.append(f"cache={self._cache}")
        return "\n".join(parts)
