"""Gas Sensor Array under Dynamic Gas Mixtures Dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from ._base import BaseEnoseDataset, SampleRecord
from ._info import get_dataset_info
from ._utils import download_and_extract


class GasSensorDynamic(BaseEnoseDataset):
    """Gas Sensor Array under Dynamic Gas Mixtures (UCI ML Repository, id=322).

    Source:
        https://archive.ics.uci.edu/dataset/322/gas+sensor+array+under+dynamic+gas+mixtures

    Dataset summary (from the UCI page):
        Time series from 16 chemical sensors exposed to gas mixtures at varying concentration levels.
        ~4.2 million instances, 19 features per instance.

    Dataset Information (UCI page):
        Two gas mixtures were generated:
        - Ethylene and Methane in air
        - Ethylene and CO in air

        Each measurement was constructed by continuous acquisition of 16-sensor array signals
        for approximately 12 hours without interruption.

        Sensor array: 16 MOX sensors from Figaro Inc., 4 different types (4 units each):
        - TGS-2600, TGS-2602, TGS-2610, TGS-2620

        Operating conditions:
        - Operating voltage: 5V constant
        - Sampling frequency: 100 Hz
        - Measurement chamber: 60 ml
        - Gas flow rate: 300 ml/min

        Concentration ranges:
        - Ethylene: 0-20 ppm
        - CO: 0-600 ppm
        - Methane: 0-300 ppm

        Concentration transitions set at random times (80-120s intervals) to random levels.
        All possible transitions are present in the data.

    Variable Information (UCI page):
        Two files:
        - ethylene_CO.txt: Ethylene + CO mixture
        - ethylene_methane.txt: Ethylene + Methane mixture

        19 columns:
        1. Time (seconds)
        2. Methane/CO concentration (ppm)
        3. Ethylene concentration (ppm)
        4-19. Sensor readings (16 channels)

        Sensor order:
        TGS2602, TGS2602, TGS2600, TGS2600, TGS2610, TGS2610, TGS2620, TGS2620,
        TGS2602, TGS2602, TGS2600, TGS2600, TGS2610, TGS2610, TGS2620, TGS2620

        Conversion to KOhms: 40.000 / S_i, where S_i is the value in the file.

    Introductory Paper:
        Fonollosa et al. "Reservoir Computing compensates slow response of chemosensor arrays
        exposed to fast varying gas concentrations in continuous monitoring",
        Sensors and Actuators B, 2015.

    DOI:
        https://doi.org/10.24432/C5WP4C

    License:
        CC BY 4.0

    Args:
        root: Root directory where the dataset is stored or will be downloaded.
        split: Optional split name (not used currently).
        download: If True, download the dataset if not found.
        cache: If True, cache processed data as .npy files for faster loading.
        mixture: Which mixture to load: "ethylene_co", "ethylene_methane", or "both".
        segment_on_change: If True, segment data when concentration changes.
        transforms: Optional transforms to apply to (data, target) pairs.
        transform: Optional transform to apply to data only.
        target_transform: Optional transform to apply to target only.
    """

    name = "gas_sensor_array_under_dynamic_gas_mixtures"

    sensor_types = [
        "TGS2602", "TGS2602", "TGS2600", "TGS2600",
        "TGS2610", "TGS2610", "TGS2620", "TGS2620",
        "TGS2602", "TGS2602", "TGS2600", "TGS2600",
        "TGS2610", "TGS2610", "TGS2620", "TGS2620",
    ]

    def __init__(
        self,
        root: Union[str, Path],
        *,
        split: Optional[str] = None,
        download: bool = False,
        cache: bool = True,
        mixture: str = "both",
        segment_on_change: bool = True,
        transforms=None,
        transform=None,
        target_transform=None,
    ):
        self._cache = cache
        self._mixture = mixture.lower()
        self._segment_on_change = segment_on_change

        if self._mixture not in ("ethylene_co", "ethylene_methane", "both"):
            raise ValueError(f"mixture must be 'ethylene_co', 'ethylene_methane', or 'both', got {mixture}")

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
        return self.dataset_dir / "cache" / self._mixture
    
    @property
    def samples_cache_dir(self) -> Path:
        """Directory for cached sample npy files."""
        return self.cache_dir / "samples"

    def download(self) -> None:
        info = get_dataset_info(self.name)
        download_and_extract(info, self.dataset_dir, force=False, verify=True)

    def _check_exists(self) -> bool:
        raw = self.raw_dir
        if not raw.exists():
            return False
        # Check for the main data files
        co_file = raw / "ethylene_CO.txt"
        me_file = raw / "ethylene_methane.txt"
        return co_file.exists() or me_file.exists()

    def _get_data_files(self) -> List[Tuple[Path, str]]:
        """Get list of (path, mixture_type) tuples."""
        files = []
        if self._mixture in ("ethylene_co", "both"):
            co_file = self.raw_dir / "ethylene_CO.txt"
            if co_file.exists():
                files.append((co_file, "ethylene_co"))
        if self._mixture in ("ethylene_methane", "both"):
            me_file = self.raw_dir / "ethylene_methane.txt"
            if me_file.exists():
                files.append((me_file, "ethylene_methane"))
        return files

    def _segment_data(self, df: pd.DataFrame, mixture_type: str) -> List[Dict[str, Any]]:
        """Segment data based on concentration changes."""
        segments = []

        # Column names
        time_col = df.columns[0]
        gas2_col = df.columns[1]  # CO or Methane
        ethylene_col = df.columns[2]

        # Find concentration change points
        gas2_vals = df[gas2_col].values
        eth_vals = df[ethylene_col].values

        gas2_changes = np.where(np.diff(gas2_vals) != 0)[0] + 1
        eth_changes = np.where(np.diff(eth_vals) != 0)[0] + 1
        change_points = np.unique(np.concatenate([[0], gas2_changes, eth_changes, [len(df)]]))

        for i in range(len(change_points) - 1):
            start_idx = change_points[i]
            end_idx = change_points[i + 1]

            if end_idx - start_idx < 10:  # Skip very short segments
                continue

            segment_df = df.iloc[start_idx:end_idx].copy()

            gas2_ppm = float(segment_df[gas2_col].iloc[0])
            eth_ppm = float(segment_df[ethylene_col].iloc[0])

            segments.append({
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "mixture_type": mixture_type,
                "gas2_name": "CO" if mixture_type == "ethylene_co" else "Methane",
                "gas2_ppm": gas2_ppm,
                "ethylene_ppm": eth_ppm,
            })

        return segments

    def _make_dataset(self) -> List[SampleRecord]:
        # Try to load from cache
        if self._cache and self.cache_dir.exists():
            meta_file = self.cache_dir / "metadata.npy"
            if meta_file.exists():
                return self._load_from_cache()

        data_files = self._get_data_files()
        samples: List[SampleRecord] = []

        for file_path, mixture_type in data_files:
            if self._segment_on_change:
                # Read file to segment
                df = pd.read_csv(file_path, sep=r"\s+", skiprows=1, header=None)

                segments = self._segment_data(df, mixture_type)

                for seg_idx, seg in enumerate(segments):
                    target = {
                        "mixture": 0 if mixture_type == "ethylene_co" else 1,
                        "gas2_ppm": seg["gas2_ppm"],
                        "ethylene_ppm": seg["ethylene_ppm"],
                    }
                    meta = {
                        "mixture_type": seg["mixture_type"],
                        "gas2_name": seg["gas2_name"],
                        "start_idx": seg["start_idx"],
                        "end_idx": seg["end_idx"],
                    }

                    sample_id = f"{mixture_type}_{seg_idx}_{seg['gas2_ppm']:.1f}_{seg['ethylene_ppm']:.1f}"
                    samples.append(
                        SampleRecord(
                            sample_id=sample_id,
                            path=file_path,
                            target=target,
                            meta=meta,
                        )
                    )
            else:
                # Whole file as single sample
                target = {
                    "mixture": 0 if mixture_type == "ethylene_co" else 1,
                    "gas2_ppm": None,
                    "ethylene_ppm": None,
                }
                meta = {
                    "mixture_type": mixture_type,
                    "gas2_name": "CO" if mixture_type == "ethylene_co" else "Methane",
                    "start_idx": None,
                    "end_idx": None,
                }
                samples.append(
                    SampleRecord(
                        sample_id=mixture_type,
                        path=file_path,
                        target=target,
                        meta=meta,
                    )
                )

        if self._cache and samples:
            self._save_to_cache(samples, data_files)

        return samples

    def _save_to_cache(self, samples: List[SampleRecord], data_files: List[Tuple[Path, str]]) -> None:
        """Save metadata and sample data to cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.samples_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load full data files once and extract all segments
        print(f"Caching {len(samples)} samples to npy files...")
        
        # Group samples by source file
        file_samples: Dict[str, List[Tuple[int, SampleRecord]]] = {}
        for idx, s in enumerate(samples):
            file_key = s.meta["mixture_type"]
            if file_key not in file_samples:
                file_samples[file_key] = []
            file_samples[file_key].append((idx, s))
        
        # Process each file once
        for file_path, mixture_type in data_files:
            if mixture_type not in file_samples:
                continue
            
            print(f"  Loading {file_path.name}...")
            df = pd.read_csv(file_path, sep=r"\s+", skiprows=1, header=None)
            
            gas2_name = "CO" if mixture_type == "ethylene_co" else "Methane"
            cols = ["time_s", f"{gas2_name.lower()}_ppm", "ethylene_ppm"] + [f"sensor_{i}" for i in range(16)]
            df.columns = cols[:df.shape[1]]
            
            # Extract sensor columns only
            sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
            
            for idx, s in tqdm(file_samples[mixture_type], desc=f"  Caching {mixture_type}"):
                start_idx = s.meta["start_idx"]
                end_idx = s.meta["end_idx"]
                
                segment = df.iloc[start_idx:end_idx][sensor_cols].values.astype(np.float32)
                # Note: Raw data has mixed scales (first 8 sensors ~[-50,5], last 8 ~[6000,55000])
                # Normalization will be handled in get_normalized_sample
                npy_path = self.samples_cache_dir / f"{s.sample_id}.npy"
                np.save(npy_path, segment)
        
        # Save metadata
        metadata = []
        for s in samples:
            metadata.append({
                "sample_id": s.sample_id,
                "path": str(self.samples_cache_dir / f"{s.sample_id}.npy"),
                "target": s.target,
                "meta": s.meta,
            })
        np.save(self.cache_dir / "metadata.npy", np.array(metadata, dtype=object))
        print(f"  Cached {len(samples)} samples.")

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
            - data: np.ndarray [T, C] sensor readings
            - target: dict with mixture info
        """
        path = Path(record.path)
        
        # Check if it's a cached npy file
        if path.suffix == ".npy":
            data = np.load(path)
            return data, dict(record.target)
        
        # Fallback: load from raw file (slow path)
        df = pd.read_csv(path, sep=r"\s+", skiprows=1, header=None)

        gas2_name = record.meta.get("gas2_name", "CO")
        cols = ["time_s", f"{gas2_name.lower()}_ppm", "ethylene_ppm"] + [f"sensor_{i}" for i in range(16)]
        df.columns = cols[:df.shape[1]]

        # Slice if segment indices provided
        start_idx = record.meta.get("start_idx")
        end_idx = record.meta.get("end_idx")
        if start_idx is not None and end_idx is not None:
            df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        # Extract sensor columns only
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
        data = df[sensor_cols].values.astype(np.float32)
        
        return data, dict(record.target)

    @staticmethod
    def convert_to_kohm(value: float) -> float:
        """Convert raw sensor value to resistance in KOhm."""
        if value == 0:
            return float("inf")
        return 40.0 / value
    
    @staticmethod
    def convert_to_kohm_array(arr: np.ndarray) -> np.ndarray:
        """Convert raw sensor array to resistance in KOhm.
        
        Formula: Rs(KOhm) = 40 / S, where S is the raw value.
        """
        # Handle zeros to avoid division by zero
        arr = np.where(arr == 0, np.nan, arr)
        return 40.0 / arr

    def extra_repr(self) -> str:
        base = super().extra_repr()
        parts = [base] if base else []
        parts.append(f"mixture={self._mixture}")
        parts.append(f"segment_on_change={self._segment_on_change}")
        parts.append(f"cache={self._cache}")
        return "\n".join(parts)
