"""Gas Sensor Array Drift Dataset at Different Concentrations."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ._base import BaseEnoseDataset, SampleRecord
from ._info import get_dataset_info
from ._utils import download_and_extract


class GasSensorArrayDrift(BaseEnoseDataset):
    """Gas Sensor Array Drift Dataset at Different Concentrations (UCI ML Repository, id=270).

    Source:
        https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations

    Dataset summary (from the UCI page):
        This data set contains 13,910 measurements from 16 chemical sensors exposed to 6 gases
        at different concentration levels.

    Dataset Information (UCI page):
        This dataset is an extension of the Gas Sensor Array Drift Dataset, providing information
        about the concentration level at which the sensors were exposed for each measurement.

        The dataset was gathered during the period of January 2008 to February 2011 (36 months)
        in a gas delivery platform facility situated at the ChemoSignals Laboratory in the
        BioCircuits Institute, University of California San Diego.

        The resulting dataset comprises recordings from six distinct pure gaseous substances:
        - Ammonia (concentration range: 50-1000 ppmv)
        - Acetaldehyde (concentration range: 5-500 ppmv)
        - Acetone (concentration range: 12-1000 ppmv)
        - Ethylene (concentration range: 10-300 ppmv)
        - Ethanol (concentration range: 10-600 ppmv)
        - Toluene (concentration range: 10-100 ppmv)

        Sensors: 16 metal oxide semiconductor (MOX) gas sensors.

    Variable Information (UCI page):
        The sensor responses are represented by a 128-dimensional feature vector (8 features × 16 sensors).
        Features include:
        - Steady-state feature (ΔR): maximum resistance change with respect to baseline
        - Normalized version: ratio of maximum resistance to baseline value
        - Exponential moving average (emaΔ): reflecting sensor dynamics

        Data format: {class};{concentration} 1:{sensor1_feature1} 2:{sensor1_feature2} ... 128:{sensor16_feature8}

        Gas class labels:
        1: Ethanol, 2: Ethylene, 3: Ammonia, 4: Acetaldehyde, 5: Acetone, 6: Toluene

    Introductory Paper:
        Rodríguez-Luján et al. "On the calibration of sensor arrays for pattern recognition
        using the minimal number of experiments", Chemometrics and Intelligent Laboratory Systems, 2014.

    DOI:
        https://doi.org/10.24432/C59K5F

    Args:
        root: Root directory where the dataset is stored or will be downloaded.
        split: Optional split name (not used currently).
        download: If True, download the dataset if not found.
        cache: If True, cache processed data as .npy files for faster loading.
        transforms: Optional transforms to apply to (data, target) pairs.
        transform: Optional transform to apply to data only.
        target_transform: Optional transform to apply to target only.
    """

    name = "gas_sensor_array_drift_dataset_at_different_concentrations"

    classes = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    # Mapping from original file labels (1-6) to our class indices (0-5)
    _label_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

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
        # Check for batch*.dat files
        dat_files = list(raw.glob("batch*.dat")) + list(raw.glob("Dataset/batch*.dat"))
        return len(dat_files) > 0

    def _get_batch_dir(self) -> Path:
        """Find the directory containing batch*.dat files."""
        if list(self.raw_dir.glob("batch*.dat")):
            return self.raw_dir
        dataset_dir = self.raw_dir / "Dataset"
        if dataset_dir.exists() and list(dataset_dir.glob("batch*.dat")):
            return dataset_dir
        return self.raw_dir

    def _parse_line(self, line: str, batch_idx: int) -> Optional[Dict[str, Any]]:
        """Parse a single line from batch file."""
        parts = line.strip().split()
        if not parts:
            return None

        try:
            label_conc = parts[0].split(";")
            gas_label = int(label_conc[0])
            concentration = float(label_conc[1])
        except (ValueError, IndexError):
            return None

        if gas_label not in self._label_map:
            return None

        # Parse 128 features (8 features × 16 sensors)
        features = []
        for feat in parts[1:]:
            try:
                idx_val = feat.split(":")
                features.append(float(idx_val[1]))
            except (ValueError, IndexError):
                continue

        if len(features) != 128:
            return None

        # Reshape to (8, 16) - 8 features per sensor, 16 sensors
        # Then transpose to (16, 8) for sensor-first layout
        feature_array = np.array(features, dtype=np.float32).reshape(8, 16).T

        return {
            "gas_idx": self._label_map[gas_label],
            "gas_name": self.idx_to_class[self._label_map[gas_label]],
            "concentration": concentration,
            "batch": batch_idx,
            "features": feature_array,
        }

    def _make_dataset(self) -> List[SampleRecord]:
        batch_dir = self._get_batch_dir()

        # Try to load from cache
        if self._cache and self.cache_dir.exists():
            cache_file = self.cache_dir / "samples.npy"
            meta_file = self.cache_dir / "metadata.npy"
            if cache_file.exists() and meta_file.exists():
                return self._load_from_cache()

        samples: List[SampleRecord] = []
        sample_id = 0

        for batch_idx in range(1, 11):
            batch_file = batch_dir / f"batch{batch_idx}.dat"
            if not batch_file.exists():
                continue

            with open(batch_file, "r", encoding="utf-8") as f:
                for line in f:
                    parsed = self._parse_line(line, batch_idx)
                    if parsed is None:
                        continue

                    target = {
                        "gas": parsed["gas_idx"],
                        "ppm": parsed["concentration"],
                        "batch": parsed["batch"],
                    }
                    meta = {
                        "gas_name": parsed["gas_name"],
                        "ppm": parsed["concentration"],
                        "batch": parsed["batch"],
                        "features": parsed["features"],
                    }

                    sid = f"{sample_id}_{parsed['gas_name']}_{parsed['concentration']:.2f}ppm_batch{batch_idx}"
                    samples.append(
                        SampleRecord(
                            sample_id=sid,
                            path=batch_file,
                            target=target,
                            meta=meta,
                        )
                    )
                    sample_id += 1

        if self._cache:
            self._save_to_cache(samples)

        return samples

    def _save_to_cache(self, samples: List[SampleRecord]) -> None:
        """Save processed samples to cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Save features as single numpy array
        features = np.stack([s.meta["features"] for s in samples], axis=0)
        np.save(self.cache_dir / "features.npy", features)

        # Save metadata
        metadata = []
        for s in samples:
            metadata.append({
                "sample_id": s.sample_id,
                "gas": s.target["gas"],
                "ppm": s.target["ppm"],
                "batch": s.target["batch"],
                "gas_name": s.meta["gas_name"],
            })
        np.save(self.cache_dir / "metadata.npy", np.array(metadata, dtype=object))

    def _load_from_cache(self) -> List[SampleRecord]:
        """Load samples from cache."""
        features = np.load(self.cache_dir / "features.npy")
        metadata = np.load(self.cache_dir / "metadata.npy", allow_pickle=True)

        samples = []
        for i, meta in enumerate(metadata):
            target = {
                "gas": meta["gas"],
                "ppm": meta["ppm"],
                "batch": meta["batch"],
            }
            sample_meta = {
                "gas_name": meta["gas_name"],
                "ppm": meta["ppm"],
                "batch": meta["batch"],
                "features": features[i],
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
            - features: numpy array of shape (16, 8) - 16 sensors, 8 features each
            - target: dict with keys 'gas', 'ppm', 'batch'
        """
        features = record.meta["features"]
        if not isinstance(features, np.ndarray):
            features = np.array(features, dtype=np.float32)
        return features, dict(record.target)

    @property
    def targets(self) -> List[int]:
        return [r.target["gas"] for r in self._samples]

    def extra_repr(self) -> str:
        base = super().extra_repr()
        parts = [base] if base else []
        parts.append(f"classes={len(self.classes)}")
        parts.append(f"cache={self._cache}")
        return "\n".join(parts)
