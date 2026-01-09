"""SJTU-G919 Gas Sensor Dataset (55 odors).

The SJTU-G919 system is developed for electronic nose (E-nose) gas sensitivity testing.
It simulates complex atmospheric conditions and supports various sensor types.

Sensors (Table S1):
    - Sen.1: GM-502B (MEMS, Winsen) - VOC detection
    - Sen.2: GM-202B (MEMS, Winsen) - Formaldehyde/VOC
    - Sen.3: SMD-1005 (MEMS, Huiwen Nano) - VOC
    - Sen.4: SMD-1013B (MEMS, Huiwen Nano) - Combustible gas
    - Sen.5: MQ-138 (Ceramic Tube, Winsen) - Benzene/Alcohol/VOC
    - Sen.6: MQ-2 (Ceramic Tube, Winsen) - Combustible gas
    - Sen.7: TGS-2609 (Ceramic Tube, Figaro) - VOC/Odor
    - Sen.8: TGS-2620 (Ceramic Tube, Figaro) - Alcohol/VOC

Odor Categories (55 total):
    - Perfumes (6): Giorgio Armani Jasmin Kusamono, Giorgio Armani The Yulong, 
      Hermes Terre d'Hermes, Hermes Un Jardin Sur Le Nil, Floridawater, Chanel Coco Mademoiselle
    - Condiments (5): Chili Oil, Cooking Wine, Sesame Oil, Soy Sauce, Vinegar
    - Fruits (7): Apple, Coco, Grape, Lemon, Melon, Orange, Pear
    - Milk (6): AD Calcium Milk, Banana Milk, Deluxe Milk, Hot Kid, Satine Milk, Yogurt
    - Spices (3): Cinnamon, Cumin, Fennel + Geranium Leaves, Peppercorn
    - Vegetables (5): Chili Pepper, Lettuce, Onion, Romaine Lettuce, Tomato
    - Wines (11): Bawei Beer, Black Whisky, Brown Liqueur, Bryka Brandy, Carterport Liqueur,
      Chinese Jing Wine, Erguotou, Fast King Brandy, Glacier Vodka, Jazz Whisky, Xuehua Beer
    - Drinks (10): Americano, Cola, Fanta, Jia Duo Bao, Latte, Milk Tea, Red Bull, Rio, Soy Milk, Sprite
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .._base import BaseEnoseDataset, SampleRecord


# Odor categories mapping
G919_CATEGORIES = {
    "perfume": [
        "Giorgio Armani Jasmin Kusamono",
        "Giorgio Armani Thé Yulong",
        "Hermes Terre dHermes",
        "Hermes Un Jardin Sur Le Nil",
        "Floridawater",
        "Chanel Coco Mademoiselle",
    ],
    "condiment": [
        "Chili oil",
        "Cooking wine",
        "Sesame oil",
        "Soy sauce",
        "Vinegar",
    ],
    "fruit": [
        "Apple",
        "Coco",
        "Grape",
        "Lemon",
        "Melon",
        "Orange",
        "Pear",
    ],
    "milk": [
        "AD calcium milk",
        "Banana milk",
        "Deluxe milk",
        "Hot kid",
        "Satine milk",
        "Yogurt",
    ],
    "spice": [
        "Cinnamon",
        "Cumin",
        "Fennel",
        "Geranium Leaves",
        "Peppercorn",
    ],
    "vegetable": [
        "Chili pepper",
        "Lettuce",
        "Onion",
        "Romaine lettuce",
        "Tomato",
    ],
    "wine": [
        "Bawei Beer",
        "Black Whisky",
        "Brown Liqueur",
        "Bryka Brandy",
        "Carterport Liqueur",
        "Chinese Jing Wine",
        "Erguotou",
        "Fast King Brandy",
        "Glacier Vodka",
        "Jazz Whisky",
        "Xuehua Beer",
    ],
    "drink": [
        "Americano",
        "Cola",
        "Fanta",
        "Jia duo bao",
        "Latte",
        "Milk tea",
        "Red bull",
        "Rio",
        "Soy milk",
        "Spirite",
    ],
}

# Build global label mapping
G919_ODOR_TO_IDX: Dict[str, int] = {}
G919_IDX_TO_ODOR: Dict[int, str] = {}
G919_ODOR_TO_CATEGORY: Dict[str, str] = {}

_idx = 0
for category, odors in G919_CATEGORIES.items():
    for odor in odors:
        G919_ODOR_TO_IDX[odor.lower()] = _idx
        G919_IDX_TO_ODOR[_idx] = odor
        G919_ODOR_TO_CATEGORY[odor.lower()] = category
        _idx += 1

G919_NUM_CLASSES = _idx  # 55 classes

# Category to index mapping
G919_CATEGORY_TO_IDX = {cat: i for i, cat in enumerate(G919_CATEGORIES.keys())}


def _normalize_odor_name(name: str) -> str:
    """Normalize odor name for matching."""
    # Remove trailing numbers (repetition markers like _1, _2)
    name = re.sub(r"_\d+(_full)?$", "", name)
    # Handle common variations
    name = name.lower().strip()
    # Some files have inconsistent naming
    name = name.replace("_", " ").replace("-", " ")
    return name


# Common typo/spelling variations in filenames
_SPELLING_FIXES = {
    "baiwei beer": "bawei beer",
    "chili peper": "chili pepper",
    "fenta": "fanta",
    "milk_tea": "milk tea",
    "banana_milk": "banana milk",
}


def _find_odor_label(filename: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """Find odor label from filename.
    
    Filename format: {odor_name}_{repetition}_{concentration}.csv
    e.g., "Floridawater_1_high.csv", "Chanel Coco Mademoiselle_2_low.csv"
    
    Returns:
        Tuple of (odor_idx, odor_name, category) or (None, None, None) if not found
    """
    # Extract base name without extension
    base = Path(filename).stem
    
    # Remove _interp suffix if present
    if base.endswith("_interp"):
        base = base[:-7]
    
    # New format: {odor_name}_{repetition}_{concentration}
    # Remove trailing _high, _low, _mid and repetition number
    parts = base.rsplit("_", 2)  # Split from right, max 2 splits
    if len(parts) >= 3:
        # Check if last part is concentration (high/low/mid)
        if parts[-1].lower() in ("high", "low", "mid"):
            # Check if second-to-last is a number (repetition)
            if parts[-2].isdigit():
                # Reconstruct odor name from remaining parts
                base = parts[0]
            else:
                # Maybe format is {name}_{concentration} without repetition
                base = "_".join(parts[:-1])
    elif len(parts) == 2:
        # Format might be {name}_{concentration}
        if parts[-1].lower() in ("high", "low", "mid"):
            base = parts[0]
    
    normalized = _normalize_odor_name(base)
    
    # Apply spelling fixes
    if normalized in _SPELLING_FIXES:
        normalized = _SPELLING_FIXES[normalized]
    
    # Direct match
    if normalized in G919_ODOR_TO_IDX:
        idx = G919_ODOR_TO_IDX[normalized]
        return idx, G919_IDX_TO_ODOR[idx], G919_ODOR_TO_CATEGORY[normalized]
    
    # Try fuzzy matching - find best match
    for odor_key in G919_ODOR_TO_IDX:
        # Check if normalized name contains the odor key or vice versa
        if normalized.startswith(odor_key) or odor_key.startswith(normalized):
            idx = G919_ODOR_TO_IDX[odor_key]
            return idx, G919_IDX_TO_ODOR[idx], G919_ODOR_TO_CATEGORY[odor_key]
        # Handle spacing differences
        if normalized.replace(" ", "") == odor_key.replace(" ", ""):
            idx = G919_ODOR_TO_IDX[odor_key]
            return idx, G919_IDX_TO_ODOR[idx], G919_ODOR_TO_CATEGORY[odor_key]
    
    return None, None, None


class G919SensorDataset(BaseEnoseDataset):
    """SJTU-G919 Gas Sensor Dataset with 55 odor classes.
    
    This dataset contains gas sensor responses from an 8-sensor array exposed to
    55 different odors across 8 categories (perfumes, condiments, fruits, milk,
    spices, vegetables, wines, drinks).
    
    The sensor array uses:
        - 4 MEMS sensors (GM-502B, GM-202B, SMD-1005, SMD-1013B)
        - 4 Ceramic Tube sensors (MQ-138, MQ-2, TGS-2609, TGS-2620)
    
    Args:
        root: Root directory where the dataset is located. The dataset should be
            in a subdirectory named 'g919_55' or specified via local_path.
        local_path: Optional path to the actual data directory if different from
            root/g919_55. Use this for loading from cache directories.
        split: Optional split ('train', 'test', or None for all data).
        download: Not supported for this dataset (local only).
        transforms: Optional transforms to apply to (data, target).
        transform: Optional transform to apply to data only.
        target_transform: Optional transform to apply to target only.
    
    Example:
        >>> ds = G919SensorDataset(
        ...     root="/root/UCI_dataset",
        ...     local_path="/root/UCI_dataset/.cache/G919-55",
        ...     split="train"
        ... )
        >>> len(ds)
        110  # ~2 repetitions * 55 odors
        >>> data, target = ds[0]
        >>> data.shape  # (time_steps, 8)
    """
    name = "g919_55"
    
    def __init__(
        self,
        root: Union[str, Path],
        *,
        local_path: Optional[Union[str, Path]] = None,
        split: Optional[str] = None,
        download: bool = False,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self._local_path = Path(local_path) if local_path else None
        super().__init__(
            root,
            split=split,
            download=download,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
    
    @property
    def dataset_dir(self) -> Path:
        """Override to support local_path.
        
        Default structure: root/g919_55/single/{category}/{category}_train|test/
        """
        if self._local_path is not None:
            return self._local_path
        # Data is in 'single' subdirectory (not 'raw')
        return self.root / self.name / "single"
    
    def download(self) -> None:
        """Not supported - this is a local-only dataset."""
        raise NotImplementedError(
            "G919SensorDataset does not support download. "
            "Please provide local_path to the data directory."
        )
    
    def _check_exists(self) -> bool:
        """Check if dataset exists at local_path or default location."""
        data_dir = self.dataset_dir
        if not data_dir.exists():
            return False
        # Check for at least one category directory
        for category in G919_CATEGORIES:
            if (data_dir / category).exists():
                return True
        return False
    
    def _make_dataset(self) -> List[SampleRecord]:
        """Build sample records from directory structure."""
        data_dir = self.dataset_dir
        samples: List[SampleRecord] = []
        
        for category in G919_CATEGORIES:
            cat_dir = data_dir / category
            if not cat_dir.exists():
                continue
            
            # Determine which subdirectories to scan based on split
            subdirs = []
            if self.split is None:
                # All data
                subdirs = list(cat_dir.iterdir())
            elif self.split == "train":
                subdirs = [d for d in cat_dir.iterdir() if d.is_dir() and "train" in d.name.lower()]
            elif self.split == "test":
                subdirs = [d for d in cat_dir.iterdir() if d.is_dir() and "test" in d.name.lower()]
            else:
                raise ValueError(f"Unknown split: {self.split}. Use 'train', 'test', or None.")
            
            for subdir in subdirs:
                if not subdir.is_dir():
                    continue
                
                for csv_path in sorted(subdir.glob("*.csv")):
                    # Skip processed files
                    if csv_path.name.startswith("processed_"):
                        continue
                    
                    # Only load files with _high, _low, _mid suffix (concentration segments)
                    stem = csv_path.stem
                    # Remove _interp if present for checking
                    check_stem = stem[:-7] if stem.endswith("_interp") else stem
                    if not any(check_stem.endswith(f"_{conc}") for conc in ("high", "low", "mid")):
                        continue
                    
                    odor_idx, odor_name, odor_category = _find_odor_label(csv_path.name)
                    if odor_idx is None:
                        # Try to infer from directory
                        continue
                    
                    # Determine if train or test from path
                    is_train = "train" in subdir.name.lower()
                    
                    target = {
                        "odor_idx": odor_idx,
                        "category_idx": G919_CATEGORY_TO_IDX[odor_category],
                    }
                    
                    meta = {
                        "odor_name": odor_name,
                        "category": odor_category,
                        "is_train": is_train,
                        "filename": csv_path.name,
                    }
                    
                    sample_id = f"{category}_{csv_path.stem}"
                    samples.append(SampleRecord(
                        sample_id=sample_id,
                        path=csv_path,
                        target=target,
                        meta=meta,
                    ))
        
        return samples
    
    def _load_sample(self, record: SampleRecord) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load a sample from CSV file.
        
        Returns:
            Tuple of (DataFrame with sensor columns, target dict)
        """
        # Check for cached npy file
        cache_path = record.path.with_suffix('.npy')
        
        if cache_path.exists():
            data_array = np.load(cache_path)
            df = pd.DataFrame(data_array, columns=[f"sensor_{i}" for i in range(8)])
        else:
            # Read CSV - columns are: Time (s), sen.1, ..., sen.8, [optional empty column]
            df = pd.read_csv(record.path)
            
            # Extract sensor columns (skip Time column and any unnamed columns)
            sensor_cols = [c for c in df.columns if c.startswith("sen.")]
            if len(sensor_cols) != 8:
                # Try alternative naming
                sensor_cols = [c for c in df.columns if "sen" in c.lower()][:8]
            
            if len(sensor_cols) < 8:
                raise RuntimeError(
                    f"Expected 8 sensor columns, found {len(sensor_cols)} in {record.path}"
                )
            
            df = df[sensor_cols].copy()
            df.columns = [f"sensor_{i}" for i in range(8)]
            
            # Save to cache
            try:
                np.save(cache_path, df.values.astype(np.float32))
            except Exception:
                pass  # Ignore cache write errors
        
        return df, dict(record.target)
    
    @property
    def num_classes(self) -> int:
        """Number of odor classes."""
        return G919_NUM_CLASSES
    
    @property
    def num_categories(self) -> int:
        """Number of odor categories."""
        return len(G919_CATEGORIES)
    
    def get_odor_name(self, idx: int) -> str:
        """Get odor name from index."""
        return G919_IDX_TO_ODOR.get(idx, "Unknown")
    
    def get_category_name(self, idx: int) -> str:
        """Get category name from index."""
        categories = list(G919_CATEGORIES.keys())
        if 0 <= idx < len(categories):
            return categories[idx]
        return "Unknown"
    
    def extra_repr(self) -> str:
        base = super().extra_repr()
        parts = [base] if base else []
        parts.append(f"num_classes={self.num_classes}")
        parts.append(f"num_categories={self.num_categories}")
        if self._local_path:
            parts.append(f"local_path={self._local_path}")
        return "\n".join(parts)
