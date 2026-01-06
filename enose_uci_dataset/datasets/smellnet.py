"""SmellNet Dataset for smell recognition classification.

This dataset is designed ONLY for classification tasks, not for MAE/imputation pretraining.

SMELLNET is the largest sensor-based multitask smell dataset, comprising 68 hours of sensor
readings with 828,000 data points for 50 base substances and 43 mixtures.

Hardware (per Arduino code):
    - Grove Multichannel Gas Sensor V2: GM-102B (NO2), GM-302B (C2H5OH), GM-502B (VOC), GM-702B (CO)
    - MQ-135 (Alcohol), MQ-9 (LPG), MQ-3 (Benzene - sometimes disabled)
    - BME680 (temperature, pressure, humidity, gas resistance)

Dataset Components (HETEROGENEOUS SAMPLING RATES):
    - SMELLNET-PURE: 50 pure substances @ 1Hz, 12 channels
      10-min sessions × 6 repeats = 1 hour/substance
    
    - SMELLNET-MIXTURE: 43 mixtures @ 10Hz, 12 channels  
      18 hours, 648,000 datapoints: train (679), test-seen (215), test-unseen (184)

Channel Selection (Appendix D):
    Only 6 channels are kept by default: NO2, C2H5OH, VOC, CO, Alcohol, LPG.
    Other channels (Benzene, Temperature, Pressure, Humidity, Gas_Resistance, Altitude)
    dropped due to sensor malfunctions in some data.

Data sources:
    - HuggingFace: DeweiFeng/smell-net (recommended)
    - Local: SmellNet-iclr project directory

Reference:
    SmellNet: Neural network framework for smell recognition using sensor data
    paired with GC-MS (Gas Chromatography-Mass Spectrometry) for multimodal fusion.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ._base import BaseEnoseDataset, SampleRecord
from ._info import get_dataset_info


# HuggingFace dataset identifier
HF_REPO_ID = "DeweiFeng/smell-net"

# 50 ingredient categories
SMELLNET_INGREDIENTS = [
    'allspice', 'almond', 'angelica', 'apple', 'asparagus',
    'avocado', 'banana', 'brazil_nut', 'broccoli', 'brussel_sprouts',
    'cabbage', 'cashew', 'cauliflower', 'chamomile', 'chervil',
    'chestnuts', 'chives', 'cinnamon', 'cloves', 'coriander',
    'cumin', 'dill', 'garlic', 'ginger', 'hazelnut',
    'kiwi', 'lemon', 'mandarin_orange', 'mango', 'mint',
    'mugwort', 'mustard', 'nutmeg', 'oregano', 'peach',
    'peanuts', 'pear', 'pecans', 'pili_nut', 'pineapple',
    'pistachios', 'potato', 'radish', 'saffron', 'star_anise',
    'strawberry', 'sweet_potato', 'tomato', 'turnip', 'walnuts'
]

# Create label encoder mapping
INGREDIENT_TO_IDX = {ing: idx for idx, ing in enumerate(SMELLNET_INGREDIENTS)}
IDX_TO_INGREDIENT = {idx: ing for ing, idx in INGREDIENT_TO_IDX.items()}

# 6 channels to keep per paper (Appendix D)
# "we decided to keep only 6 channels (NO2, C2H5OH, VOC, CO, Alcohol, LPG)"
SMELLNET_6_CHANNELS = ['NO2', 'C2H5OH', 'VOC', 'CO', 'Alcohol', 'LPG']

# All 12 channels available in the full dataset
SMELLNET_ALL_CHANNELS = [
    'NO2', 'C2H5OH', 'VOC', 'CO', 'Alcohol', 'LPG',
    'Benzene', 'Temperature', 'Pressure', 'Humidity', 'Gas_Resistance', 'Altitude'
]

# =============================================================================
# Global Normalization Constants (like ImageNet mean/std)
# Computed from entire training set only, used for train/val/test uniformly.
# =============================================================================

# SmellNet Pure (6 channels) - from base_training (250 samples, ~143k timesteps)
# Order: NO2, C2H5OH, VOC, CO, Alcohol, LPG
SMELLNET_PURE_MEAN = [26.867, 14.647, 29.401, 22.487, 0.004, 2.011]
SMELLNET_PURE_STD = [77.336, 70.735, 99.810, 34.405, 1.721, 17.164]

# SmellNet Mixture (6 channels) - TODO: compute from training_new
SMELLNET_MIXTURE_MEAN = None  # To be computed
SMELLNET_MIXTURE_STD = None


class SmellNet(BaseEnoseDataset):
    """SmellNet dataset for smell recognition classification.
    
    This dataset is designed for CLASSIFICATION ONLY - it is NOT intended for
    MAE/imputation pretraining tasks in the enose_uci_dataset framework.
    
    IMPORTANT: SmellNet has heterogeneous sampling rates!
        - subset='pure': 50 pure substances @ 1Hz (smellnet_pure)
        - subset='mixture': 43 mixtures @ 10Hz (smellnet_mixture)
    
    Sensors (per Arduino code):
        - Grove Multichannel V2: GM-102B, GM-302B, GM-502B, GM-702B
        - MQ-135 (Alcohol), MQ-9 (LPG)
        - BME680 (environmental)
    
    Channel Selection:
        Per paper Appendix D, only 6 channels are used by default:
        NO2, C2H5OH, VOC, CO, Alcohol, LPG.
        Set use_all_channels=True to use all 12 channels including environmental sensors.
    
    Args:
        root: Root directory for caching downloaded data.
        subset: Which subset to use (affects sampling rate):
            - 'pure': Pure substances @ 1Hz (default)
            - 'mixture': Mixtures @ 10Hz
        split: Data split to use:
            - 'train': offline_training (250 samples for pure, 679 for mixture)
            - 'test': offline_testing (50 for pure)
            - 'test_seen': mixture test-seen (215)
            - 'test_unseen': mixture test-unseen (184)
            - None: all splits combined
        download: If True, download from HuggingFace (DeweiFeng/smell-net).
        use_all_channels: If True, use all 12 channels instead of paper's 6 channels.
        smellnet_root: Optional path to local SmellNet data directory.
    
    Example:
        >>> ds_pure = SmellNet(root=".cache", subset="pure", split="train")
        >>> ds_mix = SmellNet(root=".cache", subset="mixture", split="train")
        >>> print(f"Pure: {ds_pure.sample_rate}Hz, Mixture: {ds_mix.sample_rate}Hz")
        Pure: 1Hz, Mixture: 10Hz
    """
    name = "smellnet"
    
    # Class attributes for label encoding
    ingredients = SMELLNET_INGREDIENTS
    ingredient_to_idx = INGREDIENT_TO_IDX
    idx_to_ingredient = IDX_TO_INGREDIENT
    
    # Default to 6 channels per paper
    sensor_columns = SMELLNET_6_CHANNELS
    
    # Split mapping: our split names -> HF directory names
    _SPLIT_TO_DIR = {
        'train': 'offline_training',
        'test': 'offline_testing',
        'online_nuts': 'online_nuts',
        'online_spices': 'online_spices',
    }
    
    # Subset configurations
    _SUBSET_CONFIG = {
        'pure': {
            'name': 'smellnet_pure',
            'sample_rate': 1,  # Hz
            'description': 'Pure substances (50 classes)',
        },
        'mixture': {
            'name': 'smellnet_mixture', 
            'sample_rate': 10,  # Hz
            'description': 'Mixtures (43 classes)',
        },
    }
    
    def __init__(
        self,
        root: Union[str, Path],
        *,
        subset: str = "pure",  # 'pure' (1Hz) or 'mixture' (10Hz)
        split: Optional[str] = None,
        download: bool = False,
        transforms=None,
        transform=None,
        target_transform=None,
        use_all_channels: bool = False,
        smellnet_root: Optional[Union[str, Path]] = None,
    ):
        if subset not in self._SUBSET_CONFIG:
            raise ValueError(f"Invalid subset '{subset}'. Expected: 'pure' or 'mixture'")
        
        self._subset = subset
        self._subset_config = self._SUBSET_CONFIG[subset]
        self._smellnet_root = Path(smellnet_root) if smellnet_root else None
        self._use_all_channels = use_all_channels
        
        # Keep name as 'smellnet' for base class compatibility (dataset_dir)
        # The actual subset name is in self._subset_config['name']
        
        # Set sensor columns based on channel mode
        if use_all_channels:
            self.sensor_columns = SMELLNET_ALL_CHANNELS
        else:
            self.sensor_columns = SMELLNET_6_CHANNELS
        
        super().__init__(
            root,
            split=split,
            download=download,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        
        # Set the full name after init (for display purposes)
        self.name = self._subset_config['name']
    
    @property
    def dataset_dir(self) -> Path:
        """Override to use consistent 'smellnet' directory for all subsets."""
        return self.root / "smellnet"
    
    @property
    def data_dir(self) -> Path:
        """Get the data directory containing CSV files."""
        if self._smellnet_root is not None:
            # Use local SmellNet project data
            return self._smellnet_root / "data"
        # Use HF cached data
        return self.dataset_dir / "data"
    
    def _check_exists(self) -> bool:
        """Override to check the correct directory."""
        data_dir = self.data_dir
        if not data_dir.exists():
            return False
        
        # Check for HF structure (offline_training, offline_testing)
        hf_dirs = ['offline_training', 'offline_testing']
        has_hf = any((data_dir / d).exists() for d in hf_dirs)
        
        # Check for local SmellNet-iclr structure (base_training)
        local_dirs = ['base_training', 'training_new']
        has_local = any((data_dir / d).exists() for d in local_dirs)
        
        return has_hf or has_local
    
    def download(self) -> None:
        """Download dataset from HuggingFace."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise RuntimeError(
                "huggingface_hub is required to download SmellNet. "
                "Install it with: pip install huggingface_hub"
            )
        
        print(f"Downloading SmellNet dataset from HuggingFace ({HF_REPO_ID})...")
        
        # Download the entire dataset repository
        cache_dir = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=self.dataset_dir,
            local_dir_use_symlinks=False,
        )
        
        print(f"SmellNet dataset downloaded to: {cache_dir}")
    
    def _get_split_dirs(self) -> List[Tuple[Path, str]]:
        """Get directories to scan based on split.
        
        Returns:
            List of (directory_path, split_name) tuples
        """
        data_dir = self.data_dir
        
        # Determine which directory mapping to use
        # HF structure: offline_training, offline_testing, online_nuts, online_spices
        # Local structure: base_training, test_seen, test_unseen
        
        has_hf_structure = (data_dir / 'offline_training').exists()
        
        if has_hf_structure:
            # Use HF directory structure
            if self.split is None:
                # Return all available directories
                dirs = []
                for split_name, dir_name in self._SPLIT_TO_DIR.items():
                    dir_path = data_dir / dir_name
                    if dir_path.exists():
                        dirs.append((dir_path, split_name))
                return dirs
            elif self.split in self._SPLIT_TO_DIR:
                dir_name = self._SPLIT_TO_DIR[self.split]
                return [(data_dir / dir_name, self.split)]
            else:
                raise ValueError(
                    f"Invalid split '{self.split}'. "
                    f"Expected one of: None, {list(self._SPLIT_TO_DIR.keys())}"
                )
        else:
            # Use local SmellNet-iclr structure
            # pure subset uses base_training/base_testing (hierarchical)
            # mixture subset uses training_new/test_seen (flat)
            if self._subset == 'pure':
                local_mapping = {
                    'train': 'base_training',
                    'test': 'base_testing',
                    'test_seen': 'base_testing',
                }
            else:  # mixture
                local_mapping = {
                    'train': 'training_new',
                    'test': 'test_seen',
                    'test_seen': 'test_seen',
                    'test_unseen': 'test_unseen',
                }
            
            if self.split is None:
                dirs = []
                # Iterate through mapping keys to get correct directories
                for split_name, dir_name in local_mapping.items():
                    dir_path = data_dir / dir_name
                    if dir_path.exists() and (dir_path, split_name) not in dirs:
                        dirs.append((dir_path, split_name))
                return dirs
            elif self.split in local_mapping:
                dir_name = local_mapping[self.split]
                return [(data_dir / dir_name, self.split)]
            else:
                raise ValueError(
                    f"Invalid split '{self.split}'. "
                    f"For local data, expected one of: None, 'train', 'test', 'test_seen', 'test_unseen'"
                )
    
    def _make_dataset(self) -> List[SampleRecord]:
        """Build the dataset by scanning CSV files."""
        samples: List[SampleRecord] = []
        split_dirs = self._get_split_dirs()
        
        for split_dir, split_name in split_dirs:
            if not split_dir.exists():
                continue
            
            # Check if this is a hierarchical structure (ingredient folders)
            subdirs = [p for p in split_dir.iterdir() if p.is_dir()]
            has_subdirs = len(subdirs) > 0
            
            if has_subdirs:
                # Hierarchical structure: iterate through ingredient folders
                for ingredient_dir in sorted(subdirs):
                    ingredient_name = ingredient_dir.name.lower()
                    
                    # Check if this is a known ingredient
                    if ingredient_name not in self.ingredient_to_idx:
                        continue
                    
                    label_idx = self.ingredient_to_idx[ingredient_name]
                    
                    # Scan CSV files in this ingredient folder
                    for csv_file in sorted(ingredient_dir.glob("*.csv")):
                        target = {
                            "ingredient": ingredient_name,
                            "label": label_idx,
                            "split": split_name,
                        }
                        meta = {
                            "ingredient": ingredient_name,
                            "label": label_idx,
                            "split": split_name,
                            "source_file": csv_file.name,
                        }
                        samples.append(SampleRecord(
                            sample_id=f"{ingredient_name}_{csv_file.stem}",
                            path=csv_file,
                            target=target,
                            meta=meta,
                        ))
            else:
                # Flat structure: parse ingredient from filename
                for csv_file in sorted(split_dir.glob("*.csv")):
                    ingredient_name = self._parse_ingredient_from_filename(csv_file.name)
                    
                    if ingredient_name is None:
                        continue  # Skip mixtures or unknown files
                    
                    label_idx = self.ingredient_to_idx[ingredient_name]
                    
                    target = {
                        "ingredient": ingredient_name,
                        "label": label_idx,
                        "split": split_name,
                    }
                    meta = {
                        "ingredient": ingredient_name,
                        "label": label_idx,
                        "split": split_name,
                        "source_file": csv_file.name,
                    }
                    samples.append(SampleRecord(
                        sample_id=f"{ingredient_name}_{csv_file.stem}",
                        path=csv_file,
                        target=target,
                        meta=meta,
                    ))
        
        return samples
    
    def _parse_ingredient_from_filename(self, filename: str) -> Optional[str]:
        """Extract ingredient name from filename.
        
        Handles various filename formats:
        - almond.xxx.csv -> almond
        - almond20_banana80.xxx.csv -> None (mixture, skip)
        """
        name = filename.lower().replace('.csv', '')
        parts = name.split('.')
        ingredient_part = parts[0]
        
        # If contains underscore with numbers, it's likely a mixture
        if '_' in ingredient_part:
            sub_parts = ingredient_part.split('_')
            first_part = sub_parts[0]
            if first_part in self.ingredient_to_idx and not any(c.isdigit() for c in first_part):
                if len(sub_parts) > 1 and any(c.isdigit() for c in sub_parts[1]):
                    return None  # It's a mixture
                return first_part
            return None
        
        # Pure ingredient name
        if ingredient_part in self.ingredient_to_idx:
            return ingredient_part
        
        return None
    
    def _load_sample(self, record: SampleRecord) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load a single sample from CSV file.
        
        Returns:
            Tuple of (DataFrame with sensor columns, target dict)
        """
        # Determine cache path based on channel mode
        suffix = '.6ch.npy' if not self._use_all_channels else '.12ch.npy'
        cache_path = record.path.with_suffix(suffix)
        
        if cache_path.exists():
            data_array = np.load(cache_path)
            df = pd.DataFrame(data_array, columns=self.sensor_columns[:data_array.shape[1]])
        else:
            # Load CSV
            df = pd.read_csv(record.path)
            
            # Drop timestamp column if present
            if 'timestamp_ms' in df.columns:
                df = df.drop(columns=['timestamp_ms'])
            
            # Select sensor columns
            selected_cols = []
            for col in self.sensor_columns:
                # Handle alternative column names (C2H5OH vs C2H5CH)
                if col in df.columns:
                    selected_cols.append(col)
                elif col == 'C2H5OH' and 'C2H5CH' in df.columns:
                    selected_cols.append('C2H5CH')
                elif col == 'C2H5CH' and 'C2H5OH' in df.columns:
                    selected_cols.append('C2H5OH')
            
            if len(selected_cols) == 0:
                raise RuntimeError(
                    f"CSV file {record.path} does not contain expected sensor columns. "
                    f"Expected: {self.sensor_columns}, Got: {list(df.columns)}"
                )
            
            df = df[selected_cols]
            # Rename columns to canonical names
            rename_map = {'C2H5CH': 'C2H5OH'}
            df = df.rename(columns=rename_map)
            df.columns = self.sensor_columns[:len(df.columns)]
            
            # Apply baseline subtraction (subtract first row) as in original SmellNet
            df = df - df.iloc[0]
            
            # Cache for future use
            try:
                np.save(cache_path, df.values.astype(np.float32))
            except Exception:
                pass  # Ignore cache write errors
        
        return df, dict(record.target)
    
    def get_classification_sample(self, index: int) -> Tuple[np.ndarray, int]:
        """Get sample formatted for classification training.
        
        Returns:
            Tuple of (data array [C, T], label index)
        """
        rec = self._samples[index]
        df, target = self._load_sample(rec)
        
        # Convert to [C, T] format
        data = df.values.T.astype(np.float32)  # [T, C] -> [C, T]
        
        return data, target['label']
    
    def get_sample_with_ingredient(self, index: int) -> Tuple[np.ndarray, str]:
        """Get sample with ingredient name.
        
        Returns:
            Tuple of (data array [C, T], ingredient name)
        """
        rec = self._samples[index]
        df, target = self._load_sample(rec)
        
        data = df.values.T.astype(np.float32)
        
        return data, target['ingredient']
    
    @property
    def num_classes(self) -> int:
        """Number of classification classes."""
        return len(self.ingredients)
    
    @property
    def num_channels(self) -> int:
        """Number of sensor channels."""
        return len(self.sensor_columns)
    
    @property
    def sample_rate(self) -> int:
        """Sampling rate in Hz (depends on subset: pure=1Hz, mixture=10Hz)."""
        return self._subset_config['sample_rate']
    
    @property
    def subset(self) -> str:
        """Current subset ('pure' or 'mixture')."""
        return self._subset
    
    def extra_repr(self) -> str:
        base = super().extra_repr()
        parts = [base] if base else []
        parts.append(f"subset={self._subset}")
        parts.append(f"sample_rate={self.sample_rate}Hz")
        parts.append(f"num_classes={self.num_classes}")
        parts.append(f"channels={self.sensor_columns} ({self.num_channels}ch)")
        parts.append(f"use_all_channels={self._use_all_channels}")
        return "\n".join(parts)
