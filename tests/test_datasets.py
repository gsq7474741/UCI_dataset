"""Unit tests for enose_uci_dataset.datasets module.

Comprehensive tests covering:
- Download functionality
- Data processing
- Cache mechanism
- Metadata retrieval
- PyTorch DataLoader integration
"""
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest import TestCase, main, skipIf

import sys
sys.path.insert(0, '/root/UCI_dataset')

import numpy as np
import pandas as pd

# Import all dataset classes
from enose_uci_dataset.datasets import (
    DATASETS,
    AlcoholQCMSensor,
    GasSensorArrayDrift,
    GasSensorDynamic,
    GasSensorFlowModulation,
    GasSensorLowConcentration,
    GasSensorTemperatureModulation,
    GasSensorTurbulent,
    GasSensorsForHomeActivityMonitoring,
    TwinGasSensorArrays,
    DatasetInfo,
    get_dataset_class,
    get_dataset_info,
    list_datasets,
)

# Try to import torch for DataLoader tests
try:
    import torch
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# Configuration
# =============================================================================

# Default data root - can be overridden via environment variable
DATA_ROOT = Path(os.environ.get("ENOSE_DATA_ROOT", "./.cache")).resolve()

# Datasets to test with download (smaller ones for CI)
DOWNLOAD_TEST_DATASETS = [
    "alcohol_qcm_sensor_dataset",  # ~15KB
]


def has_dataset(name: str, root: Path = DATA_ROOT) -> bool:
    """Check if a dataset exists at the given root."""
    dataset_path = root / name
    return dataset_path.exists() and any(dataset_path.iterdir())


def numpy_collate_fn(batch: List[Tuple[Any, Any]]) -> Tuple[np.ndarray, List[Dict]]:
    """Custom collate function for numpy arrays and dict targets."""
    data_list = []
    target_list = []
    for data, target in batch:
        if isinstance(data, pd.DataFrame):
            data = data.values
        if isinstance(data, np.ndarray):
            data_list.append(data)
        target_list.append(target)
    
    # Try to stack if all shapes are the same
    try:
        stacked_data = np.stack(data_list, axis=0)
    except ValueError:
        stacked_data = data_list  # Return as list if shapes differ
    
    return stacked_data, target_list


# =============================================================================
# Test: Dataset Registry
# =============================================================================

class TestDatasetRegistry(TestCase):
    """Test dataset registry functions."""

    def test_list_datasets_returns_list(self):
        """Test that list_datasets returns a non-empty list."""
        datasets = list_datasets()
        self.assertIsInstance(datasets, list)
        self.assertGreater(len(datasets), 0)

    def test_list_datasets_contains_expected(self):
        """Test that list_datasets contains expected datasets."""
        datasets = list_datasets()
        expected = [
            "twin_gas_sensor_arrays",
            "gas_sensors_for_home_activity_monitoring",
            "alcohol_qcm_sensor_dataset",
            "gas_sensor_array_drift_dataset_at_different_concentrations",
        ]
        for name in expected:
            self.assertIn(name, datasets)

    def test_get_dataset_class_all(self):
        """Test that get_dataset_class returns correct classes for all datasets."""
        for name, cls in DATASETS.items():
            retrieved = get_dataset_class(name)
            self.assertEqual(retrieved, cls)

    def test_get_dataset_class_invalid(self):
        """Test that get_dataset_class raises KeyError for unknown datasets."""
        with self.assertRaises(KeyError):
            get_dataset_class("nonexistent_dataset")


# =============================================================================
# Test: Metadata
# =============================================================================

class TestDatasetMetadata(TestCase):
    """Test dataset metadata retrieval."""

    def test_get_dataset_info_returns_datasetinfo(self):
        """Test that get_dataset_info returns DatasetInfo objects."""
        for name in list_datasets():
            info = get_dataset_info(name)
            self.assertIsInstance(info, DatasetInfo)

    def test_dataset_info_fields(self):
        """Test that DatasetInfo has required fields."""
        for name in list_datasets():
            info = get_dataset_info(name)
            self.assertEqual(info.name, name)
            self.assertIsNotNone(info.url)
            self.assertIsInstance(info.url, str)
            self.assertTrue(info.url.startswith("http"))

    def test_dataset_info_sha1(self):
        """Test that DatasetInfo has SHA1 checksum."""
        for name in list_datasets():
            info = get_dataset_info(name)
            if info.sha1:  # SHA1 is optional but should be string if present
                self.assertIsInstance(info.sha1, str)
                self.assertEqual(len(info.sha1), 40)  # SHA1 is 40 hex chars


# =============================================================================
# Test: Class Attributes
# =============================================================================

class TestDatasetClassAttributes(TestCase):
    """Test individual dataset class attributes."""

    def test_twin_gas_sensor_arrays_attributes(self):
        """Test TwinGasSensorArrays class attributes."""
        self.assertEqual(TwinGasSensorArrays.name, "twin_gas_sensor_arrays")
        self.assertIn("Ea", TwinGasSensorArrays.gas_to_idx)
        self.assertIn("CO", TwinGasSensorArrays.gas_to_idx)

    def test_gas_sensor_drift_attributes(self):
        """Test GasSensorArrayDrift class attributes."""
        self.assertEqual(
            GasSensorArrayDrift.name,
            "gas_sensor_array_drift_dataset_at_different_concentrations"
        )
        self.assertEqual(len(GasSensorArrayDrift.classes), 6)
        self.assertIn("Ethanol", GasSensorArrayDrift.classes)
        self.assertIn("Ammonia", GasSensorArrayDrift.classes)

    def test_alcohol_qcm_attributes(self):
        """Test AlcoholQCMSensor class attributes."""
        self.assertEqual(AlcoholQCMSensor.name, "alcohol_qcm_sensor_dataset")
        self.assertEqual(len(AlcoholQCMSensor.classes), 5)
        self.assertEqual(len(AlcoholQCMSensor.sensors), 5)

    def test_gas_sensor_turbulent_attributes(self):
        """Test GasSensorTurbulent class attributes."""
        self.assertEqual(
            GasSensorTurbulent.name,
            "gas_sensor_array_exposed_to_turbulent_gas_mixtures"
        )
        self.assertEqual(len(GasSensorTurbulent.sensor_types), 8)

    def test_gas_sensor_dynamic_attributes(self):
        """Test GasSensorDynamic class attributes."""
        self.assertEqual(
            GasSensorDynamic.name,
            "gas_sensor_array_under_dynamic_gas_mixtures"
        )
        self.assertEqual(len(GasSensorDynamic.sensor_types), 16)

    def test_gas_sensor_temperature_modulation_attributes(self):
        """Test GasSensorTemperatureModulation class attributes."""
        self.assertEqual(
            GasSensorTemperatureModulation.name,
            "gas_sensor_array_temperature_modulation"
        )
        self.assertEqual(len(GasSensorTemperatureModulation.sensor_types), 14)

    def test_gas_sensor_flow_modulation_attributes(self):
        """Test GasSensorFlowModulation class attributes."""
        self.assertEqual(
            GasSensorFlowModulation.name,
            "gas_sensor_array_under_flow_modulation"
        )
        self.assertEqual(len(GasSensorFlowModulation.gas_classes), 4)

    def test_gas_sensor_low_concentration_attributes(self):
        """Test GasSensorLowConcentration class attributes."""
        self.assertEqual(
            GasSensorLowConcentration.name,
            "gas_sensor_array_low_concentration"
        )
        self.assertEqual(len(GasSensorLowConcentration.classes), 6)
        self.assertEqual(len(GasSensorLowConcentration.sensor_types), 10)
        self.assertEqual(GasSensorLowConcentration.points_per_sensor, 900)


# =============================================================================
# Test: Error Handling
# =============================================================================

class TestDatasetErrors(TestCase):
    """Test that datasets raise appropriate errors."""

    def test_missing_data_raises_error(self):
        """Test that missing data raises RuntimeError when download=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for name, cls in DATASETS.items():
                with self.assertRaises(RuntimeError, msg=f"Dataset {name} should raise error"):
                    cls(tmpdir, download=False)


# =============================================================================
# Test: Conversion Functions
# =============================================================================

class TestConversionFunctions(TestCase):
    """Test resistance conversion functions."""

    def test_turbulent_conversion(self):
        """Test GasSensorTurbulent conversion function."""
        # Rs(KOhm) = 10 * (3110 - A) / A
        result = GasSensorTurbulent.convert_to_resistance(1000)
        expected = 10 * (3110 - 1000) / 1000
        self.assertAlmostEqual(result, expected)

    def test_dynamic_conversion(self):
        """Test GasSensorDynamic conversion function."""
        # KOhm = 40.0 / S_i
        result = GasSensorDynamic.convert_to_kohm(10)
        self.assertAlmostEqual(result, 4.0)

    def test_conversion_zero_handling(self):
        """Test that zero values return infinity."""
        self.assertEqual(GasSensorTurbulent.convert_to_resistance(0), float("inf"))
        self.assertEqual(GasSensorDynamic.convert_to_kohm(0), float("inf"))


# =============================================================================
# Test: Download, Process, Cache (Integration Tests)
# =============================================================================

class TestDownloadProcessCache(TestCase):
    """Integration tests for download, processing, and caching."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.test_root = DATA_ROOT
        cls.test_root.mkdir(parents=True, exist_ok=True)

    def test_download_alcohol_qcm(self):
        """Test downloading AlcoholQCMSensor dataset."""
        ds = AlcoholQCMSensor(str(self.test_root), download=True, cache=True)
        self.assertGreater(len(ds), 0)
        # Verify raw data exists
        self.assertTrue(ds.raw_dir.exists())

    def test_cache_creation_alcohol_qcm(self):
        """Test that cache is created after processing."""
        ds = AlcoholQCMSensor(str(self.test_root), download=True, cache=True)
        # Access data to trigger processing
        _ = ds[0]
        # Cache should exist
        self.assertTrue(ds.cache_dir.exists())

    def test_cache_reload_alcohol_qcm(self):
        """Test that data can be loaded from cache."""
        # First load creates cache
        ds1 = AlcoholQCMSensor(str(self.test_root), download=True, cache=True)
        len1 = len(ds1)
        data1, target1 = ds1[0]
        
        # Second load should use cache
        ds2 = AlcoholQCMSensor(str(self.test_root), download=False, cache=True)
        len2 = len(ds2)
        data2, target2 = ds2[0]
        
        self.assertEqual(len1, len2)
        np.testing.assert_array_equal(data1, data2)

    def test_dataset_iteration(self):
        """Test full dataset iteration."""
        ds = AlcoholQCMSensor(str(self.test_root), download=True, cache=True)
        count = 0
        for data, target in ds:
            self.assertIsInstance(data, np.ndarray)
            self.assertIsInstance(target, dict)
            count += 1
        self.assertEqual(count, len(ds))


# =============================================================================
# Test: PyTorch DataLoader Integration
# =============================================================================

@skipIf(not HAS_TORCH, "PyTorch not installed")
class TestPyTorchDataLoader(TestCase):
    """Test PyTorch DataLoader integration."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.test_root = DATA_ROOT
        cls.test_root.mkdir(parents=True, exist_ok=True)
        # Ensure dataset is downloaded
        cls.dataset = AlcoholQCMSensor(str(cls.test_root), download=True, cache=True)

    def test_dataloader_basic(self):
        """Test basic DataLoader iteration."""
        loader = DataLoader(
            self.dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=numpy_collate_fn,
        )
        
        batch_count = 0
        for batch_data, batch_targets in loader:
            self.assertIsInstance(batch_targets, list)
            batch_count += 1
            if batch_count >= 3:  # Test first 3 batches
                break
        
        self.assertGreater(batch_count, 0)

    def test_dataloader_shuffle(self):
        """Test DataLoader with shuffle."""
        loader = DataLoader(
            self.dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=numpy_collate_fn,
        )
        
        batches = []
        for batch_data, batch_targets in loader:
            batches.append(batch_targets)
            if len(batches) >= 2:
                break
        
        self.assertEqual(len(batches), 2)

    def test_dataloader_num_workers(self):
        """Test DataLoader with multiple workers."""
        loader = DataLoader(
            self.dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,  # Use 0 for safety in tests
            collate_fn=numpy_collate_fn,
        )
        
        for batch_data, batch_targets in loader:
            self.assertIsInstance(batch_targets, list)
            break

    def test_dataloader_full_epoch(self):
        """Test complete epoch iteration."""
        loader = DataLoader(
            self.dataset,
            batch_size=16,
            shuffle=False,
            collate_fn=numpy_collate_fn,
        )
        
        total_samples = 0
        for batch_data, batch_targets in loader:
            total_samples += len(batch_targets)
        
        self.assertEqual(total_samples, len(self.dataset))


# =============================================================================
# Test: Real Data Tests (require download)
# =============================================================================

@skipIf(not has_dataset("twin_gas_sensor_arrays"), "Dataset not available")
class TestTwinGasSensorArraysWithData(TestCase):
    """Tests for TwinGasSensorArrays with actual data."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = TwinGasSensorArrays(str(DATA_ROOT), download=False)

    def test_len(self):
        """Test dataset length."""
        print(f"\n[TwinGasSensorArrays] 样本数: {len(self.dataset)}")
        self.assertGreater(len(self.dataset), 0)

    def test_getitem(self):
        """Test __getitem__ returns valid data."""
        data, target = self.dataset[0]
        print(f"\n[TwinGasSensorArrays] 样本0:")
        print(f"  数据类型: {type(data).__name__}, 形状: {data.shape if hasattr(data, 'shape') else 'N/A'}")
        print(f"  标签: {target}")
        self.assertIsNotNone(data)
        self.assertIsInstance(target, dict)
        self.assertIn("gas", target)
        self.assertIn("ppm", target)

    @skipIf(not HAS_TORCH, "PyTorch not installed")
    def test_dataloader(self):
        """Test PyTorch DataLoader integration."""
        loader = DataLoader(
            self.dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=numpy_collate_fn,
        )
        for batch_data, batch_targets in loader:
            print(f"\n[TwinGasSensorArrays] DataLoader 批次:")
            print(f"  批次数据类型: {type(batch_data).__name__}")
            print(f"  批次大小: {len(batch_targets)}")
            self.assertIsInstance(batch_targets, list)
            break


@skipIf(not has_dataset("gas_sensor_array_drift_dataset_at_different_concentrations"), "Dataset not available")
class TestGasSensorArrayDriftWithData(TestCase):
    """Tests for GasSensorArrayDrift with actual data."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = GasSensorArrayDrift(str(DATA_ROOT), download=False, cache=True)

    def test_len(self):
        """Test dataset length (should have ~13910 samples)."""
        print(f"\n[GasSensorArrayDrift] 样本数: {len(self.dataset)}")
        self.assertGreater(len(self.dataset), 10000)

    def test_getitem(self):
        """Test __getitem__ returns valid data."""
        data, target = self.dataset[0]
        print(f"\n[GasSensorArrayDrift] 样本0:")
        print(f"  数据形状: {data.shape}")
        print(f"  标签: {target}")
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.shape, (16, 8))  # 16 sensors, 8 features
        self.assertIn("gas", target)
        self.assertIn("ppm", target)
        self.assertIn("batch", target)

    def test_targets(self):
        """Test targets property."""
        targets = self.dataset.targets
        self.assertEqual(len(targets), len(self.dataset))
        self.assertTrue(all(0 <= t < 6 for t in targets))

    def test_cache_exists(self):
        """Test that cache was created."""
        self.assertTrue(self.dataset.cache_dir.exists())

    @skipIf(not HAS_TORCH, "PyTorch not installed")
    def test_dataloader(self):
        """Test PyTorch DataLoader integration."""
        loader = DataLoader(
            self.dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=numpy_collate_fn,
        )
        for batch_data, batch_targets in loader:
            print(f"\n[GasSensorArrayDrift] DataLoader 批次:")
            print(f"  批次形状: {batch_data.shape if isinstance(batch_data, np.ndarray) else 'list'}")
            if isinstance(batch_data, np.ndarray):
                self.assertEqual(batch_data.shape[1:], (16, 8))
            break


@skipIf(not has_dataset("gas_sensor_array_exposed_to_turbulent_gas_mixtures"), "Dataset not available")
class TestGasSensorTurbulentWithData(TestCase):
    """Tests for GasSensorTurbulent with actual data."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = GasSensorTurbulent(str(DATA_ROOT), download=False, use_downsampled=True)

    def test_len(self):
        """Test dataset length."""
        print(f"\n[GasSensorTurbulent] 样本数: {len(self.dataset)}")
        self.assertGreater(len(self.dataset), 0)

    def test_getitem(self):
        """Test __getitem__ returns valid data."""
        data, target = self.dataset[0]
        print(f"\n[GasSensorTurbulent] 样本0:")
        print(f"  数据形状: {data.shape}")
        print(f"  列名: {list(data.columns)}")
        print(f"  标签: {target}")
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn("ethylene_level", target)
        self.assertIn("second_gas", target)

    @skipIf(not HAS_TORCH, "PyTorch not installed")
    def test_dataloader(self):
        """Test PyTorch DataLoader integration."""
        loader = DataLoader(
            self.dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=numpy_collate_fn,
        )
        for batch_data, batch_targets in loader:
            print(f"\n[GasSensorTurbulent] DataLoader 批次: {len(batch_targets)} 样本")
            self.assertIsInstance(batch_targets, list)
            break


@skipIf(not has_dataset("gas_sensor_array_under_dynamic_gas_mixtures"), "Dataset not available")
class TestGasSensorDynamicWithData(TestCase):
    """Tests for GasSensorDynamic with actual data."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = GasSensorDynamic(
            str(DATA_ROOT),
            download=False,
            mixture="both",
            segment_on_change=True
        )

    def test_len(self):
        """Test dataset length."""
        print(f"\n[GasSensorDynamic] 样本数: {len(self.dataset)}")
        self.assertGreater(len(self.dataset), 0)

    def test_getitem(self):
        """Test __getitem__ returns valid data."""
        data, target = self.dataset[0]
        print(f"\n[GasSensorDynamic] 样本0:")
        print(f"  数据形状: {data.shape}")
        print(f"  列名: {list(data.columns)[:5]}...")
        print(f"  标签: {target}")
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn("mixture", target)

    @skipIf(not HAS_TORCH, "PyTorch not installed")
    def test_dataloader(self):
        """Test PyTorch DataLoader integration."""
        loader = DataLoader(
            self.dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=numpy_collate_fn,
        )
        for batch_data, batch_targets in loader:
            print(f"\n[GasSensorDynamic] DataLoader 批次: {len(batch_targets)} 样本")
            self.assertIsInstance(batch_targets, list)
            break


@skipIf(not has_dataset("gas_sensor_array_temperature_modulation"), "Dataset not available")
class TestGasSensorTemperatureModulationWithData(TestCase):
    """Tests for GasSensorTemperatureModulation with actual data."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = GasSensorTemperatureModulation(str(DATA_ROOT), download=False)

    def test_len(self):
        """Test dataset length."""
        print(f"\n[GasSensorTemperatureModulation] 样本数: {len(self.dataset)}")
        self.assertGreater(len(self.dataset), 0)

    def test_getitem(self):
        """Test __getitem__ returns valid data."""
        data, target = self.dataset[0]
        print(f"\n[GasSensorTemperatureModulation] 样本0:")
        print(f"  数据形状: {data.shape}")
        print(f"  列名: {list(data.columns)[:6]}...")
        print(f"  标签: {target}")
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn("day", target)

    @skipIf(not HAS_TORCH, "PyTorch not installed")
    def test_dataloader(self):
        """Test PyTorch DataLoader integration."""
        loader = DataLoader(
            self.dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=numpy_collate_fn,
        )
        for batch_data, batch_targets in loader:
            print(f"\n[GasSensorTemperatureModulation] DataLoader 批次: {len(batch_targets)} 样本")
            self.assertIsInstance(batch_targets, list)
            break


@skipIf(not has_dataset("gas_sensor_array_under_flow_modulation"), "Dataset not available")
class TestGasSensorFlowModulationWithData(TestCase):
    """Tests for GasSensorFlowModulation with actual data."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = GasSensorFlowModulation(str(DATA_ROOT), download=False, use_features=True)

    def test_len(self):
        """Test dataset length."""
        print(f"\n[GasSensorFlowModulation] 样本数: {len(self.dataset)}")
        self.assertGreater(len(self.dataset), 0)

    def test_getitem(self):
        """Test __getitem__ returns valid data."""
        data, target = self.dataset[0]
        print(f"\n[GasSensorFlowModulation] 样本0:")
        print(f"  数据形状: {data.shape}")
        print(f"  标签: {target}")
        self.assertIsInstance(data, np.ndarray)
        self.assertIn("gas", target)
        self.assertIn("ace_conc", target)

    @skipIf(not HAS_TORCH, "PyTorch not installed")
    def test_dataloader(self):
        """Test PyTorch DataLoader integration."""
        loader = DataLoader(
            self.dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=numpy_collate_fn,
        )
        for batch_data, batch_targets in loader:
            print(f"\n[GasSensorFlowModulation] DataLoader 批次:")
            print(f"  批次形状: {batch_data.shape}")
            self.assertIsInstance(batch_data, np.ndarray)
            break


@skipIf(not has_dataset("gas_sensor_array_low_concentration"), "Dataset not available")
class TestGasSensorLowConcentrationWithData(TestCase):
    """Tests for GasSensorLowConcentration with actual data."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = GasSensorLowConcentration(str(DATA_ROOT), download=False, cache=True)

    def test_len(self):
        """Test dataset length (should be exactly 90 samples)."""
        print(f"\n[GasSensorLowConcentration] 样本数: {len(self.dataset)}")
        self.assertEqual(len(self.dataset), 90)

    def test_getitem(self):
        """Test __getitem__ returns valid data."""
        data, target = self.dataset[0]
        print(f"\n[GasSensorLowConcentration] 样本0:")
        print(f"  数据形状: {data.shape}")
        print(f"  标签: {target}")
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.shape, (10, 900))  # 10 sensors, 900 time points
        self.assertIn("gas", target)
        self.assertIn("concentration_ppb", target)

    def test_targets(self):
        """Test targets property."""
        targets = self.dataset.targets
        self.assertEqual(len(targets), 90)
        self.assertTrue(all(0 <= t < 6 for t in targets))

    @skipIf(not HAS_TORCH, "PyTorch not installed")
    def test_dataloader(self):
        """Test PyTorch DataLoader integration."""
        loader = DataLoader(
            self.dataset,
            batch_size=10,
            shuffle=True,
            collate_fn=numpy_collate_fn,
        )
        for batch_data, batch_targets in loader:
            print(f"\n[GasSensorLowConcentration] DataLoader 批次:")
            print(f"  批次形状: {batch_data.shape}")
            self.assertIsInstance(batch_data, np.ndarray)
            self.assertEqual(batch_data.shape[1:], (10, 900))
            break


@skipIf(not has_dataset("alcohol_qcm_sensor_dataset"), "Dataset not available")
class TestAlcoholQCMSensorWithData(TestCase):
    """Tests for AlcoholQCMSensor with actual data."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = AlcoholQCMSensor(str(DATA_ROOT), download=False)

    def test_len(self):
        """Test dataset length."""
        print(f"\n[AlcoholQCMSensor] 样本数: {len(self.dataset)}")
        self.assertGreater(len(self.dataset), 0)

    def test_getitem(self):
        """Test __getitem__ returns valid data."""
        data, target = self.dataset[0]
        print(f"\n[AlcoholQCMSensor] 样本0:")
        print(f"  数据形状: {data.shape}")
        print(f"  标签: {target}")
        self.assertIsInstance(data, np.ndarray)
        self.assertIn("alcohol", target)

    @skipIf(not HAS_TORCH, "PyTorch not installed")
    def test_dataloader(self):
        """Test PyTorch DataLoader integration."""
        loader = DataLoader(
            self.dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=numpy_collate_fn,
        )
        for batch_data, batch_targets in loader:
            print(f"\n[AlcoholQCMSensor] DataLoader 批次:")
            print(f"  批次形状: {batch_data.shape}")
            self.assertIsInstance(batch_data, np.ndarray)
            break


@skipIf(not has_dataset("gas_sensors_for_home_activity_monitoring"), "Dataset not available")
class TestGasSensorsForHomeActivityMonitoringWithData(TestCase):
    """Tests for GasSensorsForHomeActivityMonitoring with actual data."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = GasSensorsForHomeActivityMonitoring(str(DATA_ROOT), download=False)

    def test_len(self):
        """Test dataset length."""
        print(f"\n[GasSensorsForHomeActivityMonitoring] 样本数: {len(self.dataset)}")
        self.assertGreater(len(self.dataset), 0)

    def test_getitem(self):
        """Test __getitem__ returns valid data."""
        data, target = self.dataset[0]
        print(f"\n[GasSensorsForHomeActivityMonitoring] 样本0:")
        print(f"  数据形状: {data.shape}")
        print(f"  列名: {list(data.columns)[:5]}...")
        print(f"  标签: {target}")
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIsInstance(target, int)

    @skipIf(not HAS_TORCH, "PyTorch not installed")
    def test_dataloader(self):
        """Test PyTorch DataLoader integration."""
        loader = DataLoader(
            self.dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=numpy_collate_fn,
        )
        for batch_data, batch_targets in loader:
            print(f"\n[GasSensorsForHomeActivityMonitoring] DataLoader 批次: {len(batch_targets)} 样本")
            self.assertIsInstance(batch_targets, list)
            break


# =============================================================================
# Test: Channel Metadata
# =============================================================================

@skipIf(not has_dataset("twin_gas_sensor_arrays"), "Dataset not available")
class TestChannelMetadata(TestCase):
    """Tests for channel metadata functionality."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = TwinGasSensorArrays(str(DATA_ROOT), download=False)

    def test_channels_property(self):
        """Test channels property returns channel configs."""
        channels = self.dataset.channels
        print(f"\n[ChannelMetadata] Number of channels: {len(channels)}")
        self.assertEqual(len(channels), 8)
        self.assertEqual(channels[0].sensor_model, "TGS2611")

    def test_channel_models(self):
        """Test channel_models property."""
        models = self.dataset.channel_models
        print(f"[ChannelMetadata] Channel models: {models}")
        self.assertEqual(len(models), 8)
        self.assertIn("TGS2611", models)
        self.assertIn("TGS2602", models)

    def test_get_channel(self):
        """Test get_channel method."""
        ch = self.dataset.get_channel(0)
        print(f"[ChannelMetadata] Channel 0: {ch.sensor_model}, {ch.target_gases}")
        self.assertEqual(ch.index, 0)
        self.assertEqual(ch.sensor_model, "TGS2611")
        self.assertIn("Methane", ch.target_gases)

    def test_get_channels_by_model(self):
        """Test get_channels_by_model method."""
        tgs2611_channels = self.dataset.get_channels_by_model("TGS2611")
        print(f"[ChannelMetadata] TGS2611 channels: {tgs2611_channels}")
        self.assertEqual(tgs2611_channels, [0, 4])

    def test_get_channels_by_target_gas(self):
        """Test get_channels_by_target_gas method."""
        voc_channels = self.dataset.get_channels_by_target_gas("VOC")
        print(f"[ChannelMetadata] VOC responsive channels: {voc_channels}")
        self.assertEqual(voc_channels, [3, 7])

    def test_get_channel_metadata_dict(self):
        """Test get_channel_metadata_dict method."""
        meta = self.dataset.get_channel_metadata_dict(0)
        print(f"[ChannelMetadata] Channel 0 metadata: {meta}")
        self.assertIn("sensor_model", meta)
        self.assertIn("target_gases", meta)
        self.assertEqual(meta["sensor_model"], "TGS2611")


# =============================================================================
# Test: Normalized Sample Interface
# =============================================================================

@skipIf(not has_dataset("twin_gas_sensor_arrays"), "Dataset not available")
class TestNormalizedSample(TestCase):
    """Tests for normalized sample interface."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = TwinGasSensorArrays(str(DATA_ROOT), download=False)

    def test_get_normalized_sample(self):
        """Test get_normalized_sample returns correct format."""
        data, meta = self.dataset.get_normalized_sample(0)
        print(f"\n[NormalizedSample] Shape: {data.shape}")
        print(f"[NormalizedSample] Metadata keys: {list(meta.keys())}")
        
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.ndim, 2)  # [C, T] format
        self.assertIn("dataset", meta)
        self.assertIn("channel_models", meta)
        self.assertIn("sample_rate_hz", meta)
        self.assertEqual(meta["dataset"], "twin_gas_sensor_arrays")

    def test_get_sample_with_mask(self):
        """Test get_sample_with_mask returns correct format."""
        masked, mask, meta = self.dataset.get_sample_with_mask(0, mask_channels=[0, 1])
        print(f"\n[MaskedSample] Shape: {masked.shape}, Mask: {mask}")
        print(f"[MaskedSample] Masked channels: {meta['masked_channels']}")
        
        self.assertIsInstance(masked, np.ndarray)
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.dtype, bool)
        self.assertFalse(mask[0])  # Channel 0 should be masked
        self.assertFalse(mask[1])  # Channel 1 should be masked
        self.assertEqual(meta["masked_channels"], [0, 1])

    def test_get_sample_with_mask_ratio(self):
        """Test get_sample_with_mask with mask_ratio."""
        np.random.seed(42)
        masked, mask, meta = self.dataset.get_sample_with_mask(0, mask_ratio=0.5)
        num_masked = len(meta["masked_channels"])
        print(f"\n[MaskedSample] Mask ratio 0.5: {num_masked} channels masked")
        self.assertGreater(num_masked, 0)


# =============================================================================
# Test: Combined Dataset
# =============================================================================

@skipIf(not (has_dataset("twin_gas_sensor_arrays") and has_dataset("gas_sensors_for_home_activity_monitoring")), 
        "Required datasets not available")
class TestCombinedDataset(TestCase):
    """Tests for CombinedEnoseDataset."""

    @classmethod
    def setUpClass(cls):
        from enose_uci_dataset.datasets import CombinedEnoseDataset
        cls.combined = CombinedEnoseDataset(
            root=str(DATA_ROOT),
            datasets=["twin_gas_sensor_arrays", "gas_sensors_for_home_activity_monitoring"],
            download=False
        )

    def test_len(self):
        """Test combined dataset length."""
        total = len(self.combined)
        print(f"\n[CombinedDataset] Total samples: {total}")
        self.assertEqual(total, 640 + 99)  # twin + home

    def test_datasets_property(self):
        """Test datasets property."""
        datasets = self.combined.datasets
        print(f"[CombinedDataset] Loaded datasets: {list(datasets.keys())}")
        self.assertEqual(len(datasets), 2)
        self.assertIn("twin_gas_sensor_arrays", datasets)

    def test_get_normalized_sample(self):
        """Test get_normalized_sample from combined dataset."""
        data, meta = self.combined.get_normalized_sample(0)
        print(f"[CombinedDataset] Sample 0: dataset={meta['dataset']}, shape={data.shape}")
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(meta["dataset"], "twin_gas_sensor_arrays")

    def test_cross_dataset_sensor_lookup(self):
        """Test cross-dataset sensor model lookup."""
        tgs2602_channels = self.combined.get_channels_by_model_cross_dataset("TGS2602")
        print(f"[CombinedDataset] TGS2602 across datasets: {tgs2602_channels}")
        self.assertGreater(len(tgs2602_channels), 2)  # Should be in both datasets

    def test_all_sensor_models(self):
        """Test get_all_sensor_models."""
        models = self.combined.get_all_sensor_models()
        print(f"[CombinedDataset] All sensor models: {models}")
        self.assertIn("TGS2602", models)
        self.assertIn("TGS2611", models)

    def test_summary(self):
        """Test summary method."""
        summary = self.combined.summary()
        print(f"\n{summary}")
        self.assertIn("CombinedEnoseDataset", summary)
        self.assertIn("twin_gas_sensor_arrays", summary)


# =============================================================================
# Test: Pretraining Dataset
# =============================================================================

@skipIf(not has_dataset("twin_gas_sensor_arrays"), "Dataset not available")
class TestPretrainingDataset(TestCase):
    """Tests for PretrainingDataset."""

    @classmethod
    def setUpClass(cls):
        from enose_uci_dataset.datasets import PretrainingDataset
        cls.base_dataset = TwinGasSensorArrays(str(DATA_ROOT), download=False)
        cls.pretrain = PretrainingDataset(cls.base_dataset, mask_ratio=0.25)

    def test_len(self):
        """Test pretraining dataset length."""
        self.assertEqual(len(self.pretrain), len(self.base_dataset))

    def test_getitem(self):
        """Test __getitem__ returns masked sample."""
        np.random.seed(42)
        masked, mask, meta = self.pretrain[0]
        print(f"\n[PretrainingDataset] Shape: {masked.shape}")
        print(f"[PretrainingDataset] Masked channels: {meta['masked_channels']}")
        
        self.assertIsInstance(masked, np.ndarray)
        self.assertIsInstance(mask, np.ndarray)
        self.assertIn("masked_channels", meta)


if __name__ == "__main__":
    main()
