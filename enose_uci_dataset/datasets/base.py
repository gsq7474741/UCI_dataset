"""
数据集基类 - 类torchvision设计

核心改进:
1. 懒加载 - 按需读取样本，节省内存
2. 索引机制 - 预构建样本索引
3. 统一接口 - 兼容PyTorch DataLoader
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import get_config_manager, DatasetInfo


class BaseDataset(ABC):
    """
    数据集抽象基类
    
    兼容torch.utils.data.Dataset接口
    """
    
    @abstractmethod
    def __len__(self) -> int:
        pass
    
    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        pass
    
    def __iter__(self) -> Iterator:
        for i in range(len(self)):
            yield self[i]


class SampleIndex:
    """
    样本索引 - 支持懒加载
    
    存储样本的元信息，不加载实际数据
    """
    
    def __init__(
        self,
        sample_id: str,
        file_path: Path,
        label: Optional[Any] = None,
        metadata: Optional[Dict] = None,
    ):
        self.sample_id = sample_id
        self.file_path = file_path
        self.label = label
        self.metadata = metadata or {}
    
    def load(self) -> pd.DataFrame:
        """懒加载：读取实际数据"""
        return pd.read_csv(self.file_path)
    
    def __repr__(self) -> str:
        return f"SampleIndex(id={self.sample_id}, label={self.label})"


class ENoseDataset(BaseDataset):
    """
    电子鼻数据集基类
    
    Args:
        root: 数据根目录
        split: 数据集划分 ("train", "val", "test", None=全部)
        transform: 数据转换 (Callable[[pd.DataFrame], pd.DataFrame])
        target_transform: 标签转换
        download: 是否自动下载
    
    子类需要实现:
        - name: 数据集名称 (类属性)
        - _build_index(): 构建样本索引
    """
    
    # 子类必须定义
    name: str = ""
    
    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        if not self.name:
            raise ValueError("子类必须定义 name 属性")
        
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # 获取配置
        manager = get_config_manager()
        self.info = manager.get_dataset(self.name)
        
        # 设置根目录
        if root is not None:
            self.root = Path(root)
        else:
            self.root = manager.get_dataset_dir(self.name)
        
        # 自动下载
        if download:
            self._download()
        
        # 构建索引 (懒加载的关键)
        self._index: List[SampleIndex] = []
        self._build_index()
    
    def _download(self) -> None:
        """下载数据集"""
        from ..downloader import download
        download(self.name, extract=True)
    
    @abstractmethod
    def _build_index(self) -> None:
        """
        构建样本索引 - 子类必须实现
        
        应该填充 self._index 列表
        """
        pass
    
    def __len__(self) -> int:
        return len(self._index)
    
    def __getitem__(self, index: int) -> Tuple[pd.DataFrame, Any]:
        """
        懒加载获取样本
        
        Returns:
            (data, label) 元组
        """
        if index < 0 or index >= len(self._index):
            raise IndexError(f"Index {index} out of range [0, {len(self._index)})")
        
        sample = self._index[index]
        
        # 懒加载数据
        data = sample.load()
        label = sample.label
        
        # 应用转换
        if self.transform is not None:
            data = self.transform(data)
        
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return data, label
    
    def get_labels(self) -> List[Any]:
        """获取所有标签"""
        return [s.label for s in self._index]
    
    @property
    def sensor_columns(self) -> List[str]:
        """获取传感器列名 (从第一个样本推断)"""
        if not self._index:
            return []
        data = self._index[0].load()
        return [c for c in data.columns if c.startswith("sensor")]
    
    @property
    def num_sensors(self) -> int:
        return len(self.sensor_columns)
    
    @property
    def sample_rate(self) -> Optional[int]:
        """采样率 (Hz)"""
        if self.info.time_series:
            return self.info.time_series.sample_rate_hz
        return None
    
    @property
    def classes(self) -> List[Any]:
        """获取所有类别"""
        return sorted(set(self.get_labels()))
    
    @property
    def num_classes(self) -> int:
        return len(self.classes)
    
    def to_numpy(
        self,
        max_length: Optional[int] = None,
        pad_value: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        转换为numpy数组 (用于传统ML)
        
        Returns:
            X: shape (N, T, C) 或 (N, C) 如果取均值
            y: shape (N,)
        """
        sequences = []
        labels = []
        
        for data, label in self:
            sensor_cols = [c for c in data.columns if c.startswith("sensor")]
            seq = data[sensor_cols].values
            sequences.append(seq)
            labels.append(label)
        
        if max_length is None:
            max_length = max(len(s) for s in sequences)
        
        n_sensors = sequences[0].shape[1] if sequences else 0
        X = np.full((len(sequences), max_length, n_sensors), pad_value)
        
        for i, seq in enumerate(sequences):
            length = min(len(seq), max_length)
            X[i, :length, :] = seq[:length]
        
        return X, np.array(labels)
    
    def to_features(self, aggregation: str = "mean") -> Tuple[np.ndarray, np.ndarray]:
        """
        提取特征 (用于传统分类器)
        
        Args:
            aggregation: 聚合方式 ("mean", "max", "min", "std", "all")
        
        Returns:
            X: shape (N, C) 或 (N, C*4) if aggregation="all"
            y: shape (N,)
        """
        features = []
        labels = []
        
        for data, label in self:
            sensor_cols = [c for c in data.columns if c.startswith("sensor")]
            vals = data[sensor_cols].values
            
            if aggregation == "mean":
                feat = vals.mean(axis=0)
            elif aggregation == "max":
                feat = vals.max(axis=0)
            elif aggregation == "min":
                feat = vals.min(axis=0)
            elif aggregation == "std":
                feat = vals.std(axis=0)
            elif aggregation == "all":
                feat = np.concatenate([
                    vals.mean(axis=0),
                    vals.std(axis=0),
                    vals.min(axis=0),
                    vals.max(axis=0),
                ])
            else:
                feat = vals.mean(axis=0)
            
            features.append(feat)
            labels.append(label)
        
        return np.array(features), np.array(labels)
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  root={self.root},\n"
            f"  split={self.split},\n"
            f"  samples={len(self)},\n"
            f"  classes={self.num_classes},\n"
            f"  sensors={self.num_sensors},\n"
            f")"
        )


class DataLoader:
    """
    简单数据加载器
    
    兼容PyTorch DataLoader接口，如使用PyTorch建议用torch.utils.data.DataLoader
    """
    
    def __init__(
        self,
        dataset: BaseDataset,
        batch_size: int = 32,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
    
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        batch = []
        for idx in indices:
            batch.append(self.dataset[idx])
            
            if len(batch) == self.batch_size:
                yield self._collate(batch)
                batch = []
        
        if batch and not self.drop_last:
            yield self._collate(batch)
    
    def _collate(self, batch: List[Tuple]) -> Tuple[List, List]:
        data_list = [item[0] for item in batch]
        label_list = [item[1] for item in batch]
        return data_list, label_list
    
    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
