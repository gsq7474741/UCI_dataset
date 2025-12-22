"""
数据集模块 - 类似torchvision.datasets的设计

层级架构:
1. BaseDataset - 数据集基类
2. ENoseDataset - 电子鼻数据集通用类
3. 具体数据集类 - 继承自ENoseDataset，实现特定的加载逻辑
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from .config_loader import get_config_manager, DatasetInfo
from .transforms import Transform, Compose


class BaseDataset(ABC):
    """
    数据集基类
    
    设计参考torch.utils.data.Dataset
    """
    
    @abstractmethod
    def __len__(self) -> int:
        """返回数据集大小"""
        pass
    
    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """获取单个样本"""
        pass
    
    def __iter__(self) -> Iterator:
        """迭代所有样本"""
        for i in range(len(self)):
            yield self[i]


class ENoseDataset(BaseDataset):
    """
    电子鼻数据集通用类
    
    Args:
        name: 数据集名称
        root: 数据根目录（可选，默认使用配置）
        split: 数据集划分 ("train", "val", "test", None=全部)
        transform: 数据转换
        target_transform: 标签转换
        download: 是否自动下载
    """
    
    def __init__(
        self,
        name: str,
        root: Optional[str] = None,
        split: Optional[str] = None,
        transform: Optional[Transform] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        self.name = name
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # 获取数据集信息
        manager = get_config_manager()
        self.info = manager.get_dataset(name)
        
        # 设置根目录
        if root is not None:
            self.root = Path(root)
        else:
            self.root = manager.get_dataset_dir(name)
        
        # 自动下载
        if download:
            self._download()
        
        # 检查数据是否存在
        self._check_exists()
        
        # 加载数据
        self.samples: List[pd.DataFrame] = []
        self.labels: List[Any] = []
        self._load_data()
    
    def _download(self) -> None:
        """下载数据集"""
        from .downloader import download
        download(self.name, extract=True)
    
    def _check_exists(self) -> bool:
        """检查数据是否存在"""
        processed_dir = self.root / "processed"
        raw_dir = self.root / "raw"
        
        if not processed_dir.exists() and not raw_dir.exists():
            raise RuntimeError(
                f"数据集 {self.name} 不存在。"
                f"请先下载: download('{self.name}')"
            )
        return True
    
    def _load_data(self) -> None:
        """
        加载数据 - 子类应重写此方法
        
        默认实现：从processed目录加载CSV文件
        """
        processed_dir = self.root / "processed" / "v1" / "ssl_samples"
        
        if not processed_dir.exists():
            # 尝试raw目录
            return
        
        # 获取分割文件（如果存在）
        split_file = self.root / "dataset_split.csv"
        split_filter = None
        
        if split_file.exists() and self.split:
            split_df = pd.read_csv(split_file)
            if self.split in split_df.columns:
                split_filter = set(split_df[split_df[self.split] == 1]["sample_id"].tolist())
        
        # 加载所有CSV文件
        for csv_path in sorted(processed_dir.glob("*.csv")):
            sample_id = csv_path.stem
            
            # 应用分割过滤
            if split_filter is not None and sample_id not in split_filter:
                continue
            
            df = pd.read_csv(csv_path)
            self.samples.append(df)
            
            # 提取标签（假设列名包含label_）
            label_cols = [c for c in df.columns if c.startswith("label")]
            if label_cols:
                self.labels.append(df[label_cols[0]].iloc[0])
            else:
                self.labels.append(None)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[pd.DataFrame, Any]:
        """
        获取单个样本
        
        Returns:
            (data, label) 元组
        """
        if index < 0 or index >= len(self.samples):
            raise IndexError(f"Index {index} out of range [0, {len(self.samples)})")
        
        data = self.samples[index].copy()
        label = self.labels[index]
        
        # 应用转换
        if self.transform is not None:
            data = self.transform(data)
        
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return data, label
    
    def get_all_data(self) -> pd.DataFrame:
        """获取所有数据（合并为单个DataFrame）"""
        if not self.samples:
            return pd.DataFrame()
        
        dfs = []
        for i, df in enumerate(self.samples):
            df_copy = df.copy()
            df_copy["sample_id"] = i
            dfs.append(df_copy)
        
        return pd.concat(dfs, ignore_index=True)
    
    @property
    def sensor_columns(self) -> List[str]:
        """获取传感器列名"""
        if not self.samples:
            return []
        return [c for c in self.samples[0].columns if c.startswith("sensor")]
    
    @property
    def num_sensors(self) -> int:
        """传感器数量"""
        return len(self.sensor_columns)
    
    @property
    def sample_rate(self) -> Optional[int]:
        """采样率 (Hz)"""
        if self.info.time_series:
            return self.info.time_series.sample_rate_hz
        return None
    
    def for_classification(
        self,
        label_column: str = "label_gas",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备分类任务数据
        
        Returns:
            (X, y) - 特征矩阵和标签向量
        """
        X_list = []
        y_list = []
        
        for data, label in self:
            # 提取传感器数据
            sensor_data = data[self.sensor_columns].values
            
            # 可以使用均值、最后一个值等作为特征
            features = sensor_data.mean(axis=0)  # 时序均值
            X_list.append(features)
            
            if label_column in data.columns:
                y_list.append(data[label_column].iloc[0])
            else:
                y_list.append(label)
        
        return np.array(X_list), np.array(y_list)
    
    def for_sequence(
        self,
        max_length: Optional[int] = None,
        pad_value: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备序列任务数据（时序分类、预测等）
        
        Returns:
            (X, y) - 形状 (N, T, C) 和 (N,)
        """
        sequences = []
        labels = []
        
        for data, label in self:
            seq = data[self.sensor_columns].values
            sequences.append(seq)
            labels.append(label)
        
        # 找到最大长度
        if max_length is None:
            max_length = max(len(s) for s in sequences)
        
        # 填充到相同长度
        X = np.full((len(sequences), max_length, self.num_sensors), pad_value)
        for i, seq in enumerate(sequences):
            length = min(len(seq), max_length)
            X[i, :length, :] = seq[:length]
        
        return X, np.array(labels)
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  name={self.name},\n"
            f"  split={self.split},\n"
            f"  samples={len(self.samples)},\n"
            f"  sensors={self.num_sensors},\n"
            f"  transform={self.transform}\n"
            f")"
        )


# ============ 数据加载器适配 ============

class DataLoader:
    """
    简单的数据加载器（兼容PyTorch DataLoader接口）
    
    如果使用PyTorch，建议直接使用torch.utils.data.DataLoader
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
    
    def _collate(self, batch: List[Tuple]) -> Tuple:
        """合并批次数据"""
        data_list = [item[0] for item in batch]
        label_list = [item[1] for item in batch]
        return data_list, label_list
    
    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


# ============ 便捷函数 ============

def load_dataset(
    name: str,
    split: Optional[str] = None,
    transform: Optional[Transform] = None,
    download: bool = False,
) -> ENoseDataset:
    """
    加载数据集
    
    Args:
        name: 数据集名称
        split: 数据划分 ("train", "val", "test", None)
        transform: 数据转换
        download: 是否自动下载
    
    Returns:
        ENoseDataset实例
    
    Example:
        >>> ds = load_dataset("gas_sensors_for_home_activity_monitoring")
        >>> print(len(ds))
        >>> data, label = ds[0]
    """
    return ENoseDataset(
        name=name,
        split=split,
        transform=transform,
        download=download,
    )
