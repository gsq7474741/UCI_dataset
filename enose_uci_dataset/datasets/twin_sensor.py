"""
Twin Gas Sensor Arrays 数据集
"""

from pathlib import Path
from typing import Optional, Callable, Union

from .base import ENoseDataset, SampleIndex


class TwinSensorDataset(ENoseDataset):
    """
    Twin Gas Sensor Arrays Dataset
    
    UCI ID: 361
    任务: 漂移补偿、迁移学习、气体分类
    传感器: 2组 x 4个MOX传感器 = 8个
    采样率: 100Hz
    
    特点: 两个相同的传感器阵列，可用于校准迁移研究
    
    Example:
        >>> dataset = TwinSensorDataset(download=True)
        >>> data, label = dataset[0]
    """
    
    name = "twin_gas_sensor_arrays"
    
    def _build_index(self) -> None:
        """构建样本索引"""
        processed_dir = self.root / "processed" / "v1" / "ssl_samples"
        
        if not processed_dir.exists():
            # 尝试raw目录
            raw_dir = self.root / "raw"
            if raw_dir.exists():
                self._build_index_from_raw(raw_dir)
            return
        
        for csv_path in sorted(processed_dir.glob("*.csv")):
            sample_id = csv_path.stem
            
            # 从文件名解析标签
            label = self._parse_label(sample_id)
            
            self._index.append(SampleIndex(
                sample_id=sample_id,
                file_path=csv_path,
                label=label,
            ))
    
    def _build_index_from_raw(self, raw_dir: Path) -> None:
        """从原始数据构建索引"""
        for dat_path in sorted(raw_dir.glob("*.dat")):
            sample_id = dat_path.stem
            
            self._index.append(SampleIndex(
                sample_id=sample_id,
                file_path=dat_path,
                label=None,
                metadata={"format": "dat"},
            ))
    
    def _parse_label(self, sample_id: str) -> Optional[int]:
        """从样本ID解析标签"""
        # 根据实际文件命名规则实现
        return None
