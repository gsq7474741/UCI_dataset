"""
Gas Sensor Array Low-Concentration 数据集
"""

from pathlib import Path
from typing import Optional

from .base import ENoseDataset, SampleIndex


class LowConcentrationDataset(ENoseDataset):
    """
    Gas Sensor Array Low-Concentration Dataset
    
    UCI ID: 1081
    任务: 气体分类、浓度预测
    传感器: 8个MOX传感器
    采样率: 100Hz
    
    特点: 低浓度气体检测，连续采样
    
    Example:
        >>> dataset = LowConcentrationDataset(download=True)
        >>> data, label = dataset[0]
    """
    
    name = "gas_sensor_array_low_concentration"
    
    def _build_index(self) -> None:
        """构建样本索引"""
        processed_dir = self.root / "processed" / "v1" / "ssl_samples"
        
        if not processed_dir.exists():
            return
        
        for csv_path in sorted(processed_dir.glob("*.csv")):
            sample_id = csv_path.stem
            label = self._parse_label(sample_id)
            
            self._index.append(SampleIndex(
                sample_id=sample_id,
                file_path=csv_path,
                label=label,
            ))
    
    def _parse_label(self, sample_id: str) -> Optional[int]:
        """从样本ID解析标签"""
        # 根据实际文件命名规则实现
        return None
