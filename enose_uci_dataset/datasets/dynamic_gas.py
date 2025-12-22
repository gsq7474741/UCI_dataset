"""
Gas Sensor Array under Dynamic Gas Mixtures 数据集
"""

from pathlib import Path
from typing import Optional

from .base import ENoseDataset, SampleIndex


class DynamicGasDataset(ENoseDataset):
    """
    Gas Sensor Array under Dynamic Gas Mixtures Dataset
    
    UCI ID: 322
    任务: 气体分类、浓度预测
    传感器: 16个MOX传感器
    采样率: 100Hz
    气体: Ethylene + CO 或 Ethylene + Methane
    
    特点: 连续采样的时序数据，需要切片处理
    
    Example:
        >>> dataset = DynamicGasDataset(download=True)
        >>> data, label = dataset[0]
    """
    
    name = "gas_sensor_array_under_dynamic_gas_mixtures"
    
    def _build_index(self) -> None:
        """构建样本索引"""
        processed_dir = self.root / "processed" / "v1" / "ssl_samples"
        
        if not processed_dir.exists():
            return
        
        for csv_path in sorted(processed_dir.glob("*.csv")):
            sample_id = csv_path.stem
            
            # 解析标签: co_00001.csv 或 methane_00001.csv
            if sample_id.startswith("co_"):
                label = 0  # CO
            elif sample_id.startswith("methane_"):
                label = 1  # Methane
            else:
                label = -1
            
            self._index.append(SampleIndex(
                sample_id=sample_id,
                file_path=csv_path,
                label=label,
                metadata={"gas_type": "co" if label == 0 else "methane"},
            ))
