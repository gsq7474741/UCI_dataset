"""
Gas Sensors for Home Activity Monitoring 数据集
"""

from pathlib import Path
from typing import Optional, Callable, Union

from .base import ENoseDataset, SampleIndex


class HomeActivityDataset(ENoseDataset):
    """
    Gas Sensors for Home Activity Monitoring Dataset
    
    UCI ID: 362
    任务: 活动识别、气体分类
    传感器: 8个MOX传感器
    采样率: ~1Hz
    
    类别:
        0: banana
        1: wine  
        2: background
    
    Example:
        >>> dataset = HomeActivityDataset(download=True)
        >>> data, label = dataset[0]
        >>> print(f"样本数: {len(dataset)}, 类别数: {dataset.num_classes}")
    """
    
    name = "gas_sensors_for_home_activity_monitoring"
    
    classes_map = {
        "banana": 0,
        "wine": 1,
        "background": 2,
    }
    
    def _build_index(self) -> None:
        """构建样本索引"""
        processed_dir = self.root / "processed" / "v1" / "ssl_samples"
        
        if not processed_dir.exists():
            return
        
        # 加载分割信息
        split_filter = self._load_split_filter()
        
        for csv_path in sorted(processed_dir.glob("*.csv")):
            sample_id = csv_path.stem
            
            # 应用分割过滤
            if split_filter is not None and sample_id not in split_filter:
                continue
            
            # 从文件名解析标签: {id}_{class}.csv
            parts = sample_id.rsplit("_", 1)
            if len(parts) == 2:
                label_str = parts[1]
                label = self.classes_map.get(label_str, -1)
            else:
                label = -1
            
            self._index.append(SampleIndex(
                sample_id=sample_id,
                file_path=csv_path,
                label=label,
                metadata={"class_name": parts[1] if len(parts) == 2 else None},
            ))
    
    def _load_split_filter(self) -> Optional[set]:
        """加载数据集分割"""
        if self.split is None:
            return None
        
        split_file = self.root / "dataset_split.csv"
        if not split_file.exists():
            return None
        
        import pandas as pd
        split_df = pd.read_csv(split_file)
        
        if self.split in split_df.columns:
            return set(split_df[split_df[self.split] == 1]["sample_id"].astype(str).tolist())
        
        return None
