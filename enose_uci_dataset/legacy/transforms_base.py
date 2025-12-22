"""
数据转换模块 - 类似torchvision.transforms的设计

层级架构:
1. Transform (基类) - 单个转换操作
2. Compose - 组合多个转换
3. 具体转换类 - 标准化、重采样、切片等
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class Transform(ABC):
    """转换基类"""
    
    @abstractmethod
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """应用转换"""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Compose(Transform):
    """组合多个转换"""
    
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        for t in self.transforms:
            data = t(data)
        return data
    
    def __repr__(self) -> str:
        lines = [f"  {t}" for t in self.transforms]
        return f"Compose([\n" + ",\n".join(lines) + "\n])"


# ============ 列操作转换 ============

class SelectColumns(Transform):
    """选择指定列"""
    
    def __init__(self, columns: List[str]):
        self.columns = columns
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self.columns if c in data.columns]
        return data[available].copy()


class RenameColumns(Transform):
    """重命名列"""
    
    def __init__(self, mapping: Dict[str, str]):
        self.mapping = mapping
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.rename(columns=self.mapping)


class NormalizeSensorColumns(Transform):
    """
    标准化传感器列名
    将 R1, R2, ... 或 sensor0, sensor1, ... 统一为 sensor_0, sensor_1, ...
    """
    
    def __init__(self, prefix: str = "sensor_"):
        self.prefix = prefix
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        mapping = {}
        sensor_idx = 0
        
        for col in data.columns:
            col_lower = col.lower()
            # 匹配 R1, R2, ... 模式
            if col.startswith("R") and col[1:].isdigit():
                mapping[col] = f"{self.prefix}{int(col[1:]) - 1}"
            # 匹配 sensor0, sensor1, ... 模式
            elif col_lower.startswith("sensor") and col_lower[6:].replace("_", "").isdigit():
                idx = int(col_lower[6:].replace("_", ""))
                mapping[col] = f"{self.prefix}{idx}"
        
        return data.rename(columns=mapping)


# ============ 时间转换 ============

class ConvertTimeUnit(Transform):
    """转换时间单位"""
    
    def __init__(
        self,
        time_column: str = "time",
        from_unit: str = "h",
        to_unit: str = "s",
        output_column: Optional[str] = None,
    ):
        self.time_column = time_column
        self.from_unit = from_unit
        self.to_unit = to_unit
        self.output_column = output_column or f"t_{to_unit}"
        
        # 单位转换因子 (转为秒)
        self._to_seconds = {
            "ms": 0.001,
            "s": 1.0,
            "min": 60.0,
            "h": 3600.0,
        }
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.time_column not in data.columns:
            return data
        
        from_factor = self._to_seconds.get(self.from_unit, 1.0)
        to_factor = self._to_seconds.get(self.to_unit, 1.0)
        
        data = data.copy()
        data[self.output_column] = data[self.time_column] * from_factor / to_factor
        return data


class AddRelativeTime(Transform):
    """添加相对时间列 (从0开始)"""
    
    def __init__(self, sample_rate_hz: int = 100, output_column: str = "t_s"):
        self.sample_rate_hz = sample_rate_hz
        self.output_column = output_column
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data[self.output_column] = np.arange(len(data)) / self.sample_rate_hz
        return data


# ============ 数值转换 ============

class ToResistance(Transform):
    """
    将传感器读数转换为电阻值
    常见公式: R = k / V 或 R = k * V
    """
    
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        formula: str = "divide",  # "divide" or "multiply"
        factor: float = 40.0,
        output_unit: str = "kohm",
    ):
        self.columns = columns
        self.formula = formula
        self.factor = factor
        self.output_unit = output_unit
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        
        # 自动检测传感器列
        if self.columns is None:
            cols = [c for c in data.columns if c.startswith("sensor")]
        else:
            cols = [c for c in self.columns if c in data.columns]
        
        for col in cols:
            vals = data[col].values.astype(float)
            eps = 1e-12
            
            if self.formula == "divide":
                with np.errstate(divide='ignore', invalid='ignore'):
                    data[col] = np.where(np.abs(vals) > eps, self.factor / vals, np.nan)
            else:  # multiply
                data[col] = vals * self.factor
        
        return data


class Normalize(Transform):
    """数值归一化"""
    
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        method: str = "minmax",  # "minmax" | "zscore" | "robust"
    ):
        self.columns = columns
        self.method = method
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        
        if self.columns is None:
            cols = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = [c for c in self.columns if c in data.columns]
        
        for col in cols:
            vals = data[col].values.astype(float)
            
            if self.method == "minmax":
                vmin, vmax = np.nanmin(vals), np.nanmax(vals)
                if vmax - vmin > 1e-12:
                    data[col] = (vals - vmin) / (vmax - vmin)
            elif self.method == "zscore":
                mean, std = np.nanmean(vals), np.nanstd(vals)
                if std > 1e-12:
                    data[col] = (vals - mean) / std
            elif self.method == "robust":
                median = np.nanmedian(vals)
                q1, q3 = np.nanpercentile(vals, [25, 75])
                iqr = q3 - q1
                if iqr > 1e-12:
                    data[col] = (vals - median) / iqr
        
        return data


class FillNaN(Transform):
    """填充缺失值"""
    
    def __init__(self, method: str = "ffill", value: Optional[float] = None):
        self.method = method
        self.value = value
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.value is not None:
            return data.fillna(self.value)
        elif self.method == "ffill":
            return data.ffill().bfill()
        elif self.method == "bfill":
            return data.bfill().ffill()
        elif self.method == "interpolate":
            return data.interpolate(method="linear").ffill().bfill()
        return data


# ============ 重采样转换 ============

class Resample(Transform):
    """
    重采样时序数据
    """
    
    def __init__(
        self,
        source_rate_hz: int,
        target_rate_hz: int,
        time_column: str = "t_s",
    ):
        self.source_rate = source_rate_hz
        self.target_rate = target_rate_hz
        self.time_column = time_column
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.source_rate == self.target_rate:
            return data
        
        ratio = self.source_rate / self.target_rate
        
        if ratio > 1:
            # 降采样
            step = int(ratio)
            return data.iloc[::step].reset_index(drop=True)
        else:
            # 上采样 (线性插值)
            n_new = int(len(data) / ratio)
            new_idx = np.linspace(0, len(data) - 1, n_new)
            
            result = {}
            for col in data.columns:
                if data[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    result[col] = np.interp(new_idx, np.arange(len(data)), data[col].values)
                else:
                    # 非数值列取最近邻
                    result[col] = data[col].iloc[np.round(new_idx).astype(int)].values
            
            return pd.DataFrame(result)


# ============ 切片转换 ============

class SlidingWindow(Transform):
    """
    滑动窗口切片 - 将连续数据切分为固定长度的样本
    
    返回: List[pd.DataFrame] 而非单个DataFrame
    """
    
    def __init__(
        self,
        window_size: int,
        stride: int,
        drop_last: bool = True,
    ):
        self.window_size = window_size
        self.stride = stride
        self.drop_last = drop_last
    
    def __call__(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        samples = []
        n = len(data)
        
        for start in range(0, n - self.window_size + 1, self.stride):
            end = start + self.window_size
            samples.append(data.iloc[start:end].reset_index(drop=True))
        
        # 处理最后一个不完整的窗口
        if not self.drop_last and n > self.window_size:
            last_start = (n - self.window_size)
            if last_start % self.stride != 0:
                samples.append(data.iloc[-self.window_size:].reset_index(drop=True))
        
        return samples


class SegmentByLabel(Transform):
    """
    按标签变化自动分段 - 用于连续采样数据
    
    返回: List[pd.DataFrame]
    """
    
    def __init__(self, label_columns: List[str], min_segment_length: int = 10):
        self.label_columns = label_columns
        self.min_length = min_segment_length
    
    def __call__(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        available_cols = [c for c in self.label_columns if c in data.columns]
        if not available_cols:
            return [data]
        
        # 检测标签变化点
        change_points = [0]
        for i in range(1, len(data)):
            changed = False
            for col in available_cols:
                if data.iloc[i][col] != data.iloc[i-1][col]:
                    changed = True
                    break
            if changed:
                change_points.append(i)
        change_points.append(len(data))
        
        # 切分
        segments = []
        for i in range(len(change_points) - 1):
            start, end = change_points[i], change_points[i+1]
            if end - start >= self.min_length:
                segments.append(data.iloc[start:end].reset_index(drop=True))
        
        return segments


# ============ 过滤转换 ============

class DropNA(Transform):
    """删除含有缺失值的行"""
    
    def __init__(self, subset: Optional[List[str]] = None):
        self.subset = subset
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.dropna(subset=self.subset).reset_index(drop=True)


class FilterByValue(Transform):
    """按值过滤"""
    
    def __init__(self, column: str, min_val: Optional[float] = None, max_val: Optional[float] = None):
        self.column = column
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.column not in data.columns:
            return data
        
        mask = pd.Series([True] * len(data))
        if self.min_val is not None:
            mask &= data[self.column] >= self.min_val
        if self.max_val is not None:
            mask &= data[self.column] <= self.max_val
        
        return data[mask].reset_index(drop=True)


# ============ 便捷函数 ============

def get_default_transforms(
    normalize_columns: bool = True,
    time_unit: str = "s",
    fill_nan: bool = True,
) -> Compose:
    """获取默认转换管道"""
    transforms = []
    
    if normalize_columns:
        transforms.append(NormalizeSensorColumns())
    
    if fill_nan:
        transforms.append(FillNaN(method="ffill"))
    
    return Compose(transforms)
