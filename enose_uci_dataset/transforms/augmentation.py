"""
时序数据增强 - 针对电子鼻传感器数据

增强方法参考时序数据和传感器数据的常用技术
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union
from abc import ABC, abstractmethod


class Transform(ABC):
    """转换基类 (本地定义避免循环导入)"""
    
    @abstractmethod
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AddGaussianNoise(Transform):
    """
    添加高斯噪声
    
    模拟传感器噪声和测量误差
    """
    
    def __init__(
        self,
        std: float = 0.01,
        columns: Optional[List[str]] = None,
        relative: bool = True,
    ):
        """
        Args:
            std: 噪声标准差
            columns: 要添加噪声的列，None表示所有传感器列
            relative: 是否相对于信号幅值
        """
        self.std = std
        self.columns = columns
        self.relative = relative
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        
        if self.columns is None:
            cols = [c for c in data.columns if c.startswith("sensor")]
        else:
            cols = [c for c in self.columns if c in data.columns]
        
        for col in cols:
            vals = data[col].values.astype(float)
            
            if self.relative:
                noise_std = self.std * np.abs(vals).mean()
            else:
                noise_std = self.std
            
            noise = np.random.normal(0, noise_std, len(vals))
            data[col] = vals + noise
        
        return data


class TimeWarp(Transform):
    """
    时间扭曲增强
    
    通过非线性时间变换模拟采样率变化
    """
    
    def __init__(self, sigma: float = 0.2, num_knots: int = 4):
        self.sigma = sigma
        self.num_knots = num_knots
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        n = len(data)
        
        # 创建扭曲曲线
        orig_steps = np.arange(n)
        random_warps = np.random.normal(1.0, self.sigma, self.num_knots + 2)
        warp_steps = np.linspace(0, n - 1, self.num_knots + 2)
        
        # 插值得到新的时间点
        from scipy.interpolate import CubicSpline
        try:
            cs = CubicSpline(warp_steps, warp_steps * random_warps)
            warped_steps = cs(orig_steps)
            warped_steps = np.clip(warped_steps, 0, n - 1)
        except:
            return data
        
        # 对数值列进行插值
        for col in data.columns:
            if data[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                try:
                    data[col] = np.interp(warped_steps, orig_steps, data[col].values)
                except:
                    pass
        
        return data


class Scaling(Transform):
    """
    幅度缩放增强
    
    模拟传感器灵敏度变化
    """
    
    def __init__(
        self,
        scale_range: tuple = (0.8, 1.2),
        columns: Optional[List[str]] = None,
        per_column: bool = True,
    ):
        self.scale_range = scale_range
        self.columns = columns
        self.per_column = per_column
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        
        if self.columns is None:
            cols = [c for c in data.columns if c.startswith("sensor")]
        else:
            cols = [c for c in self.columns if c in data.columns]
        
        if self.per_column:
            for col in cols:
                scale = np.random.uniform(*self.scale_range)
                data[col] = data[col] * scale
        else:
            scale = np.random.uniform(*self.scale_range)
            for col in cols:
                data[col] = data[col] * scale
        
        return data


class Permutation(Transform):
    """
    段落重排增强
    
    将时序分段后随机重排，用于提取不依赖时序顺序的特征
    """
    
    def __init__(self, num_segments: int = 4):
        self.num_segments = num_segments
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        n = len(data)
        if n < self.num_segments:
            return data
        
        segment_len = n // self.num_segments
        segments = []
        
        for i in range(self.num_segments):
            start = i * segment_len
            end = start + segment_len if i < self.num_segments - 1 else n
            segments.append(data.iloc[start:end])
        
        np.random.shuffle(segments)
        return pd.concat(segments, ignore_index=True)


class Rotation(Transform):
    """
    传感器旋转增强
    
    随机交换传感器列，模拟传感器位置变化
    """
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        
        sensor_cols = [c for c in data.columns if c.startswith("sensor")]
        if len(sensor_cols) < 2:
            return data
        
        # 随机排列传感器列
        shuffled = sensor_cols.copy()
        np.random.shuffle(shuffled)
        
        # 重命名
        rename_map = {old: new for old, new in zip(shuffled, sensor_cols)}
        
        # 复制值
        temp = {}
        for col in sensor_cols:
            temp[col] = data[col].values.copy()
        
        for old, new in rename_map.items():
            data[new] = temp[old]
        
        return data


class MagnitudeWarp(Transform):
    """
    幅度扭曲增强
    
    对信号幅度进行平滑的非线性变换
    """
    
    def __init__(self, sigma: float = 0.2, num_knots: int = 4):
        self.sigma = sigma
        self.num_knots = num_knots
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        n = len(data)
        
        sensor_cols = [c for c in data.columns if c.startswith("sensor")]
        
        for col in sensor_cols:
            # 生成平滑的缩放曲线
            orig_steps = np.arange(n)
            random_warps = np.random.normal(1.0, self.sigma, self.num_knots + 2)
            warp_steps = np.linspace(0, n - 1, self.num_knots + 2)
            
            try:
                from scipy.interpolate import CubicSpline
                cs = CubicSpline(warp_steps, random_warps)
                warp_factors = cs(orig_steps)
                data[col] = data[col].values * warp_factors
            except:
                pass
        
        return data


class Jitter(Transform):
    """
    抖动增强
    
    在时间维度上添加小的随机偏移
    """
    
    def __init__(self, sigma: float = 0.03):
        self.sigma = sigma
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        
        sensor_cols = [c for c in data.columns if c.startswith("sensor")]
        
        for col in sensor_cols:
            vals = data[col].values.astype(float)
            noise = np.random.normal(0, self.sigma * vals.std(), len(vals))
            data[col] = vals + noise
        
        return data


class Dropout(Transform):
    """
    随机丢弃
    
    随机将部分值设为0，模拟传感器故障
    """
    
    def __init__(self, p: float = 0.1, columns: Optional[List[str]] = None):
        self.p = p
        self.columns = columns
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        
        if self.columns is None:
            cols = [c for c in data.columns if c.startswith("sensor")]
        else:
            cols = [c for c in self.columns if c in data.columns]
        
        for col in cols:
            mask = np.random.random(len(data)) < self.p
            data.loc[mask, col] = 0
        
        return data


class RandomCrop(Transform):
    """
    随机裁剪
    
    从序列中随机裁剪固定长度的片段
    """
    
    def __init__(self, length: int):
        self.length = length
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        n = len(data)
        
        if n <= self.length:
            return data
        
        start = np.random.randint(0, n - self.length)
        return data.iloc[start:start + self.length].reset_index(drop=True)


class CenterCrop(Transform):
    """
    中心裁剪
    
    从序列中心裁剪固定长度的片段
    """
    
    def __init__(self, length: int):
        self.length = length
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        n = len(data)
        
        if n <= self.length:
            return data
        
        start = (n - self.length) // 2
        return data.iloc[start:start + self.length].reset_index(drop=True)


class Compose(Transform):
    """组合多个转换 (本地定义避免循环导入)"""
    
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        for t in self.transforms:
            data = t(data)
        return data
    
    def __repr__(self) -> str:
        lines = [f"  {t}" for t in self.transforms]
        return f"Compose([\n" + ",\n".join(lines) + "\n])"


# 预定义的增强组合
def get_weak_augmentation():
    """弱增强 - 保守的数据增强"""
    return Compose([
        AddGaussianNoise(std=0.01),
        Scaling(scale_range=(0.95, 1.05)),
    ])


def get_strong_augmentation():
    """强增强 - 激进的数据增强"""
    return Compose([
        AddGaussianNoise(std=0.05),
        Scaling(scale_range=(0.8, 1.2)),
        Jitter(sigma=0.05),
        MagnitudeWarp(sigma=0.2),
    ])
