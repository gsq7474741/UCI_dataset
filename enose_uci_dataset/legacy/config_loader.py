"""
配置加载器 - 从YAML文件加载数据集配置和运行配置
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml


# 配置文件路径
CONFIG_DIR = Path(__file__).parent
DATASETS_YAML = CONFIG_DIR / "datasets.yaml"
CONFIG_YAML = CONFIG_DIR / "config.yaml"


@dataclass
class ExtractConfig:
    """解压配置"""
    type: str = "standard"  # standard | turbo | nested
    subdir: Optional[str] = None
    nested_zip: Optional[str] = None


@dataclass
class TimeSeriesConfig:
    """时序数据配置"""
    continuous: bool = False
    sample_rate_hz: int = 100


@dataclass 
class SensorConfig:
    """传感器配置"""
    type: str = "MOX"
    count: int = 8


@dataclass
class DatasetInfo:
    """数据集信息"""
    name: str
    uci_id: int
    file_name: str
    sha1_hash: str
    url: str
    description: str = ""
    dir_name: Optional[str] = None  # 目录名，如果与name不同
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    tasks: List[str] = field(default_factory=list)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    time_series: Optional[TimeSeriesConfig] = None
    
    @classmethod
    def from_dict(cls, name: str, data: dict) -> "DatasetInfo":
        """从字典创建DatasetInfo"""
        extract_data = data.get("extract", {})
        extract = ExtractConfig(
            type=extract_data.get("type", "standard"),
            subdir=extract_data.get("subdir"),
            nested_zip=extract_data.get("nested_zip"),
        )
        
        sensors_data = data.get("sensors", {})
        sensors = SensorConfig(
            type=sensors_data.get("type", "MOX"),
            count=sensors_data.get("count", 8),
        )
        
        ts_data = data.get("time_series")
        time_series = None
        if ts_data:
            time_series = TimeSeriesConfig(
                continuous=ts_data.get("continuous", False),
                sample_rate_hz=ts_data.get("sample_rate_hz", 100),
            )
        
        return cls(
            name=name,
            uci_id=data["uci_id"],
            file_name=data["file_name"],
            sha1_hash=data["sha1_hash"],
            url=data["url"],
            description=data.get("description", ""),
            dir_name=data.get("dir_name"),
            extract=extract,
            tasks=data.get("tasks", []),
            sensors=sensors,
            time_series=time_series,
        )


@dataclass
class DownloadOptions:
    """下载选项"""
    max_retries: int = 10
    timeout: int = 30
    chunk_size: int = 65536
    use_wget: bool = False


@dataclass
class ExtractOptions:
    """解压选项"""
    enable_turbo: bool = True
    num_workers: int = 0  # 0表示自动
    use_ram_disk: bool = False
    auto_cleanup: bool = True


@dataclass
class SlicingOptions:
    """切片选项"""
    enabled: bool = False
    window_size: int = 100
    stride: int = 50
    auto_segment: bool = True


@dataclass
class Config:
    """主配置"""
    # 下载配置
    download_datasets: List[str] = field(default_factory=list)
    download_exclude: List[str] = field(default_factory=list)
    download_options: DownloadOptions = field(default_factory=DownloadOptions)
    
    # 解压配置
    extract_options: ExtractOptions = field(default_factory=ExtractOptions)
    
    # 处理配置
    output_dir: str = "processed/v1"
    output_formats: List[str] = field(default_factory=lambda: ["csv"])
    
    # 转换配置
    normalize_columns: bool = True
    time_unit: str = "s"
    default_sample_rate: int = 100
    
    # 切片配置
    slicing: SlicingOptions = field(default_factory=SlicingOptions)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """从字典创建Config"""
        download = data.get("download", {})
        download_opts = download.get("options", {})
        
        extract = data.get("extract", {})
        process = data.get("process", {})
        transform = data.get("transform", {})
        slicing_data = data.get("slicing", {})
        
        return cls(
            download_datasets=download.get("datasets", []),
            download_exclude=download.get("exclude", []),
            download_options=DownloadOptions(
                max_retries=download_opts.get("max_retries", 10),
                timeout=download_opts.get("timeout", 30),
                chunk_size=download_opts.get("chunk_size", 65536),
                use_wget=download_opts.get("use_wget", False),
            ),
            extract_options=ExtractOptions(
                enable_turbo=extract.get("enable_turbo", True),
                num_workers=extract.get("num_workers", 0),
                use_ram_disk=extract.get("use_ram_disk", False),
                auto_cleanup=extract.get("auto_cleanup", True),
            ),
            output_dir=process.get("output_dir", "processed/v1"),
            output_formats=process.get("formats", ["csv"]),
            normalize_columns=transform.get("normalize_columns", True),
            time_unit=transform.get("time_unit", "s"),
            default_sample_rate=transform.get("default_sample_rate", 100),
            slicing=SlicingOptions(
                enabled=slicing_data.get("enabled", False),
                window_size=slicing_data.get("window_size", 100),
                stride=slicing_data.get("stride", 50),
                auto_segment=slicing_data.get("auto_segment", True),
            ),
        )


class ConfigManager:
    """配置管理器 - 单例模式"""
    
    _instance = None
    _datasets: Dict[str, DatasetInfo] = {}
    _config: Optional[Config] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_datasets()
            cls._instance._load_config()
        return cls._instance
    
    def _load_datasets(self) -> None:
        """加载数据集配置"""
        if DATASETS_YAML.exists():
            with open(DATASETS_YAML, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            
            datasets = data.get("datasets", {})
            for name, info in datasets.items():
                self._datasets[name] = DatasetInfo.from_dict(name, info)
    
    def _load_config(self) -> None:
        """加载运行配置"""
        if CONFIG_YAML.exists():
            with open(CONFIG_YAML, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self._config = Config.from_dict(data)
        else:
            self._config = Config()
    
    def reload(self) -> None:
        """重新加载配置"""
        self._datasets.clear()
        self._load_datasets()
        self._load_config()
    
    @property
    def datasets(self) -> Dict[str, DatasetInfo]:
        """获取所有数据集信息"""
        return self._datasets
    
    @property
    def config(self) -> Config:
        """获取运行配置"""
        return self._config
    
    def get_dataset(self, name: str) -> DatasetInfo:
        """获取指定数据集信息"""
        normalized = name.replace("-", "_")
        if normalized not in self._datasets:
            available = ", ".join(self._datasets.keys())
            raise ValueError(f"Unknown dataset: {name}. Available: {available}")
        return self._datasets[normalized]
    
    def list_datasets(self) -> List[str]:
        """列出所有数据集名称"""
        return list(self._datasets.keys())
    
    def get_datasets_to_download(self) -> List[str]:
        """获取需要下载的数据集列表"""
        if self._config.download_datasets:
            # 指定了具体数据集
            return [d for d in self._config.download_datasets 
                    if d in self._datasets and d not in self._config.download_exclude]
        else:
            # 全部数据集（排除exclude列表）
            return [d for d in self._datasets.keys() 
                    if d not in self._config.download_exclude]
    
    def get_dataset_dir(self, name: str) -> Path:
        """获取数据集目录路径"""
        info = self.get_dataset(name)
        base_dir = CONFIG_DIR
        
        # 使用配置中的dir_name或自动检测
        if info.dir_name:
            return base_dir / info.dir_name
        
        # 尝试不同的目录名格式
        candidates = [
            base_dir / info.name,
            base_dir / info.name.replace("_", "-"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        
        return base_dir / info.name


# 全局配置管理器实例
_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """获取配置管理器实例"""
    global _manager
    if _manager is None:
        _manager = ConfigManager()
    return _manager


def reload_config() -> None:
    """重新加载配置"""
    get_config_manager().reload()


# 便捷函数
def get_dataset_info(name: str) -> DatasetInfo:
    """获取数据集信息"""
    return get_config_manager().get_dataset(name)


def list_datasets() -> List[str]:
    """列出所有数据集"""
    return get_config_manager().list_datasets()


def get_config() -> Config:
    """获取运行配置"""
    return get_config_manager().config


def get_dataset_dir(name: str) -> Path:
    """获取数据集目录"""
    return get_config_manager().get_dataset_dir(name)


if __name__ == "__main__":
    # 测试配置加载
    manager = get_config_manager()
    
    print("=== 数据集列表 ===")
    for name in manager.list_datasets():
        info = manager.get_dataset(name)
        print(f"  {name}: UCI-{info.uci_id}, {info.extract.type} extract")
    
    print("\n=== 运行配置 ===")
    config = manager.config
    print(f"  下载重试次数: {config.download_options.max_retries}")
    print(f"  启用Turbo解压: {config.extract_options.enable_turbo}")
    print(f"  输出目录: {config.output_dir}")
