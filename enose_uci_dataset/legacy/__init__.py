# Package marker for enose_uci_dataset
"""
UCI电子鼻数据集管理包

用法:
    from enose_uci_dataset import download, list_datasets, print_status
    
    # 查看可用数据集
    list_datasets()
    
    # 下载数据集
    download("twin_gas_sensor_arrays")
    
    # 命令行使用
    python -m enose_uci_dataset status
    python -m enose_uci_dataset download --all
"""

from .config_loader import (
    DatasetInfo,
    Config,
    get_config,
    get_dataset_info,
    list_datasets,
    get_dataset_dir,
    get_config_manager,
    reload_config,
)

from .downloader import (
    download,
    download_all,
    verify_dataset,
    get_status,
    print_status,
)

from .extractor import (
    extract_dataset,
)

from .dataset import (
    BaseDataset,
    ENoseDataset,
    DataLoader,
    load_dataset,
)

from .transforms import (
    Transform,
    Compose,
    NormalizeSensorColumns,
    ConvertTimeUnit,
    Resample,
    SlidingWindow,
    SegmentByLabel,
    Normalize,
    FillNaN,
    get_default_transforms,
)

__all__ = [
    # 配置
    "DatasetInfo",
    "Config",
    "get_config",
    "get_dataset_info",
    "list_datasets",
    "get_dataset_dir",
    "get_config_manager",
    "reload_config",
    # 下载
    "download",
    "download_all",
    "verify_dataset",
    "get_status",
    "print_status",
    # 解压
    "extract_dataset",
    # 数据集
    "BaseDataset",
    "ENoseDataset",
    "DataLoader",
    "load_dataset",
    # 转换
    "Transform",
    "Compose",
    "NormalizeSensorColumns",
    "ConvertTimeUnit",
    "Resample",
    "SlidingWindow",
    "SegmentByLabel",
    "Normalize",
    "FillNaN",
    "get_default_transforms",
]
