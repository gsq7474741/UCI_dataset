"""
命令行主入口 - 可配置的数据集管理工具
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .config_loader import (
    get_config_manager,
    get_config,
    get_dataset_info,
    list_datasets,
    reload_config,
)


def cmd_status(args) -> None:
    """显示数据集状态"""
    manager = get_config_manager()
    
    print("数据集状态:")
    print("-" * 80)
    print(f"{'名称':<50} {'下载':^6} {'解压':^6} {'类型':^10}")
    print("-" * 80)
    
    for name in manager.list_datasets():
        info = manager.get_dataset(name)
        dataset_dir = manager.get_dataset_dir(name)
        zip_path = dataset_dir / info.file_name
        raw_dir = dataset_dir / "raw"
        
        downloaded = "✓" if zip_path.exists() else "✗"
        extracted = "✓" if (raw_dir.exists() and any(raw_dir.iterdir())) else "✗"
        
        print(f"  {name:<48} {downloaded:^6} {extracted:^6} {info.extract.type:^10}")


def cmd_download(args) -> None:
    """下载数据集"""
    from .downloader import download, download_all
    
    if args.all:
        download_all(
            force=args.force,
            extract=not args.no_extract,
            verify=not args.no_verify,
        )
    elif args.datasets:
        for name in args.datasets:
            try:
                download(
                    name,
                    force=args.force,
                    extract=not args.no_extract,
                    verify=not args.no_verify,
                    use_wget=args.wget,
                )
                print()
            except Exception as e:
                print(f"[{name}] 下载失败: {e}")
                if not args.skip_errors:
                    sys.exit(1)
    else:
        # 使用配置文件中的数据集列表
        manager = get_config_manager()
        datasets = manager.get_datasets_to_download()
        
        if not datasets:
            print("没有需要下载的数据集。请在config.yaml中配置或使用--all参数。")
            return
        
        print(f"根据配置下载 {len(datasets)} 个数据集...")
        for name in datasets:
            try:
                download(
                    name,
                    force=args.force,
                    extract=not args.no_extract,
                    verify=not args.no_verify,
                    use_wget=args.wget,
                )
                print()
            except Exception as e:
                print(f"[{name}] 下载失败: {e}")
                if not args.skip_errors:
                    sys.exit(1)


def cmd_extract(args) -> None:
    """解压数据集"""
    from .extractor import extract_dataset
    
    manager = get_config_manager()
    
    datasets = args.datasets if args.datasets else manager.list_datasets()
    
    for name in datasets:
        try:
            info = manager.get_dataset(name)
            dataset_dir = manager.get_dataset_dir(name)
            zip_path = dataset_dir / info.file_name
            raw_dir = dataset_dir / "raw"
            
            if not zip_path.exists():
                print(f"[{name}] zip文件不存在，跳过")
                continue
            
            extract_dataset(info, zip_path, raw_dir, force=args.force)
        except Exception as e:
            print(f"[{name}] 解压失败: {e}")
            if not args.skip_errors:
                sys.exit(1)


def cmd_list(args) -> None:
    """列出数据集"""
    manager = get_config_manager()
    
    if args.verbose:
        print("可用数据集:")
        print("-" * 80)
        for name in manager.list_datasets():
            info = manager.get_dataset(name)
            print(f"\n{name}:")
            print(f"  UCI ID: {info.uci_id}")
            print(f"  描述: {info.description}")
            print(f"  解压类型: {info.extract.type}")
            print(f"  任务: {', '.join(info.tasks)}")
            print(f"  传感器: {info.sensors.type} x {info.sensors.count}")
            if info.time_series:
                print(f"  时序: 连续采样 @ {info.time_series.sample_rate_hz}Hz")
    else:
        print("可用数据集:")
        for name in manager.list_datasets():
            info = manager.get_dataset(name)
            print(f"  - {name} (UCI-{info.uci_id})")


def cmd_info(args) -> None:
    """显示数据集详细信息"""
    manager = get_config_manager()
    
    try:
        info = manager.get_dataset(args.dataset)
        dataset_dir = manager.get_dataset_dir(args.dataset)
        zip_path = dataset_dir / info.file_name
        
        print(f"数据集: {info.name}")
        print(f"  UCI ID: {info.uci_id}")
        print(f"  描述: {info.description}")
        print(f"  URL: {info.url}")
        print(f"  SHA1: {info.sha1_hash}")
        print(f"  文件: {info.file_name}")
        print(f"  目录: {dataset_dir}")
        print(f"  解压类型: {info.extract.type}")
        if info.extract.subdir:
            print(f"  解压子目录: {info.extract.subdir}")
        print(f"  任务: {', '.join(info.tasks) if info.tasks else '未指定'}")
        print(f"  传感器: {info.sensors.type} x {info.sensors.count}")
        if info.time_series:
            print(f"  时序配置:")
            print(f"    连续采样: {info.time_series.continuous}")
            print(f"    采样率: {info.time_series.sample_rate_hz} Hz")
        
        # 文件状态
        print(f"\n状态:")
        print(f"  已下载: {'是' if zip_path.exists() else '否'}")
        if zip_path.exists():
            size_mb = zip_path.stat().st_size / 1024 / 1024
            print(f"  文件大小: {size_mb:.1f} MB")
        
        raw_dir = dataset_dir / "raw"
        print(f"  已解压: {'是' if (raw_dir.exists() and any(raw_dir.iterdir())) else '否'}")
        
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)


def cmd_config(args) -> None:
    """显示当前配置"""
    config = get_config()
    
    print("当前配置:")
    print("\n[下载]")
    print(f"  数据集: {config.download_datasets or '全部'}")
    print(f"  排除: {config.download_exclude or '无'}")
    print(f"  重试次数: {config.download_options.max_retries}")
    print(f"  超时: {config.download_options.timeout}s")
    print(f"  使用wget: {config.download_options.use_wget}")
    
    print("\n[解压]")
    print(f"  启用Turbo: {config.extract_options.enable_turbo}")
    print(f"  并行数: {config.extract_options.num_workers or '自动'}")
    print(f"  使用RAM磁盘: {config.extract_options.use_ram_disk}")
    
    print("\n[处理]")
    print(f"  输出目录: {config.output_dir}")
    print(f"  输出格式: {config.output_formats}")
    
    print("\n[切片]")
    print(f"  启用: {config.slicing.enabled}")
    print(f"  窗口大小: {config.slicing.window_size}")
    print(f"  步长: {config.slicing.stride}")


def main(argv: Optional[List[str]] = None) -> None:
    """主入口"""
    parser = argparse.ArgumentParser(
        prog="enose-dataset",
        description="UCI电子鼻数据集管理工具",
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # status命令
    p_status = subparsers.add_parser("status", help="显示数据集状态")
    p_status.set_defaults(func=cmd_status)
    
    # download命令
    p_download = subparsers.add_parser("download", help="下载数据集")
    p_download.add_argument("datasets", nargs="*", help="要下载的数据集名称")
    p_download.add_argument("--all", "-a", action="store_true", help="下载全部数据集")
    p_download.add_argument("--force", "-f", action="store_true", help="强制重新下载")
    p_download.add_argument("--no-extract", action="store_true", help="下载后不解压")
    p_download.add_argument("--no-verify", action="store_true", help="跳过SHA1校验")
    p_download.add_argument("--wget", "-w", action="store_true", help="使用wget下载")
    p_download.add_argument("--skip-errors", action="store_true", help="出错时继续")
    p_download.set_defaults(func=cmd_download)
    
    # extract命令
    p_extract = subparsers.add_parser("extract", help="解压数据集")
    p_extract.add_argument("datasets", nargs="*", help="要解压的数据集名称")
    p_extract.add_argument("--force", "-f", action="store_true", help="强制重新解压")
    p_extract.add_argument("--skip-errors", action="store_true", help="出错时继续")
    p_extract.set_defaults(func=cmd_extract)
    
    # list命令
    p_list = subparsers.add_parser("list", help="列出可用数据集")
    p_list.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")
    p_list.set_defaults(func=cmd_list)
    
    # info命令
    p_info = subparsers.add_parser("info", help="显示数据集详细信息")
    p_info.add_argument("dataset", help="数据集名称")
    p_info.set_defaults(func=cmd_info)
    
    # config命令
    p_config = subparsers.add_parser("config", help="显示当前配置")
    p_config.set_defaults(func=cmd_config)
    
    args = parser.parse_args(argv)
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    args.func(args)


if __name__ == "__main__":
    main()
