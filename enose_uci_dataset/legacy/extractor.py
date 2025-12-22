"""
解压模块 - 支持标准解压、并行解压(ZipTurbo)、嵌套解压
"""

import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Optional

from .config_loader import DatasetInfo, get_config


def extract_dataset(
    info: DatasetInfo,
    zip_path: Path,
    raw_dir: Path,
    force: bool = False,
) -> None:
    """
    解压数据集，根据配置选择解压方式
    
    Args:
        info: 数据集信息
        zip_path: zip文件路径
        raw_dir: 解压目标目录
        force: 是否强制重新解压
    """
    if raw_dir.exists() and any(raw_dir.iterdir()) and not force:
        print(f"[{info.name}] raw目录已存在，跳过解压")
        return
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    config = get_config()
    extract_type = info.extract.type
    
    # 根据类型选择解压方式
    if extract_type == "turbo" and config.extract_options.enable_turbo:
        print(f"[{info.name}] 使用并行解压 (ZipTurbo)...")
        _extract_turbo(zip_path, raw_dir, config.extract_options.num_workers)
    elif extract_type == "nested":
        print(f"[{info.name}] 解压嵌套zip...")
        _extract_nested(info, zip_path, raw_dir)
    else:
        print(f"[{info.name}] 标准解压...")
        _extract_standard(zip_path, raw_dir)
    
    # 后处理：移动子目录内容
    _post_extract_cleanup(info, raw_dir)
    
    print(f"[{info.name}] 解压完成 ✓")


def _extract_standard(zip_path: Path, raw_dir: Path) -> None:
    """标准解压"""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)


def _extract_turbo(
    zip_path: Path,
    raw_dir: Path,
    num_workers: int = 0,
) -> None:
    """
    并行解压 (ZipTurbo) - 适用于文件数量多的数据集
    
    使用GNU parallel进行并行解压，显著加速大量小文件的解压过程
    """
    # 检查是否有parallel命令
    if not shutil.which("parallel"):
        print("  警告: 未安装GNU parallel，回退到标准解压")
        _extract_standard(zip_path, raw_dir)
        return
    
    if num_workers <= 0:
        num_workers = os.cpu_count() or 4
    
    try:
        # 1. 先创建目录结构
        print(f"  准备目录结构...")
        result = subprocess.run(
            f'unzip -Z1 "{zip_path}" | grep "/$"',
            shell=True,
            capture_output=True,
            text=True,
        )
        for dir_line in result.stdout.strip().split("\n"):
            if dir_line:
                (raw_dir / dir_line).mkdir(parents=True, exist_ok=True)
        
        # 2. 获取文件总数
        result = subprocess.run(
            ["unzip", "-Z1", str(zip_path)],
            capture_output=True,
            text=True,
        )
        total_files = len(result.stdout.strip().split("\n"))
        print(f"  总文件数: {total_files}, 使用 {num_workers} 线程并行解压...")
        
        # 3. 并行解压
        cmd = f'unzip -Z1 "{zip_path}" | parallel -j {num_workers} --bar unzip -qq -o "{zip_path}" -d "{raw_dir}" {{}}'
        subprocess.run(cmd, shell=True, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"  并行解压失败: {e}，回退到标准解压")
        _extract_standard(zip_path, raw_dir)
    except Exception as e:
        print(f"  解压异常: {e}，回退到标准解压")
        _extract_standard(zip_path, raw_dir)


def _extract_nested(info: DatasetInfo, zip_path: Path, raw_dir: Path) -> None:
    """解压嵌套的zip文件"""
    # 先解压外层
    _extract_standard(zip_path, raw_dir)
    
    # 解压内层zip
    if info.extract.nested_zip:
        nested_zip = raw_dir / info.extract.nested_zip
        if nested_zip.exists():
            print(f"  解压内层: {info.extract.nested_zip}")
            with zipfile.ZipFile(nested_zip, "r") as zf:
                zf.extractall(raw_dir)
            nested_zip.unlink()


def _post_extract_cleanup(info: DatasetInfo, raw_dir: Path) -> None:
    """解压后的目录整理"""
    # 删除 __MACOSX 目录
    macosx_dir = raw_dir / "__MACOSX"
    if macosx_dir.exists():
        shutil.rmtree(macosx_dir)
    
    # 移动子目录内容
    if info.extract.subdir:
        subdir = raw_dir / info.extract.subdir
        if subdir.exists() and subdir.is_dir():
            print(f"  整理目录: 移动 {info.extract.subdir}/ 内容到 raw/")
            for item in subdir.iterdir():
                dest = raw_dir / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
            # 删除空的子目录
            if subdir.exists():
                shutil.rmtree(subdir)


def get_zip_info(zip_path: Path) -> dict:
    """获取zip文件信息"""
    with zipfile.ZipFile(zip_path, "r") as zf:
        file_count = len(zf.namelist())
        total_size = sum(info.file_size for info in zf.infolist())
        compressed_size = sum(info.compress_size for info in zf.infolist())
        
    return {
        "file_count": file_count,
        "total_size": total_size,
        "compressed_size": compressed_size,
        "compression_ratio": compressed_size / total_size if total_size > 0 else 0,
    }
