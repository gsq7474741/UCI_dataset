"""
数据集下载器 - 提供统一的数据集下载、校验和解压功能

特性:
- HTTP Range断点续传
- 自动重试机制
- 进度条显示
- SHA1完整性校验
- 支持wget备选下载
"""

import hashlib
import time
from pathlib import Path
from typing import Optional, List

import requests
from tqdm import tqdm

from .config_loader import (
    get_config_manager,
    get_config,
    get_dataset_info,
    list_datasets,
    DatasetInfo,
)
from .extractor import extract_dataset


def _calculate_sha1(file_path: Path, chunk_size: int = 8192) -> str:
    """计算文件的SHA1哈希值"""
    sha1 = hashlib.sha1()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha1.update(chunk)
    return sha1.hexdigest()


def _get_remote_file_size(url: str, timeout: int = 10) -> Optional[int]:
    """获取远程文件大小"""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; UCI-Dataset-Downloader/1.0)"}
    try:
        response = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)
        if response.status_code == 200:
            content_length = response.headers.get("Content-Length")
            if content_length:
                return int(content_length)
    except Exception:
        pass
    return None


def _supports_resume(url: str, timeout: int = 10) -> bool:
    """检查服务器是否支持断点续传"""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; UCI-Dataset-Downloader/1.0)",
        "Range": "bytes=0-0"
    }
    try:
        response = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)
        return response.status_code == 206 or "Accept-Ranges" in response.headers
    except Exception:
        return False


def _download_with_resume(
    url: str,
    dest_path: Path,
    desc: Optional[str] = None,
    chunk_size: int = 65536,
    max_retries: int = 10,
    timeout: int = 30,
) -> None:
    """
    带断点续传的文件下载
    
    特性:
    - HTTP Range断点续传
    - 自动重试
    - 进度条显示
    - 下载进度持久化
    """
    desc = desc or dest_path.name
    temp_path = dest_path.with_suffix(dest_path.suffix + ".downloading")
    headers_base = {"User-Agent": "Mozilla/5.0 (compatible; UCI-Dataset-Downloader/1.0)"}
    
    # 获取文件总大小
    total_size = _get_remote_file_size(url, timeout)
    supports_resume = _supports_resume(url, timeout) if total_size else False
    
    if supports_resume:
        print(f"  服务器支持断点续传 ✓")
    
    downloaded = 0
    if temp_path.exists() and supports_resume:
        downloaded = temp_path.stat().st_size
        if total_size and downloaded >= total_size:
            # 下载已完成，重命名
            temp_path.rename(dest_path)
            return
        print(f"  发现未完成下载，从 {_format_size(downloaded)} 处继续...")
    
    for attempt in range(max_retries):
        try:
            # 每次重试时重新检查已下载大小（关键：修复ChunkedEncodingError断点续传）
            if temp_path.exists() and supports_resume:
                downloaded = temp_path.stat().st_size
            
            headers = headers_base.copy()
            mode = "wb"
            initial = 0
            
            # 断点续传
            if downloaded > 0 and supports_resume:
                headers["Range"] = f"bytes={downloaded}-"
                mode = "ab"
                initial = downloaded
                if attempt > 0:
                    print(f"  从 {_format_size(downloaded)} 处继续下载...")
            
            response = requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=timeout,
            )
            
            # 检查响应
            if response.status_code == 416:  # Range Not Satisfiable
                # 文件可能已完整，尝试验证
                if temp_path.exists():
                    temp_path.rename(dest_path)
                    return
                raise RuntimeError("Range请求失败，文件可能已被修改")
            
            response.raise_for_status()
            
            # 更新总大小
            if not total_size:
                content_length = response.headers.get("Content-Length")
                if content_length:
                    total_size = int(content_length) + initial
            
            with open(temp_path, mode) as f:
                with tqdm(
                    total=total_size,
                    initial=initial,
                    unit="B",
                    unit_scale=True,
                    desc=desc,
                    miniters=1,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            f.flush()  # 及时刷新到磁盘
                            pbar.update(len(chunk))
                            downloaded += len(chunk)
            
            # 下载完成，验证并重命名
            if total_size and temp_path.stat().st_size != total_size:
                raise RuntimeError(f"文件大小不匹配: 期望 {total_size}, 实际 {temp_path.stat().st_size}")
            
            temp_path.rename(dest_path)
            return
            
        except requests.exceptions.HTTPError as e:
            if temp_path.exists():
                downloaded = temp_path.stat().st_size
            raise RuntimeError(f"HTTP Error: {e}") from e
            
        except (requests.exceptions.ConnectionError, 
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
                ConnectionError,
                OSError) as e:
            # 保存当前进度
            if temp_path.exists():
                downloaded = temp_path.stat().st_size
            
            if attempt < max_retries - 1:
                wait_time = min((attempt + 1) * 2, 30)  # 最多等30秒
                print(f"\n  连接中断 ({type(e).__name__})，已下载 {_format_size(downloaded)}")
                print(f"  {wait_time}秒后重试 ({attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(
                    f"下载失败 (已重试{max_retries}次)\n"
                    f"  已下载: {_format_size(downloaded)}\n"
                    f"  临时文件: {temp_path}\n"
                    f"  错误: {e}"
                ) from e
                
        except Exception as e:
            if temp_path.exists():
                downloaded = temp_path.stat().st_size
            
            if attempt < max_retries - 1:
                wait_time = min((attempt + 1) * 2, 30)
                print(f"\n  下载异常: {e}")
                print(f"  已下载 {_format_size(downloaded)}，{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"下载失败: {e}") from e


def _format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def _download_with_wget(url: str, dest_path: Path) -> bool:
    """使用wget下载文件 (备选方案，支持断点续传)"""
    import subprocess
    try:
        result = subprocess.run(
            ["wget", "-c", "--progress=bar:force", "-O", str(dest_path), url],
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _download_with_progress(
    url: str,
    dest_path: Path,
    desc: Optional[str] = None,
    chunk_size: int = 65536,
    max_retries: int = 10,
    timeout: int = 30,
    use_wget: bool = False,
) -> None:
    """
    带进度条和断点续传的文件下载
    
    Args:
        url: 下载链接
        dest_path: 目标路径
        desc: 进度条描述
        chunk_size: 分块大小
        max_retries: 最大重试次数
        timeout: 超时时间
        use_wget: 是否使用wget下载
    """
    desc = desc or dest_path.name
    
    # 如果指定使用wget
    if use_wget:
        print(f"  使用wget下载 (支持断点续传)...")
        if _download_with_wget(url, dest_path):
            return
        print("  wget失败，切换到Python下载...")
    
    # 使用带断点续传的下载
    _download_with_resume(
        url=url,
        dest_path=dest_path,
        desc=desc,
        chunk_size=chunk_size,
        max_retries=max_retries,
        timeout=timeout,
    )


def download(
    dataset_name: str,
    force: bool = False,
    extract: bool = True,
    verify: bool = True,
    use_wget: bool = False,
) -> Path:
    """
    下载单个数据集
    
    Args:
        dataset_name: 数据集名称
        force: 是否强制重新下载
        extract: 是否自动解压
        verify: 是否校验SHA1
        use_wget: 是否优先使用wget下载 (网络不稳定时推荐)
    
    Returns:
        数据集目录路径
    """
    manager = get_config_manager()
    info = manager.get_dataset(dataset_name)
    dataset_dir = manager.get_dataset_dir(dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = dataset_dir / info.file_name
    raw_dir = dataset_dir / "raw"
    
    # 从配置获取下载选项
    config = get_config()
    if use_wget is False:
        use_wget = config.download_options.use_wget
    
    # 检查是否需要下载
    if zip_path.exists() and not force:
        print(f"[{info.name}] 文件已存在: {zip_path.name}")
    else:
        print(f"[{info.name}] 开始下载...")
        _download_with_progress(
            info.url,
            zip_path,
            desc=info.file_name,
            use_wget=use_wget,
            max_retries=config.download_options.max_retries,
            timeout=config.download_options.timeout,
        )
    
    # SHA1校验
    if verify:
        print(f"[{info.name}] 校验SHA1...")
        computed_hash = _calculate_sha1(zip_path)
        if computed_hash != info.sha1_hash:
            raise ValueError(
                f"SHA1校验失败!\n"
                f"  期望: {info.sha1_hash}\n"
                f"  实际: {computed_hash}"
            )
        print(f"[{info.name}] SHA1校验通过 ✓")
    
    # 解压 (使用新的extractor模块)
    if extract:
        extract_dataset(info, zip_path, raw_dir)
    
    return dataset_dir


def download_all(
    force: bool = False,
    extract: bool = True,
    verify: bool = True,
    skip_on_error: bool = True,
) -> List[str]:
    """
    下载所有数据集
    
    Args:
        force: 是否强制重新下载
        extract: 是否自动解压
        verify: 是否校验SHA1
        skip_on_error: 出错时是否跳过继续
    
    Returns:
        成功下载的数据集列表
    """
    datasets = list_datasets()
    successful = []
    failed = []
    
    print(f"准备下载 {len(datasets)} 个数据集...\n")
    
    for name in datasets:
        try:
            download(name, force=force, extract=extract, verify=verify)
            successful.append(name)
            print()
        except Exception as e:
            failed.append((name, str(e)))
            if skip_on_error:
                print(f"[{name}] 下载失败: {e}\n")
            else:
                raise
    
    # 打印汇总
    print("=" * 50)
    print(f"下载完成: {len(successful)}/{len(datasets)}")
    if failed:
        print(f"失败: {len(failed)}")
        for name, err in failed:
            print(f"  - {name}: {err}")
    
    return successful


def verify_dataset(dataset_name: str) -> bool:
    """验证数据集完整性"""
    manager = get_config_manager()
    info = manager.get_dataset(dataset_name)
    dataset_dir = manager.get_dataset_dir(dataset_name)
    zip_path = dataset_dir / info.file_name
    
    if not zip_path.exists():
        print(f"[{info.name}] 文件不存在")
        return False
    
    computed_hash = _calculate_sha1(zip_path)
    if computed_hash != info.sha1_hash:
        print(f"[{info.name}] SHA1不匹配")
        return False
    
    print(f"[{info.name}] 验证通过 ✓")
    return True


def get_status() -> dict:
    """获取所有数据集的下载状态"""
    manager = get_config_manager()
    status = {}
    for name in manager.list_datasets():
        info = manager.get_dataset(name)
        dataset_dir = manager.get_dataset_dir(name)
        zip_path = dataset_dir / info.file_name
        raw_dir = dataset_dir / "raw"
        
        status[name] = {
            "downloaded": zip_path.exists(),
            "extracted": raw_dir.exists() and any(raw_dir.iterdir()) if raw_dir.exists() else False,
            "zip_path": str(zip_path) if zip_path.exists() else None,
            "extract_type": info.extract.type,
        }
    return status


def print_status() -> None:
    """打印数据集下载状态"""
    status = get_status()
    print("数据集下载状态:")
    print("-" * 70)
    for name, info in status.items():
        dl = "✓" if info["downloaded"] else "✗"
        ex = "✓" if info["extracted"] else "✗"
        print(f"  {name:<55} 下载:{dl} 解压:{ex}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python -m enose_uci_dataset.downloader status      # 查看状态")
        print("  python -m enose_uci_dataset.downloader all         # 下载全部")
        print("  python -m enose_uci_dataset.downloader <name>      # 下载指定数据集")
        print("\n可用数据集:")
        for name in list_datasets():
            print(f"  - {name}")
        sys.exit(0)
    
    cmd = sys.argv[1]
    
    if cmd == "status":
        print_status()
    elif cmd == "all":
        download_all()
    else:
        download(cmd)
