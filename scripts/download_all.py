#!/usr/bin/env python
"""一键下载所有 UCI 电子鼻数据集并运行测试。

使用方法:
    python scripts/download_all.py           # 仅下载
    python scripts/download_all.py --test    # 下载并测试
    python scripts/download_all.py --root /path/to/data  # 指定数据目录
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from enose_uci_dataset.datasets import DATASETS, list_datasets, get_dataset_info


def download_dataset(name: str, cls, root: Path, verbose: bool = True) -> bool:
    """下载单个数据集。"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"📦 下载: {name}")
        info = get_dataset_info(name)
        print(f"   URL: {info.url}")
    
    start_time = time.time()
    try:
        # Try with cache parameter first, fall back to without
        import inspect
        sig = inspect.signature(cls.__init__)
        if 'cache' in sig.parameters:
            ds = cls(str(root), download=True, cache=True)
        else:
            ds = cls(str(root), download=True)
        elapsed = time.time() - start_time
        if verbose:
            print(f"   ✅ 成功! {len(ds)} 样本, 耗时 {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start_time
        if verbose:
            print(f"   ❌ 失败: {e}")
            print(f"   耗时 {elapsed:.1f}s")
        return False


def download_all(root: Path, verbose: bool = True) -> dict:
    """下载所有数据集。"""
    results = {"success": [], "failed": []}
    
    print("=" * 60)
    print("🚀 开始下载所有 UCI 电子鼻数据集")
    print(f"📁 数据目录: {root.resolve()}")
    print(f"📊 数据集数量: {len(DATASETS)}")
    print("=" * 60)
    
    total_start = time.time()
    
    for name, cls in DATASETS.items():
        success = download_dataset(name, cls, root, verbose)
        if success:
            results["success"].append(name)
        else:
            results["failed"].append(name)
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("📊 下载统计")
    print("=" * 60)
    print(f"✅ 成功: {len(results['success'])}/{len(DATASETS)}")
    print(f"❌ 失败: {len(results['failed'])}/{len(DATASETS)}")
    print(f"⏱️  总耗时: {total_elapsed:.1f}s")
    
    if results["failed"]:
        print(f"\n失败的数据集:")
        for name in results["failed"]:
            print(f"  - {name}")
    
    return results


def run_tests(root: Path) -> int:
    """运行测试。"""
    print("\n" + "=" * 60)
    print("🧪 运行测试")
    print("=" * 60)
    
    import os
    env = os.environ.copy()
    env["ENOSE_DATA_ROOT"] = str(root.resolve())
    
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_datasets", "-v"],
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="下载所有 UCI 电子鼻数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / ".cache",
        help="数据存储目录 (默认: ./.cache)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="下载后运行测试",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="仅列出可用数据集",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式",
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("可用数据集:")
        for name in list_datasets():
            info = get_dataset_info(name)
            print(f"  - {name}")
            print(f"    URL: {info.url}")
        return 0
    
    # 确保目录存在
    args.root.mkdir(parents=True, exist_ok=True)
    
    # 下载所有数据集
    results = download_all(args.root, verbose=not args.quiet)
    
    # 运行测试
    if args.test:
        return run_tests(args.root)
    
    return 0 if not results["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
