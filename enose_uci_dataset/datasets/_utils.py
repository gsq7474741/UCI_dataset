from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path
from typing import Optional

from ._info import DatasetInfo


def _sha1(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def check_sha1(path: Path, sha1: str) -> bool:
    if not path.is_file():
        return False
    return _sha1(path) == sha1


def download_url(url: str, dst: Path, *, overwrite: bool = False) -> None:
    import urllib.request

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and not overwrite:
        return

    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    try:
        with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out)
        tmp.replace(dst)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def extract_zip(zip_path: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst_dir)


def _post_extract_cleanup(raw_dir: Path, *, subdir: Optional[str]) -> None:
    macosx_dir = raw_dir / "__MACOSX"
    if macosx_dir.exists():
        shutil.rmtree(macosx_dir)

    if not subdir:
        return

    sd = raw_dir / subdir
    if not (sd.exists() and sd.is_dir()):
        return

    for item in sd.iterdir():
        dest = raw_dir / item.name
        if not dest.exists():
            shutil.move(str(item), str(dest))

    if sd.exists():
        shutil.rmtree(sd)


def extract_dataset(
    info: DatasetInfo,
    zip_path: Path,
    raw_dir: Path,
    *,
    force: bool = False,
) -> None:
    if raw_dir.exists() and any(raw_dir.iterdir()) and not force:
        return

    if raw_dir.exists() and force:
        shutil.rmtree(raw_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)

    extract_zip(zip_path, raw_dir)

    if info.extract.type == "nested" and info.extract.nested_zip:
        nested = raw_dir / info.extract.nested_zip
        if nested.exists():
            extract_zip(nested, raw_dir)
            nested.unlink()

    _post_extract_cleanup(raw_dir, subdir=info.extract.subdir)


def download_and_extract(
    info: DatasetInfo,
    dataset_dir: Path,
    *,
    force: bool = False,
    verify: bool = True,
) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)

    archive_path = dataset_dir / info.file_name

    if force and archive_path.exists():
        archive_path.unlink()

    if not archive_path.exists():
        download_url(info.url, archive_path)

    if verify and not check_sha1(archive_path, info.sha1):
        archive_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA1 校验失败: {archive_path}\n"
            f"  expected: {info.sha1}\n"
            f"  got: {_sha1(archive_path) if archive_path.exists() else 'missing'}"
        )

    raw_dir = dataset_dir / "raw"
    extract_dataset(info, archive_path, raw_dir, force=force)
    return dataset_dir
