"""
数据集自动下载工具（魔搭 ModelScope）。

魔搭数据集仓库：``douyimin/CIG-Bench-Dataset``，目录结构::

    CIG-Bench-Dataset/
    ├── Structure/      # 结构解释 (Structure) 数据
    └── Geobody/        # 地质体识别 (Geobody) 数据

一行代码即可下载到「指定目录」并拿到本地路径::

    from cig_bench.dataset import cig_structureData, cig_geobodyData

    # 默认下载到当前目录下的 ./CIG-Bench-Dataset/<子集>
    structure_dir = cig_structureData()
    geobody_dir   = cig_geobodyData()

    # 强制指定下载目录
    structure_dir = cig_structureData("/data/seis")     # -> /data/seis/Structure
    geobody_dir   = cig_geobodyData(download_dir="./gb") # -> ./gb/Geobody

依赖：
    pip install modelscope
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple


DATASET_DEFAULT_ID = "douyimin/CIG-Bench-Dataset"

# 默认下载根目录（相对当前工作目录，可见、可控，而不是魔搭隐藏缓存）
DEFAULT_DOWNLOAD_ROOT = "./CIG-Bench-Dataset"

# 子集名 -> (dataset_id, 仓库内子目录)
DATASET_REGISTRY: Dict[str, Tuple[str, str]] = {
    "structure": (DATASET_DEFAULT_ID, "Structure"),
    "geobody":   (DATASET_DEFAULT_ID, "Geobody"),
}


def _snapshot_download_subdir(dataset_id: str,
                              subdir: str,
                              download_dir: str,
                              revision: str = "master") -> str:
    """从魔搭数据集仓库下载指定子目录到 ``download_dir``，返回该子目录的本地绝对路径。

    文件会被放在 ``download_dir/<subdir>/...``（保留仓库内的相对路径），
    因此返回的就是 ``download_dir/<subdir>`` 的绝对路径。
    """
    try:
        from modelscope.hub.snapshot_download import dataset_snapshot_download
    except ImportError as e:
        raise ImportError(
            "Auto-download from ModelScope requires the `modelscope` package.\n"
            "    pip install modelscope"
        ) from e

    download_dir = os.path.abspath(os.path.expanduser(download_dir))
    os.makedirs(download_dir, exist_ok=True)

    # local_dir：强制下载到该目录（真实文件，而非缓存软链接）
    dataset_snapshot_download(
        dataset_id=dataset_id,
        allow_patterns=[f"{subdir}/*"],
        revision=revision,
        local_dir=download_dir,
    )

    subset_path = os.path.join(download_dir, subdir)
    return subset_path if os.path.isdir(subset_path) else download_dir


def ensure_dataset(name: str,
                   download_dir: Optional[str] = None,
                   revision: str = "master",
                   dataset_id: Optional[str] = None,
                   subdir: Optional[str] = None) -> str:
    """
    下载指定数据子集到 ``download_dir`` 并返回本地路径。

    Args:
        name: 子集名，'structure' 或 'geobody'（仅在走默认注册表时用到）。
        download_dir: 强制指定的下载目录。None 时使用默认 ``./CIG-Bench-Dataset``。
        revision: 仓库分支/版本，默认 'master'。
        dataset_id: 覆盖默认数据集仓库 ID。
        subdir: 覆盖默认子目录名。

    Returns:
        子集数据所在本地目录的绝对路径（``download_dir/<subdir>``）。
    """
    key = name.lower().strip()

    if dataset_id is not None or subdir is not None:
        if dataset_id is None or subdir is None:
            raise ValueError(
                "`dataset_id` and `subdir` must be provided together "
                "when overriding the default registry."
            )
        did, sub = dataset_id, subdir
    else:
        if key not in DATASET_REGISTRY:
            raise KeyError(
                f"Unknown dataset {name!r}. "
                f"Known datasets: {list(DATASET_REGISTRY)}. "
                f"Either register it in DATASET_REGISTRY "
                f"or pass dataset_id/subdir explicitly."
            )
        did, sub = DATASET_REGISTRY[key]

    root = DEFAULT_DOWNLOAD_ROOT if download_dir is None else download_dir
    return _snapshot_download_subdir(did, sub, root, revision)


def cig_structureData(download_dir: Optional[str] = None,
                      revision: str = "master") -> str:
    """一行下载 Structure（结构解释）数据集到指定目录，返回本地路径。

    Args:
        download_dir: 强制指定下载目录。None -> ``./CIG-Bench-Dataset``。
                      文件落在 ``<download_dir>/Structure``。
        revision: 仓库分支/版本，默认 'master'。

    >>> from cig_bench.dataset import cig_structureData
    >>> path = cig_structureData("/data/seis")   # -> /data/seis/Structure
    """
    return ensure_dataset("structure", download_dir=download_dir, revision=revision)


def cig_geobodyData(download_dir: Optional[str] = None,
                    revision: str = "master") -> str:
    """一行下载 Geobody（地质体识别）数据集到指定目录，返回本地路径。

    Args:
        download_dir: 强制指定下载目录。None -> ``./CIG-Bench-Dataset``。
                      文件落在 ``<download_dir>/Geobody``。
        revision: 仓库分支/版本，默认 'master'。

    >>> from cig_bench.dataset import cig_geobodyData
    >>> path = cig_geobodyData("/data/seis")   # -> /data/seis/Geobody
    """
    return ensure_dataset("geobody", download_dir=download_dir, revision=revision)


__all__ = [
    "cig_structureData",
    "cig_geobodyData",
    "ensure_dataset",
    "DATASET_REGISTRY",
    "DATASET_DEFAULT_ID",
    "DEFAULT_DOWNLOAD_ROOT",
]
