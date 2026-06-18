"""CIG-Bench 数据集下载接口。

一行代码下载数据集（魔搭 ModelScope: douyimin/CIG-Bench-Dataset）::

    from cig_bench.dataset import cig_structureData
    from cig_bench.dataset import cig_geobodyData

    structure_dir = cig_structureData()          # -> ./CIG-Bench-Dataset/Structure
    geobody_dir   = cig_geobodyData()            # -> ./CIG-Bench-Dataset/Geobody
    structure_dir = cig_structureData("/data")   # 强制指定下载目录 -> /data/Structure
"""

from ._download import (  # noqa: F401
    cig_structureData,
    cig_geobodyData,
    ensure_dataset,
    DATASET_REGISTRY,
    DATASET_DEFAULT_ID,
    DEFAULT_DOWNLOAD_ROOT,
)

__all__ = [
    "cig_structureData",
    "cig_geobodyData",
    "ensure_dataset",
    "DATASET_REGISTRY",
    "DATASET_DEFAULT_ID",
    "DEFAULT_DOWNLOAD_ROOT",
]
