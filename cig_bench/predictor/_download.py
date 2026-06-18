"""
权重自动下载工具（仅支持魔搭 ModelScope）。

提供给各个 Predictor 调用：
    weight_path = ensure_weight('fault')
    # 找不到本地缓存就从魔搭仓库下载 fault 任务的 .pth 文件

每个任务对应的 (model_id, file_path) 集中在 ``MODELSCOPE_REGISTRY`` 中维护。

依赖：
    pip install modelscope
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple


MODELSCOPE_DEFAULT_MODEL_ID = "douyimin/CIG-Bench"

# 已上传到魔搭的权重(根目录), 文件名以 CIG-Bench-<Task>.pth 命名
MODELSCOPE_REGISTRY: Dict[str, Tuple[str, str]] = {
    # task name -> (model_id, file_path_in_repo)
    "channel":  (MODELSCOPE_DEFAULT_MODEL_ID, "CIG-Bench-Channel.pth"),
    "fault":    (MODELSCOPE_DEFAULT_MODEL_ID, "CIG-Bench-Fault.pth"),
    "karst":    (MODELSCOPE_DEFAULT_MODEL_ID, "CIG-Bench-Karst.pth"),
    "property": (MODELSCOPE_DEFAULT_MODEL_ID, "CIG-Bench-Property.pth"),
    "rgt":      (MODELSCOPE_DEFAULT_MODEL_ID, "CIG-Bench-RGT.pth"),
}

# 向后兼容：保留旧名字 WEIGHT_REGISTRY / DEFAULT_MODEL_ID
WEIGHT_REGISTRY = MODELSCOPE_REGISTRY
DEFAULT_MODEL_ID = MODELSCOPE_DEFAULT_MODEL_ID


# ---------------------------------------------------------------------------
# 下载实现
# ---------------------------------------------------------------------------
def _download_from_modelscope(model_id: str,
                              file_path: str,
                              cache_dir: Optional[str] = None,
                              revision: Optional[str] = None) -> str:
    """调用 modelscope.hub.file_download.model_file_download 拉取单文件。"""
    try:
        from modelscope.hub.file_download import model_file_download
    except ImportError as e:
        raise ImportError(
            "Auto-download from ModelScope requires the `modelscope` package.\n"
            "    pip install modelscope\n"
            "Or pass `restore_path=...` to the predictor manually."
        ) from e

    kwargs = dict(model_id=model_id, file_path=file_path)
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if revision is not None:
        kwargs["revision"] = revision

    return model_file_download(**kwargs)


def ensure_weight(task: str,
                  restore_path: Optional[str] = None,
                  model_id: Optional[str] = None,
                  file_path: Optional[str] = None,
                  cache_dir: Optional[str] = None,
                  revision: Optional[str] = None) -> str:
    """
    返回最终用于加载的权重本地路径。优先级：

      1. ``restore_path`` 不为 None 且本地存在 → 直接使用，不下载。
      2. 通过 ``model_id`` + ``file_path`` 显式覆盖 → 从魔搭下载。
      3. 否则按 ``task`` 在 ``MODELSCOPE_REGISTRY`` 中查 → 从魔搭下载。

    Args:
        task: 任务名(channel/fault/karst/property/rgt)，仅在走默认注册表分支时用到。
        restore_path: 用户显式给定的本地路径；存在时直接返回。
        model_id: 覆盖默认 model_id。
        file_path: 覆盖默认 file_path。
        cache_dir: 缓存目录(None 用平台默认)。
        revision: 仓库版本 / 分支(None 用主分支)。

    Returns:
        本地权重文件绝对路径。
    """
    # 1) 用户显式给本地路径
    if restore_path is not None and os.path.exists(restore_path):
        return restore_path

    # 2) 用户显式覆盖远端坐标
    if model_id is not None or file_path is not None:
        if model_id is None or file_path is None:
            raise ValueError(
                "`model_id` and `file_path` must be provided together "
                "when overriding the default registry."
            )
        return _download_from_modelscope(model_id, file_path, cache_dir, revision)

    # 3) 走默认注册表
    if task not in MODELSCOPE_REGISTRY:
        raise KeyError(
            f"Unknown task {task!r}. "
            f"Known tasks: {list(MODELSCOPE_REGISTRY)}. "
            f"Either register it in MODELSCOPE_REGISTRY "
            f"or pass model_id/file_path explicitly."
        )
    mid, fp = MODELSCOPE_REGISTRY[task]
    return _download_from_modelscope(mid, fp, cache_dir, revision)


__all__ = [
    "ensure_weight",
    "MODELSCOPE_REGISTRY",
    "MODELSCOPE_DEFAULT_MODEL_ID",
    # 向后兼容
    "WEIGHT_REGISTRY", "DEFAULT_MODEL_ID",
]
