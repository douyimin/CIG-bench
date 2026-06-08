"""
统一的权重自动下载工具，支持两个源：

  - 魔搭 ModelScope  (默认, source="modelscope")
  - Hugging Face Hub (source="huggingface")

提供给各个 Predictor 调用：
    weight_path = ensure_weight('fault')                       # 默认魔搭
    weight_path = ensure_weight('fault', source='huggingface') # 改用 HF
    # 等价于：找不到本地缓存就从对应仓库下载 fault 任务的 .pth 文件

每个任务在两个平台上的 (model_id, file_path) 分别集中在
``MODELSCOPE_REGISTRY`` 和 ``HF_REGISTRY`` 中维护。

也可以通过环境变量 ``CIG_BENCH_WEIGHT_SOURCE`` 全局切换默认源：
    export CIG_BENCH_WEIGHT_SOURCE=huggingface

依赖：
    pip install modelscope        # 使用魔搭源时需要
    pip install huggingface_hub   # 使用 Hugging Face 源时需要
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# 仓库注册表
# ---------------------------------------------------------------------------
# 每个 task 对应:
#   model_id : 仓库 ID(形如 "<group>/<repo>")
#   file_path: 该仓库下的相对文件路径(必须是 .pth, 即 state_dict)
#
# ⚠️  下面的占位符请根据你在魔搭 / Hugging Face 实际创建的仓库进行修改：
#     - MODELSCOPE_DEFAULT_MODEL_ID → 你的魔搭仓库 ID (例如 "JintaoLee/CIG_Bench")
#     - HF_DEFAULT_MODEL_ID         → 你的 Hugging Face 仓库 ID (例如 "JintaoLee/CIG_Bench")
#     如果不同任务存放在不同仓库，直接修改下面对应行的元组即可。
#
#     注意所有权重统一使用 .pth (load_state_dict 加载方式)，不再使用 jit 的 .pt。
# ---------------------------------------------------------------------------

MODELSCOPE_DEFAULT_MODEL_ID = "douyimin/CIG-Bench"
HF_DEFAULT_MODEL_ID = "douyimin/CIG-Bench"

# 已上传到魔搭的权重(根目录), 文件名以 CIG-Bench-<Task>.pth 命名
MODELSCOPE_REGISTRY: Dict[str, Tuple[str, str]] = {
    # task name -> (model_id, file_path_in_repo)
    "channel":  (MODELSCOPE_DEFAULT_MODEL_ID, "CIG-Bench-Channel.pth"),
    "fault":    (MODELSCOPE_DEFAULT_MODEL_ID, "CIG-Bench-Fault.pth"),
    "karst":    (MODELSCOPE_DEFAULT_MODEL_ID, "CIG-Bench-Karst.pth"),
    "property": (MODELSCOPE_DEFAULT_MODEL_ID, "CIG-Bench-Property.pth"),
    "rgt":      (MODELSCOPE_DEFAULT_MODEL_ID, "CIG-Bench-RGT.pth"),
}

# Hugging Face 仓库尚未创建，先填上和魔搭一致的文件名占位，
# 等你在 HF 上传后，如果文件名/仓库名不一样，再改这里即可。
HF_REGISTRY: Dict[str, Tuple[str, str]] = {
    "channel":  (HF_DEFAULT_MODEL_ID, "CIG-Bench-Channel.pth"),
    "fault":    (HF_DEFAULT_MODEL_ID, "CIG-Bench-Fault.pth"),
    "karst":    (HF_DEFAULT_MODEL_ID, "CIG-Bench-Karst.pth"),
    "property": (HF_DEFAULT_MODEL_ID, "CIG-Bench-Property.pth"),
    "rgt":      (HF_DEFAULT_MODEL_ID, "CIG-Bench-RGT.pth"),
}

# 向后兼容：保留旧名字 WEIGHT_REGISTRY 指向魔搭注册表
WEIGHT_REGISTRY = MODELSCOPE_REGISTRY
DEFAULT_MODEL_ID = MODELSCOPE_DEFAULT_MODEL_ID

# 受支持的 source 字符串
_SUPPORTED_SOURCES = ("modelscope", "huggingface")

# 环境变量：全局切换默认源
_ENV_SOURCE_KEY = "CIG_BENCH_WEIGHT_SOURCE"


def _resolve_source(source: Optional[str]) -> str:
    """根据显式入参 / 环境变量 / 默认值返回最终 source。"""
    if source is None:
        source = os.environ.get(_ENV_SOURCE_KEY, "modelscope")
    source = source.lower().strip()
    # 一些常见别名
    aliases = {
        "ms": "modelscope",
        "model_scope": "modelscope",
        "hf": "huggingface",
        "hugging_face": "huggingface",
        "huggingface_hub": "huggingface",
    }
    source = aliases.get(source, source)
    if source not in _SUPPORTED_SOURCES:
        raise ValueError(
            f"Unknown source {source!r}. Must be one of {_SUPPORTED_SOURCES}."
        )
    return source


# ---------------------------------------------------------------------------
# 下载实现
# ---------------------------------------------------------------------------
def _download_from_modelscope(model_id: str,
                              file_path: str,
                              cache_dir: Optional[str] = None,
                              revision: Optional[str] = None) -> str:
    """
    调用 modelscope.hub.file_download.model_file_download 拉取单文件。
    """
    try:
        from modelscope.hub.file_download import model_file_download
    except ImportError as e:
        raise ImportError(
            "Auto-download from ModelScope requires the `modelscope` package.\n"
            "    pip install modelscope\n"
            "Or switch to Hugging Face (source='huggingface'), or pass "
            "`restore_path=...` to the predictor manually."
        ) from e

    kwargs = dict(model_id=model_id, file_path=file_path)
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if revision is not None:
        kwargs["revision"] = revision

    return model_file_download(**kwargs)


def _download_from_huggingface(model_id: str,
                               file_path: str,
                               cache_dir: Optional[str] = None,
                               revision: Optional[str] = None) -> str:
    """
    调用 huggingface_hub.hf_hub_download 拉取单文件。
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "Auto-download from Hugging Face requires the `huggingface_hub` package.\n"
            "    pip install huggingface_hub\n"
            "Or switch to ModelScope (source='modelscope'), or pass "
            "`restore_path=...` to the predictor manually."
        ) from e

    kwargs = dict(repo_id=model_id, filename=file_path)
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if revision is not None:
        kwargs["revision"] = revision

    return hf_hub_download(**kwargs)


def _download(source: str,
              model_id: str,
              file_path: str,
              cache_dir: Optional[str],
              revision: Optional[str]) -> str:
    """根据 source 派发到对应平台的下载函数。"""
    if source == "modelscope":
        return _download_from_modelscope(model_id, file_path, cache_dir, revision)
    elif source == "huggingface":
        return _download_from_huggingface(model_id, file_path, cache_dir, revision)
    else:
        # _resolve_source 已经校验过，这里只是保险
        raise ValueError(f"Unknown source {source!r}.")


def ensure_weight(task: str,
                  restore_path: Optional[str] = None,
                  model_id: Optional[str] = None,
                  file_path: Optional[str] = None,
                  cache_dir: Optional[str] = None,
                  revision: Optional[str] = None,
                  source: Optional[str] = None) -> str:
    """
    返回最终用于加载的权重本地路径。优先级：

      1. ``restore_path`` 不为 None 且本地存在 → 直接使用，不下载。
      2. 通过 ``model_id`` + ``file_path`` 显式覆盖 → 从 ``source`` 平台下载。
      3. 否则按 ``task`` 在 ``source`` 对应注册表中查 → 从 ``source`` 平台下载。

    Args:
        task: 任务名(channel/fault/karst/property/rgt)，仅在走默认注册表分支时用到。
        restore_path: 用户显式给定的本地路径；存在时直接返回。
        model_id: 覆盖默认 model_id。
        file_path: 覆盖默认 file_path。
        cache_dir: 平台对应的缓存目录(None 用平台默认)。
        revision: 仓库版本 / 分支(None 用主分支)。
        source: 'modelscope' (默认) 或 'huggingface'。
                None 时读取环境变量 ``CIG_BENCH_WEIGHT_SOURCE``，再 fallback 到
                'modelscope'。

    Returns:
        本地权重文件绝对路径。
    """
    # 1) 用户显式给本地路径
    if restore_path is not None and os.path.exists(restore_path):
        return restore_path

    src = _resolve_source(source)
    registry = MODELSCOPE_REGISTRY if src == "modelscope" else HF_REGISTRY

    # 2) 用户显式覆盖远端坐标
    if model_id is not None or file_path is not None:
        if model_id is None or file_path is None:
            raise ValueError(
                "`model_id` and `file_path` must be provided together "
                "when overriding the default registry."
            )
        return _download(src, model_id, file_path, cache_dir, revision)

    # 3) 走默认注册表
    if task not in registry:
        raise KeyError(
            f"Unknown task {task!r} for source {src!r}. "
            f"Known tasks: {list(registry)}. "
            f"Either register it in the corresponding registry "
            f"or pass model_id/file_path explicitly."
        )
    mid, fp = registry[task]
    return _download(src, mid, fp, cache_dir, revision)


__all__ = [
    "ensure_weight",
    "MODELSCOPE_REGISTRY", "HF_REGISTRY",
    "MODELSCOPE_DEFAULT_MODEL_ID", "HF_DEFAULT_MODEL_ID",
    # 向后兼容
    "WEIGHT_REGISTRY", "DEFAULT_MODEL_ID",
]
