import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import contextlib
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cigvis
from cigvis import colormap

from ..networks.hrnet import HRNet
from .utils import z_score_clip, remove_small_instances
from ._download import ensure_weight

# 单次 scale 可以是标量（同尺度），也可以是 (st, sh, sw) 三元组（各向异性）
ScaleSpec = Union[float, Tuple[float, float, float]]


# ---------------------------------------------------------------------------
# 推理类
# ---------------------------------------------------------------------------
class KarstPredictor:
    """
    把 HRNet 河道（karst）预测的完整流程封装为一个类。

    与 FaultPredictor 的差异：
      - 推理走 **多尺度集成**：对 scales 列表里每个尺度跑一次推理，结果
        插值回原尺寸后累加。最终阈值化用的是累加值。
      - 后处理可选 **小连通体过滤**（remove_small_instances）。
      - 可视化包含 **isosurface bodies** 面板（cigvis.create_bodys）。

    权重加载方式：
      - 统一使用 state_dict (.pth)：先实例化 HRNet()，再用
        ``model.load_state_dict(torch.load(restore_path, map_location='cpu'))`` 加载。
      - ``restore_path`` 不传时，会自动从魔搭(ModelScope) 下载默认权重。

    典型用法
    --------
    >>> # 1) 自动从魔搭下载默认权重
    >>> predictor = karstPredictor(device='cuda')
    >>> # 2) 手动指定本地权重
    >>> predictor = karstPredictor('model_ema_step_47500.pth', device='cuda')
    >>> sum_probs, seis_used = predictor.predict(seis)
    >>> mask = predictor.postprocess(sum_probs, threshold=0.75, min_size=50000)
    >>> predictor.visualize(seis_used, sum_probs, mask)
    """

    DEFAULT_SCALES: List[float] = [1.0]
    TASK_NAME: str = "karst"

    # ------------------------- 初始化 ------------------------- #
    def __init__(self,
                 restore_path: Optional[str] = None,
                 device: str = 'cuda',
                 align: int = 16,
                 use_autocast: bool = True,
                 model_id: Optional[str] = None,
                 file_path: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 revision: Optional[str] = None):
        """
        Args:
            restore_path: 权重文件本地路径(.pth, state_dict)。
                          为 None 或文件不存在时，会自动下载。
            device: 'cuda' 或 'cpu'。
            align: 输入空间尺寸需要对齐到的倍数(HRNet 下采样要求)。
            use_autocast: 推理时是否使用 torch.autocast。
            model_id / file_path / cache_dir / revision: 下载参数(可选)。
        """
        self.restore_path = ensure_weight(
            task=self.TASK_NAME,
            restore_path=restore_path,
            model_id=model_id,
            file_path=file_path,
            cache_dir=cache_dir,
            revision=revision,
        )
        self.device = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
        self.align = align
        self.use_autocast = use_autocast and self.device.type == 'cuda'

        self.model = self._build_model()

    def _build_model(self) -> nn.Module:
        """实例化 HRNet 并用 state_dict 加载权重。"""
        model = HRNet(in_channel=1, norm_type='bn')
        if self.use_autocast:
            state_dict = torch.load(self.restore_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        return model

    # ------------------------- 预处理 ------------------------- #
    def preprocess(self, seis: np.ndarray, clp_s: float = 3.0) -> torch.Tensor:
        """
        把 numpy 体数据 (T, H, W) 转成 (1, 1, T, H, W) 张量并放到设备上。
        z-score 截断到 [0,1] → 映射到 [-1,1]，不做 resize（resize 在多尺度循环内做）。
        """
        x = z_score_clip(seis.astype(np.float32), clp_s=clp_s) * 2 - 1
        return torch.from_numpy(x)[None, None].float().to(self.device), x

    # ------------------------- 多尺度推理 ------------------------- #
    @staticmethod
    def _normalize_scale(s: ScaleSpec) -> Tuple[float, float, float]:
        """统一成 (st, sh, sw) 三元组。"""
        if isinstance(s, (int, float)):
            return float(s), float(s), float(s)
        if len(s) != 3:
            raise ValueError(f"scale tuple must have 3 elements, got {s}")
        return tuple(float(v) for v in s)

    def _target_shape(self, t: int, h: int, w: int,
                      st: float, sh: float, sw: float):
        t1 = max(self.align, (int(t // self.align * st)) * self.align)
        h1 = max(self.align, (int(h // self.align * sh)) * self.align)
        w1 = max(self.align, (int(w // self.align * sw)) * self.align)
        return t1, h1, w1

    @torch.no_grad()
    def predict(self,
                seis: np.ndarray,
                scales: Optional[List[ScaleSpec]] = None,
                clp_s: float = 3.0,
                accumulate: str = 'sum'):
        """
        多尺度集成预测。

        Args:
            seis: numpy 数组，(T, H, W)。
            scales: 尺度列表，每项可以是 float（同尺度）或 (st, sh, sw) 三元组。
                    传 [1.0] 即单尺度。
            clp_s: numpy z-score 截断的 sigma。
            accumulate: 集成方式：
                - 'sum' (默认)：累加 sigmoid 概率，后续用绝对阈值（如 0.75）。
                - 'mean'：平均，后续阈值应在 [0,1] 区间（如 0.5）。

        Returns:
            scores_np: (T, H, W) 集成后的分数体（未阈值化），float32 numpy。
            seis_used: (T, H, W) 预处理后的地震体，float32 numpy（用于可视化）。
        """
        if scales is None:
            scales = self.DEFAULT_SCALES
        if not scales:
            raise ValueError("scales must contain at least one element.")
        if accumulate not in ('sum', 'mean'):
            raise ValueError(f"accumulate must be 'sum' or 'mean', got {accumulate!r}")

        data_tensor, seis_used = self.preprocess(seis, clp_s=clp_s)
        _, _, T, H, W = data_tensor.shape

        # 累加器放 CPU，避免一直占显存
        scores = torch.zeros((1, 1, T, H, W), dtype=torch.float32)

        if self.use_autocast:
            ctx_factory = lambda: torch.autocast(self.device.type)
        else:
            ctx_factory = contextlib.nullcontext

        for raw in scales:
            st, sh, sw = self._normalize_scale(raw)
            for name, s in (('scale_t', st), ('scale_h', sh), ('scale_w', sw)):
                if s <= 0:
                    raise ValueError(f"{name} must be > 0, got {s}")

            t1, h1, w1 = self._target_shape(T, H, W, st, sh, sw)
            x = F.interpolate(data_tensor, (t1, h1, w1), mode='trilinear')

            with ctx_factory():
                logits = self.model(x)
                prob = torch.sigmoid(logits)
                prob = F.interpolate(prob, (T, H, W), mode='trilinear')
            scores += prob.float().cpu()

        if accumulate == 'mean':
            scores /= float(len(scales))

        scores_np = scores[0, 0].numpy()
        return scores_np, seis_used

    # ------------------------- 后处理 ------------------------- #
    @staticmethod
    def postprocess(scores: np.ndarray,
                    threshold: float = 0.75,
                    min_size: Optional[int] = None,
                    connectivity: int = 1) -> np.ndarray:
        """
        分数体 → 二值 mask（可选移除小连通体）。

        Args:
            scores: (T, H, W) predict 返回的累加/平均分数体。
            threshold: 阈值。
                - accumulate='sum' 时，默认 0.75 是相对于累加值的；
                  若 scales 数变了，相应调整。
                - accumulate='mean' 时，常用 0.5。
            min_size: 若给定且 >0，对 mask 调用 remove_small_instances 过滤小连通体。
            connectivity: 连通性，1/2/3。

        Returns:
            float32 mask 体，形状同 scores（0/1）。
        """
        mask = scores > threshold
        if min_size is not None and min_size > 0:
            mask = remove_small_instances(mask, min_size=min_size, connectivity=connectivity)
        return mask.astype(np.float32)

    # ------------------------- 可视化 ------------------------- #
    @staticmethod
    def visualize(seis_np: np.ndarray,
                  scores_np: np.ndarray,
                  mask_np: Optional[np.ndarray] = None,
                  overlay_threshold: float = 0.75,
                  body_color: str = '#0499FD',
                  body_level: float = 0.5,
                  body_filter_sigma: float = 1.0,
                  body_margin: int = 0,
                  fg_cmap_name: str = 'jet'):
        """
        三联视图（与原脚本一致）：
          1. 仅地震；
          2. 地震 + 概率叠加（按 overlay_threshold 显示）；
          3. 地震 + isosurface 实体（只在传了 mask_np 时显示）。

        Args:
            seis_np: (T, H, W) 预处理过的地震体。
            scores_np: (T, H, W) predict 返回的分数体（未阈值化）。
            mask_np: (T, H, W) 后处理后的二值 mask（用于实体面板）。None 时第三面板退化为仅地震。
            overlay_threshold: 第二面板叠加显示用的阈值。
            body_color / body_level / body_filter_sigma / body_margin:
                透传给 cigvis.create_bodys 的参数。
            fg_cmap_name: 叠加层的基础 colormap，最小值会被设为透明。
        """
        seis_vol = seis_np.transpose(1, 2, 0)
        overlay_vol = (scores_np > overlay_threshold).astype(np.float32).transpose(1, 2, 0)

        fg_cmap = colormap.set_alpha_except_min(fg_cmap_name, alpha=1)

        node0 = cigvis.create_slices(seis_vol, cmap='gray')

        node1 = cigvis.create_slices(seis_vol, cmap='gray')
        node1 = cigvis.add_mask(node1, overlay_vol, cmaps=fg_cmap, interpolation='nearest')

        node2 = cigvis.create_slices(seis_vol, cmap='gray')
        if mask_np is not None:
            node2 += cigvis.create_bodys(
                mask_np.astype(np.float32).transpose(1, 2, 0),
                level=body_level,
                margin=body_margin,
                filter_sigma=body_filter_sigma,
                color=body_color,
            )

        cigvis.plot3D([node0, node1, node2], grid=[1, 3], share=1)
