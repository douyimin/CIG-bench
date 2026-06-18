"""
PropertyPredictor — encapsulates the GEM (rock-property) inference workflow.

Workflow:
    1. Load seismic, z-score-clip and map to [-1, 1].
    2. Load sparse well-log property volume (e.g. Vp, Density, Impedance),
       z-score-clip with zero-preservation, optional grey-dilation, map to [-1, 1].
    3. Resample seismic / property to a fixed inference grid (default 640 x 512 x 512).
       Seismic & property use trilinear; the well mask uses nearest-exact.
    4. Concatenate three channels (seismic + sparse property + binary well mask).
    5. Run model, apply tanh.
    6. Resample back to the original grid and (optionally) z-score-clip normalize.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import contextlib
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import grey_dilation

import cigvis
from cigvis import colormap

from ..networks.hrnet import HRNet

from .utils import (
    z_score_clip,
    z_score_clip_expzero,
)
from ._download import ensure_weight


class PropertyPredictor:
    """
    GEM 属性预测推理流程封装(Vp / Density / Impedance / GR 等)。

    权重加载方式：
      - 统一使用 state_dict (.pth)：实例化 ``model_cls()``，然后用
        ``model.load_state_dict(torch.load(restore_path, map_location='cpu'))`` 加载。
      - ``model_cls`` 默认是 ``lambda: HRNet(48)``，与作者发布的属性预测权重匹配。
      - ``restore_path`` 不传时，会自动从魔搭(ModelScope) 下载默认权重。

    备注：infer_shape 在 predict() 时传入，必须满足模型期望
    (默认 (640, 512, 512))，其它尺寸会导致结果变差或推理失败。

    典型用法
    --------
    >>> # 1) 自动从魔搭下载默认权重
    >>> predictor = PropertyPredictor(device="cuda")
    >>> # 2) 手动指定本地权重 + 自定义模型构造
    >>> predictor = PropertyPredictor(
    ...     restore_path="model_ema_step_162500.pth",
    ...     device="cuda",
    ...     model_cls=lambda: HRNet(48),
    ... )
    """

    TASK_NAME: str = "property"

    # ------------------------- 初始化 ------------------------- #
    def __init__(self,
                 restore_path: Optional[str] = None,
                 device: str = "cuda",
                 use_autocast: bool = True,
                 use_tanh: bool = True,
                 model_cls=None,
                 model_id: Optional[str] = None,
                 file_path: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 revision: Optional[str] = None):
        """
        Args:
            restore_path: 权重文件本地路径(.pth, state_dict)。
                          为 None 或文件不存在时，会自动下载。
            device: 'cuda' 或 'cpu'。
            use_autocast: 是否启用 torch.autocast。
            use_tanh: 是否对模型输出再过一次 tanh(原脚本行为)。
            model_cls: 用于实例化模型的无参 callable。默认 ``lambda: HRNet(48)``。
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
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.use_autocast = use_autocast and self.device.type == "cuda"
        self.use_tanh = use_tanh
        self.model_cls = model_cls if model_cls is not None else (lambda: HRNet(48))

        self.model = self._build_model()

    def _build_model(self) -> nn.Module:
        """实例化 model_cls() 并用 state_dict 加载权重。"""
        model = self.model_cls()
        state_dict = torch.load(self.restore_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        return model

    # ------------------------- 预处理 ------------------------- #
    def preprocess(self,
                   seis: np.ndarray,
                   prop: np.ndarray,
                   infer_shape: Tuple[int, int, int],
                   clp_s_seis: float = 3.0,
                   clp_s_prop: float = 2.0,
                   dilation_size: Tuple[int, int, int] = (1, 3, 3),
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
    Tuple[int, int, int], np.ndarray, np.ndarray]:
        """
        把 numpy 体数据转成模型可吃的张量，并返回稀疏井点信息用于可视化。

        过程：
            seis: z_score_clip → 映射到 [-1, 1] → trilinear 插值到 infer_shape
            prop: z_score_clip_expzero(只在非零位置 z-score) → 非零位置映射到
                  [-1, 1](零保持为零) → grey_dilation 膨胀稀疏井道 →
                  nearest-exact 插值到 infer_shape
            mask: 由 prop 非零位置生成的二值通道, nearest-exact 插值

        Args:
            seis: (T, H, W) numpy 数组。
            prop: (T, H, W) 稀疏井道属性体。
            infer_shape: 模型期望的 (T', H', W') 推理尺寸。

        Returns:
            seis_tensor: (1, 1, *infer_shape) 已在 device 上
            prop_tensor: (1, 1, *infer_shape) 已在 device 上
            mask_tensor: (1, 1, *infer_shape) 已在 device 上
            (T, H, W) 原始尺寸
            well_positions: (N, 3) int 数组, 原始体内非零井点位置 (用于可视化)
            well_values:    (N,)   float 数组, 对应的归一化后属性值 (用于可视化)
        """
        infer_shape = tuple(infer_shape)

        # ---- seismic ----
        s = z_score_clip(seis.astype(np.float32), clp_s=clp_s_seis) * 2 - 1  # [-1, 1]
        t, h, w = s.shape

        # ---- sparse property (only well-log locations are nonzero) ----
        p = z_score_clip_expzero(prop.astype(np.float32), clp_s=clp_s_prop)  # nonzero → [eps, 1]
        if dilation_size is not None:
            p = grey_dilation(p, size=dilation_size)
        # 零位置仍是零；非零位置从 [0, 1] 拉到 [-1, 1]
        p = np.where(p == 0, 0.0, p * 2 - 1).astype(np.float32)

        # 记录井点用于可视化(在原始尺寸下)
        well_positions = np.argwhere(p != 0)
        well_values = p[p != 0].astype(np.float32)

        # ---- to tensors & resize to infer_shape ----
        seis_t = torch.from_numpy(s)[None, None]  # (1,1,T,H,W)
        prop_t = torch.from_numpy(p)[None, None]
        seis_t = F.interpolate(seis_t, infer_shape, mode="trilinear")
        prop_t = F.interpolate(prop_t, infer_shape, mode="nearest-exact")
        mask_t = (prop_t != 0).to(prop_t.dtype)

        return (seis_t.to(self.device),
                prop_t.to(self.device),
                mask_t.to(self.device),
                (t, h, w),
                well_positions,
                well_values)

    # ------------------------- 推理 ------------------------- #
    @torch.no_grad()
    def predict(self,
                seis: np.ndarray,
                prop: Optional[np.ndarray] = None,
                infer_shape: Tuple[int, int, int] = (512, 512, 512),
                clp_s_seis: float = 3.5,
                clp_s_prop: float = 3.5,
                dilation_size: Tuple[int, int, int] = (1, 3, 3),
                resize_back: bool = True,
                normalize_output: bool = True,
                normalize_output_clp_s: float = 3.5):
        """
        对一个三维地震体 + 稀疏井道属性做属性预测。

        Args:
            seis: (T, H, W) numpy 数组。
            prop: (T, H, W) 稀疏井道属性体(零表示该位置无井)。
                  为 None 时使用全零(纯地震驱动)。
            infer_shape: 模型期望的 (T', H', W') 推理尺寸。默认 (640, 512, 512)。
                         不同数据/模型可以传入不同尺寸；其它尺寸可能导致结果变差。
            clp_s_seis: seismic 的 z-score 截断 sigma。
            clp_s_prop: 属性体 z-score 截断 sigma。
            dilation_size: 井道在水平方向的膨胀尺寸 (1, dy, dx)。
                           设为 None 跳过膨胀。
            resize_back: 是否把输出插值回原始 (T, H, W)。
            normalize_output: 是否对返回的属性体再做一次 z_score_clip 归一化。
            normalize_output_clp_s: 上述归一化使用的 sigma(原脚本是 2)。

        Returns:
            prop_volume: (T, H, W) 或 (T', H', W') 的属性体 (numpy float32)。
            seis_used:   与 prop_volume 同形的、预处理过的地震体 (numpy float32)。
            well_info:   dict {'positions': (N,3), 'values': (N,)},
                         供 visualize 叠加井点。
        """
        if prop is None:
            prop = np.zeros_like(seis, dtype=np.float32)

        (seis_t, prop_t, mask_t,
         (T, H, W),
         well_positions, well_values) = self.preprocess(
            seis, prop,
            infer_shape=infer_shape,
            clp_s_seis=clp_s_seis,
            clp_s_prop=clp_s_prop,
            dilation_size=dilation_size,
        )

        input_tensor = torch.cat([seis_t, prop_t, mask_t], dim=1)

        if self.use_autocast:
            ctx = torch.autocast(self.device.type)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            out = self.model(input_tensor)
            if self.use_tanh:
                out = torch.tanh(out)

        if resize_back:
            out = F.interpolate(out.float(), (T, H, W), mode="trilinear")
            seis_out = F.interpolate(seis_t.float(), (T, H, W), mode="trilinear")
        else:
            out = out.float()
            seis_out = seis_t.float()

        prop_np = out.cpu().numpy()[0, 0]
        seis_np = seis_out.cpu().numpy()[0, 0]

        if normalize_output:
            prop_np = z_score_clip(prop_np, clp_s=normalize_output_clp_s)

        well_info = {"positions": well_positions, "values": well_values}
        return prop_np, seis_np, well_info

    # ------------------------- 可视化 ------------------------- #
    @staticmethod
    def visualize(seis_np: np.ndarray,
                  prop_np: np.ndarray,
                  well_info: Optional[dict] = None,
                  fg_cmap_name: str = "AI"):
        """
        双联视图：seismic+wells / 预测属性+wells。

        Args:
            seis_np:    (T, H, W) 已预处理的地震体(predict 返回的 seis_used)。
            prop_np:    (T, H, W) 预测属性体(predict 返回的 prop_volume)。
            well_info:  dict, predict 返回的 well_info。为 None 时不叠加井点。
            fg_cmap_name: 属性体 colormap 名(默认 'AI')。
        """
        seis_vol = seis_np.transpose(1, 2, 0)  # (H, W, T)
        prop_vol = prop_np.transpose(1, 2, 0)

        fg_cmap = colormap.cmap_to_vispy(fg_cmap_name)

        # seismic + 井点
        nodes_seis = cigvis.create_slices(seis_vol, cmap="gray")
        if well_info is not None and len(well_info["positions"]) > 0:
            nodes_seis += cigvis.create_points(
                well_info["positions"], r=2, color=None,
                cmap=fg_cmap, vertex_values=well_info["values"].tolist(),
                shading=None,
            )

        # 预测属性 + 井点
        nodes_prop = cigvis.create_slices(prop_vol, cmap=fg_cmap)
        if well_info is not None and len(well_info["positions"]) > 0:
            nodes_prop += cigvis.create_points(
                well_info["positions"], r=2, color=None,
                cmap=fg_cmap, vertex_values=well_info["values"].tolist(),
                shading=None,
            )

        cigvis.plot3D([nodes_seis, nodes_prop], grid=(1, 2), share=True)


