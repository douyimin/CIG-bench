import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import contextlib
import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cigvis
from cigvis import colormap
from ..networks.hrnet_skipconect_opt import HRNet
from .utils import z_score_clip, tensor_z_score_clip
from ._download import ensure_weight


# ---------------------------------------------------------------------------
# 推理类
# ---------------------------------------------------------------------------
class FaultPredictor:
    """
    把 HRNet 故障/通道预测的完整流程封装为一个类。

    权重加载方式：
      - 统一使用 state_dict (.pth)：先 ``HRNet(c=...)`` 实例化，再用
        ``model.load_state_dict(torch.load(restore_path, map_location='cpu'))`` 加载。
      - ``restore_path`` 不传时，会自动下载默认权重，下载源由 ``source`` 指定：
          * ``source='modelscope'`` (默认) —— 从魔搭(ModelScope) 下载
          * ``source='huggingface'``       —— 从 Hugging Face Hub 下载
      - 推理时支持 ``rank`` / ``chunk_size`` 这类显存优化参数。

    缩放系数 scale_t / scale_h / scale_w 在 predict() 调用时传入，
    比如 scale_t=0.5 表示 t 方向缩到 0.5 倍后再推理。

    Z-score 截断 sigma 也在 predict() 调用时传入：
      - clp_s_pre:  numpy 阶段（载入直后）的截断 sigma，默认 4.0
      - clp_s_in:   送入模型前的截断 sigma，默认 1.5

    典型用法
    --------
    >>> # 1) 自动从魔搭下载默认权重
    >>> predictor = FaultPredictor(c=32, device='cuda')
    >>> # 2) 改用 Hugging Face 源
    >>> predictor = FaultPredictor(c=32, device='cuda', source='huggingface')
    >>> # 3) 手动指定本地权重
    >>> predictor = FaultPredictor('fautmodel.pth', c=32, device='cuda')
    >>> result, seis_used = predictor.predict(seis, rank=4, chunk_size=64)
    """

    TASK_NAME: str = "fault"

    # ------------------------- 初始化 ------------------------- #
    def __init__(self,
                 restore_path: Optional[str] = None,
                 c: int = 32,
                 device: str = 'cuda',
                 align: int = 16,
                 use_autocast: bool = True,
                 model_id: Optional[str] = None,
                 file_path: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 revision: Optional[str] = None,
                 source: Optional[str] = None):
        """
        Args:
            restore_path: 权重文件本地路径(.pth, state_dict)。
                          为 None 或文件不存在时，会自动下载。
            c: HRNet 的 base channel。
            device: 'cuda' 或 'cpu'。
            align: 输入空间尺寸需要对齐到的倍数(HRNet 下采样要求)。
            use_autocast: 推理时是否使用 torch.autocast。
            model_id / file_path / cache_dir / revision: 下载参数(可选)。
            source: 'modelscope' (默认) 或 'huggingface'。
        """
        self.restore_path = ensure_weight(
            task=self.TASK_NAME,
            restore_path=restore_path,
            model_id=model_id,
            file_path=file_path,
            cache_dir=cache_dir,
            revision=revision,
            source=source,
        )
        self.c = c
        self.device = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
        self.align = align
        self.use_autocast = use_autocast and self.device.type == 'cuda'

        self.model = self._build_model()

    def _build_model(self) -> nn.Module:
        """实例化 HRNet(c=...) 并用 state_dict 加载权重。"""
        model = HRNet(c=self.c)
        state_dict = torch.load(self.restore_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        return model

    # ------------------------- 预处理 ------------------------- #
    def _target_shape(self, t: int, h: int, w: int,
                      scale_t: float, scale_h: float, scale_w: float):
        """根据 scale 与 align 计算实际送进模型的 (t1, h1, w1)。"""
        t1 = max(self.align, (int(round(t * scale_t)) // self.align) * self.align)
        h1 = max(self.align, (int(round(h * scale_h)) // self.align) * self.align)
        w1 = max(self.align, (int(round(w * scale_w)) // self.align) * self.align)
        return t1, h1, w1

    def preprocess(self, seis: np.ndarray,
                   scale_t: float = 1.0,
                   scale_h: float = 1.0,
                   scale_w: float = 1.0,
                   #clp_s_pre: float = 4.0
                   ):
        """
        把 numpy 体数据 (t, h, w) 转成模型可吃的张量 (1,1,t1,h1,w1)：
        numpy z-score 截断（clp_s_pre）→ 按 scale_* 缩放 → trilinear 对齐到 self.align 的倍数。

        Returns:
            input_tensor: 已经放到 self.device 上、缩放并对齐后的张量
            orig_shape:   (t, h, w) 原始尺寸，用于 resize_back
        """
        #seis = z_score_clip(seis.astype(np.float32), clp_s=clp_s_pre)
        x = torch.from_numpy(np.asarray(seis))[None, None].float()
        _, _, t, h, w = x.shape
        t1, h1, w1 = self._target_shape(t, h, w, scale_t, scale_h, scale_w)
        x = F.interpolate(x, (t1, h1, w1), mode='trilinear')
        return x.to(self.device), (t, h, w)

    # ------------------------- 推理 ------------------------- #
    @torch.no_grad()
    def predict(self,
                seis: np.ndarray,
                rank: int = 4,
                chunk_size: int = 64,
                threshold: float = 0.5,
                resize_back: bool = False,
                scale_t: float = 1.0,
                scale_h: float = 1.0,
                scale_w: float = 1.0,
                scale: float = None,
                clp_s_pre: float = 4.0,
                clp_s_in: float = 3.0):
        """
        对一个三维地震体进行预测。

        Args:
            seis: numpy 数组，(t, h, w)。
            rank, chunk_size: 透传给 HRNet.forward 的显存优化参数。
                              jit 模式下被忽略（只支持 rank=0 直接推理）。
            threshold: 概率阈值，小于该值置零。
            resize_back: 若为 True，则把 result 与 input 都插值回原始 (t, h, w) 尺寸；
                         否则返回缩放后的工作尺寸。
            scale_t/h/w: 三个轴各自的缩放系数（先缩放再 align 对齐）。
                         例如 scale_t=0.5 表示 t 方向缩到 0.5 倍后再推理。
            scale: 便捷参数；若给定，会同时覆盖 scale_t/h/w 三个值。
            clp_s_pre: numpy 阶段 z-score 截断的 sigma，默认 4.0。
            clp_s_in:  送入模型前 tensor 阶段 z-score 截断的 sigma，默认 1.5。
                       输入会被 clip 到 [-clp_s_in, clp_s_in] 然后归一化并 *2-1 映射到 [-1, 1]。

        Returns:
            result_np: 概率体（已阈值化）的 numpy
            input_np:  对应的输入体 numpy（与 result_np 同形）
        """
        # 处理统一 scale 覆盖
        if scale is not None:
            scale_t = scale_h = scale_w = scale
        for name, s in (('scale_t', scale_t), ('scale_h', scale_h), ('scale_w', scale_w)):
            if s <= 0:
                raise ValueError(f"{name} must be > 0, got {s}")
        for name, s in (('clp_s_pre', clp_s_pre), ('clp_s_in', clp_s_in)):
            if s <= 0:
                raise ValueError(f"{name} must be > 0, got {s}")

        input_tensor, orig_shape = self.preprocess(
            seis, scale_t, scale_h, scale_w, clp_s_pre=clp_s_pre)

        # 推理阶段再做一次更紧的 z-score 截断并映射到 [-1, 1]
        x = tensor_z_score_clip(input_tensor, clp_s=clp_s_in) * 2 - 1

        if self.use_autocast:
            ctx = torch.autocast(self.device.type)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            logits = self.model(x, rank=rank, chunk_size=chunk_size)
            result = torch.sigmoid(logits)

        # 阈值化
        result = torch.where(result > threshold, result, torch.zeros_like(result))

        if resize_back:
            result = F.interpolate(result, orig_shape, mode='nearest-exact')
            input_tensor = F.interpolate(input_tensor, orig_shape, mode='trilinear')

        result_np = result[0, 0].float().cpu().numpy()
        input_np = input_tensor[0, 0].float().cpu().numpy()
        return result_np, input_np

    # ------------------------- 可视化 ------------------------- #
    @staticmethod
    def visualize(seis_np: np.ndarray,
                  result_np: np.ndarray,
                  seis_cmap: str = 'gray',
                  fg_cmap_name: str = 'jet',
                  show_input_panel: bool = True):
        """
        用 cigvis 显示叠加结果。输入约定 shape 为 (t, h, w)，内部会 transpose 成 (h, w, t)。
        """
        seis_vol = seis_np.transpose(1, 2, 0)
        mask_vol = result_np.transpose(1, 2, 0).astype(np.float32)

        fg_cmap = colormap.set_alpha_except_min(fg_cmap_name, alpha=1)
        base_node = cigvis.create_slices(seis_vol, cmap=seis_cmap)
        overlay = cigvis.add_mask(base_node, mask_vol,
                                  cmaps=fg_cmap, interpolation='nearest')

        if show_input_panel:
            ref_node = cigvis.create_slices(seis_vol, cmap=seis_cmap)
            cigvis.plot3D([ref_node, overlay], grid=[1, 2], share=1)
        else:
            cigvis.plot3D(overlay)

    @staticmethod
    def preview_input(seis: np.ndarray, cmap: str = 'seismic', clp_s: float = 4.0):
        """
        快速看一眼原始（未对齐）地震体，等价于原脚本最开始那个 cigvis.plot3D。
        输入 (t, h, w)。

        Args:
            clp_s: numpy z-score 截断 sigma。
        """
        vol = z_score_clip(seis.astype(np.float32).transpose(1, 2, 0), clp_s=clp_s)
        cigvis.plot3D(cigvis.create_slices(vol, cmap=cmap))