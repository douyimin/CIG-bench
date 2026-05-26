"""
RGTPredictor — encapsulates the RGT-Est inference workflow.

Workflow:
    1. Load seismic, z-score-clip to [-1, 1].
    2. Resample to a fixed inference grid (default 400 x 512 x 512).
    3. Concatenate two auxiliary channels (horizon RGT + labelled mask),
       zeros by default = no horizon constraints.
    4. Replication-pad by 8, run model, center-crop.
    5. Resample back to the original grid and normalize.
    6. (Optional) Trace horizons as iso-surfaces of the RGT volume.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import contextlib
import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cigvis
from cigvis import colormap



from .utils import z_score_clip, normalization, horizons_from_rgt, horizon_image
from ._download import ensure_weight
from ..networks.hrnet import HRNet


class RGTPredictor:
    """
    RGT-Est 推理流程封装。

    权重加载方式：
      - 统一使用 state_dict (.pth)：实例化 ``model_cls()``，然后用
        ``model.load_state_dict(torch.load(restore_path, map_location='cpu'))`` 加载。
      - ``model_cls`` 默认是 ``lambda: HRNet()`` (c=48, 3 输入通道：
        seismic + horizon_rgt + horizon_mask)，与 RGT-Est 权重匹配。
      - ``restore_path`` 不传时，会自动下载默认权重，下载源由 ``source`` 指定：
          * ``source='modelscope'`` (默认) —— 从魔搭(ModelScope) 下载
          * ``source='huggingface'``       —— 从 Hugging Face Hub 下载

    备注：infer_shape 必须满足模型期望(论文/作者建议 (400, 512, 512))，
    其它尺寸会导致结果变差或推理失败。

    典型用法
    --------
    >>> # 1) 自动从魔搭下载默认权重
    >>> predictor = RGTPredictor(device="cuda")
    >>> # 2) 改用 Hugging Face 源
    >>> predictor = RGTPredictor(device="cuda", source="huggingface")
    >>> # 3) 手动指定本地权重
    >>> predictor = RGTPredictor("RGT-Est_CIG-Benchmark.pth", device="cuda")
    """

    TASK_NAME: str = "rgt"

    # ------------------------- 初始化 ------------------------- #
    def __init__(self,
                 restore_path: Optional[str] = None,
                 device: str = "cuda",
                 infer_shape: Tuple[int, int, int] = (400, 512, 512),
                 pad: int = 8,
                 use_autocast: bool = True,
                 model_cls=None,
                 model_id: Optional[str] = None,
                 file_path: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 revision: Optional[str] = None,
                 source: Optional[str] = None):
        """
        Args:
            restore_path: 权重文件本地路径(.pth, state_dict)。
                          为 None 或文件不存在时，会自动下载。
            device: 'cuda' 或 'cpu'。
            infer_shape: 模型期望的 (T, H, W) 推理尺寸。默认 (400, 512, 512)。
            pad: 推理前 ReplicationPad3d 的 padding 大小，与中心裁剪保持一致。
            use_autocast: 是否启用 torch.autocast。
            model_cls: 用于实例化模型的无参 callable。默认 ``lambda: HRNet()`` (c=48)。
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
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.infer_shape = tuple(infer_shape)
        self.pad = pad
        self.use_autocast = use_autocast and self.device.type == "cuda"
        self.model_cls = model_cls if model_cls is not None else (lambda: HRNet())

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
                   clp_s: float = 2.0) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        把 numpy 体数据 (T, H, W) 转成模型可吃的张量 (1, 1, T', H', W')。

        过程：z-score 截断到 [0,1] → 映射到 [-1,1] → resize 到 infer_shape。
        返回 (seis_tensor_on_device, (T, H, W) 原始尺寸)。
        """
        x = z_score_clip(seis.astype(np.float32), clp_s=clp_s) * 2 - 1   # (T, H, W) in [-1, 1]
        t, h, w = x.shape
        x = torch.from_numpy(x)[None, None]                              # (1, 1, T, H, W)
        x = F.interpolate(x, self.infer_shape, mode="trilinear")
        return x.to(self.device), (t, h, w)

    # ------------------------- 推理 ------------------------- #
    @torch.no_grad()
    def predict(self,
                seis: np.ndarray,
                clp_s: float = 2.0,
                horizon_rgt: Optional[np.ndarray] = None,
                horizon_mask: Optional[np.ndarray] = None,
                resize_back: bool = True,
                normalize_output: bool = True):
        """
        对一个三维地震体跑 RGT 预测。

        Args:
            seis: (T, H, W) numpy 数组。
            clp_s: z-score 截断的 sigma 阈值。
            horizon_rgt:  (T, H, W) 可选 horizon RGT 值体。None → 零通道。
            horizon_mask: (T, H, W) 可选标注 mask 体。None → 零通道。
            resize_back: 是否把 RGT 插值回原始 (T, H, W)。
            normalize_output: 是否对返回的 RGT 体再做一次 z_score_clip 归一化
                              （和原脚本一致）。

        Returns:
            rgt_volume: (T, H, W) 或 (T', H', W') 的 RGT 体（numpy float32）。
            seis_used:  与 rgt_volume 同形的、预处理过的地震体（numpy float32），
                        方便直接喂 visualize。
        """
        seis_tensor, orig_shape = self.preprocess(seis, clp_s=clp_s)
        T, H, W = orig_shape

        # 两个辅助通道：默认零，等价于"无 horizon 约束"
        if horizon_rgt is None:
            ch_rgt = torch.zeros_like(seis_tensor)
        else:
            ch_rgt = self._prep_aux_channel(horizon_rgt)
        if horizon_mask is None:
            ch_msk = torch.zeros_like(seis_tensor)
        else:
            ch_msk = self._prep_aux_channel(horizon_mask)

        input_tensor = torch.cat([seis_tensor, ch_rgt, ch_msk], dim=1)

        if self.use_autocast:
            ctx = torch.autocast(self.device.type)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            p = self.pad
            padded = nn.ReplicationPad3d(p)(input_tensor)
            rgt = self.model(padded)
            if p > 0:
                rgt = rgt[:, :, p:-p, p:-p, p:-p]

        if resize_back:
            rgt = F.interpolate(rgt.float(), (T, H, W), mode="trilinear")
            seis_out = F.interpolate(seis_tensor.float(), (T, H, W), mode="trilinear")
        else:
            rgt = rgt.float()
            seis_out = seis_tensor.float()

        rgt_np = rgt.cpu().numpy()[0, 0]
        seis_np = seis_out.cpu().numpy()[0, 0]

        if normalize_output:
            rgt_np = z_score_clip(rgt_np)

        return rgt_np, seis_np

    def _prep_aux_channel(self, vol: np.ndarray) -> torch.Tensor:
        """把一个 (T, H, W) 辅助体转换成 (1, 1, *infer_shape) 张量。"""
        v = torch.from_numpy(vol.astype(np.float32))[None, None]
        v = F.interpolate(v, self.infer_shape, mode="trilinear")
        return v.to(self.device)

    # ------------------------- horizon 提取 ------------------------- #
    @staticmethod
    def extract_horizons(rgt_volume: np.ndarray,
                         n_horizons: int = 100,
                         sigma: float = 1.0,
                         d: int = 3,
                         image_sigma: float = 0.0) -> np.ndarray:
        """
        把 (T, H, W) 的 RGT 体转成 horizon mask 体，mask 值为该 horizon 上的 RGT 值。

        Returns:
            (H, W, T) 的 float32 数组，可直接用 cigvis.add_mask 叠加。
        """
        ux = rgt_volume.transpose(1, 2, 0)        # (H, W, T)
        n3, n2, n1 = ux.shape

        hu = np.linspace(0.0, 1.0, n_horizons)
        hzs = horizons_from_rgt(sig=sigma, hu=hu, ux=ux)

        mask = horizon_image(n1=n1, n2=n2, n3=n3, d=d, sig=image_sigma, x1=hzs)
        mask = (mask > 0).astype(np.float32) * ux
        return mask

    # ------------------------- 可视化 ------------------------- #
    @staticmethod
    def visualize(seis_np: np.ndarray,
                  rgt_np: np.ndarray,
                  horizon_mask: Optional[np.ndarray] = None,
                  fg_cmap_name: str = "AI"):
        """
        三联视图：seismic / RGT / seismic + horizons（如提供 horizon_mask）。

        Args:
            seis_np: (T, H, W) 已预处理的地震体（predict 返回的 seis_used）。
            rgt_np:  (T, H, W) RGT 体（predict 返回的 rgt_volume）。
            horizon_mask: (H, W, T) horizon mask 体（extract_horizons 返回值）。
                          为 None 时不显示叠加面板。
        """
        seis_vol = seis_np.transpose(1, 2, 0)         # (H, W, T)
        rgt_vol = normalization(rgt_np.transpose(1, 2, 0))

        nodes_seis = cigvis.create_slices(seis_vol, cmap="gray")
        nodes_rgt = cigvis.create_slices(rgt_vol, cmap=fg_cmap_name)

        if horizon_mask is not None:
            fg_cmap = colormap.set_alpha_except_min(fg_cmap_name, alpha=1)
            nodes_overlay = cigvis.create_slices(seis_vol, cmap="gray")
            cigvis.add_mask(nodes_overlay, horizon_mask,
                            cmaps=fg_cmap, interpolation="nearest")
            cigvis.plot3D([nodes_seis, nodes_rgt, nodes_overlay],
                          grid=(1, 3), share=True)
        else:
            cigvis.plot3D([nodes_seis, nodes_rgt], grid=(1, 2), share=True)


# ---------------------------------------------------------------------------
# 脚本入口：复现原 demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")

    seis = np.load(r"../RealData/Poseidon_part2.npy").astype(np.float32)  # (T, H, W)

    # 不传 restore_path → 自动从魔搭下载
    predictor = RGTPredictor(device="cuda")

    rgt_volume, seis_used = predictor.predict(seis, clp_s=2.0)
    horizon_mask = predictor.extract_horizons(rgt_volume, n_horizons=100)
    predictor.visualize(seis_used, rgt_volume, horizon_mask)