import itertools
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


Int3 = Tuple[int, int, int]


def _to_3tuple(x: Union[int, Sequence[int]]) -> Int3:
    if isinstance(x, int):
        return (x, x, x)
    x = tuple(int(v) for v in x)
    assert len(x) == 3
    return x


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _tile_index_3d(
    i: int,
    j: int,
    k: int,
    block_size: int,
    shape: Int3,
    halo: int,
):
    """Return valid region, padded patch region, crop region and F.pad list.

    F.pad order for a 5D tensor is [w0, w1, h0, h1, d0, d1].
    """
    d, h, w = shape
    vd0, vh0, vw0 = i * block_size, j * block_size, k * block_size
    vd1, vh1, vw1 = min(vd0 + block_size, d), min(vh0 + block_size, h), min(vw0 + block_size, w)

    pd0, ph0, pw0 = max(0, vd0 - halo), max(0, vh0 - halo), max(0, vw0 - halo)
    pd1, ph1, pw1 = min(d, vd1 + halo), min(h, vh1 + halo), min(w, vw1 + halo)

    pad_d0 = max(0, halo - (vd0 - pd0))
    pad_h0 = max(0, halo - (vh0 - ph0))
    pad_w0 = max(0, halo - (vw0 - pw0))
    pad_d1 = max(0, halo - (pd1 - vd1))
    pad_h1 = max(0, halo - (ph1 - vh1))
    pad_w1 = max(0, halo - (pw1 - vw1))
    padlist = [pad_w0, pad_w1, pad_h0, pad_h1, pad_d0, pad_d1]

    rd0 = vd0 - pd0 + pad_d0
    rh0 = vh0 - ph0 + pad_h0
    rw0 = vw0 - pw0 + pad_w0
    rd1 = rd0 + (vd1 - vd0)
    rh1 = rh0 + (vh1 - vh0)
    rw1 = rw0 + (vw1 - vw0)

    valid = (vd0, vh0, vw0, vd1, vh1, vw1)
    patch = (pd0, ph0, pw0, pd1, ph1, pw1)
    crop = (rd0, rh0, rw0, rd1, rh1, rw1)
    return valid, patch, crop, padlist


def set_pad_as_zero(x: Tensor, padlist: Sequence[int]) -> Tensor:
    """Set artificial halo regions to zero after an intermediate padded convolution.

    This is required when a fused block contains more than one padded convolution.
    Otherwise, the first convolution would create non-zero activations outside the
    real volume boundary, and the next convolution could incorrectly use them.
    """
    if padlist[0] > 0:
        x[:, :, :, :, :padlist[0]] = 0
    if padlist[1] > 0:
        x[:, :, :, :, -padlist[1]:] = 0
    if padlist[2] > 0:
        x[:, :, :, :padlist[2], :] = 0
    if padlist[3] > 0:
        x[:, :, :, -padlist[3]:, :] = 0
    if padlist[4] > 0:
        x[:, :, :padlist[4], :, :] = 0
    if padlist[5] > 0:
        x[:, :, -padlist[5]:, :, :] = 0
    return x


def _conv_out_size_1d(in_size: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    return (in_size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


@torch.no_grad()
def interpolate3d_chunked(
    x: Tensor,
    block_size: int = 64,
    scale_factor: int = 2,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    size: Optional[Int3] = None,
) -> Tensor:
    """Operator-level chunked interpolation.

    This is different from patch-wise inference: only the interpolation operator is tiled, while the
    upstream and downstream tensors remain full-volume tensors.
    """
    assert x.ndim == 5, "Only 5D tensors (N, C, D, H, W) are supported."
    if mode == "nearest":
        align_corners = None
        halo = 0
    else:
        # A one-voxel halo is sufficient for trilinear local interpolation.
        halo = 1

    b, c, d, h, w = x.shape
    if size is None:
        out_d, out_h, out_w = d * scale_factor, h * scale_factor, w * scale_factor
    else:
        out_d, out_h, out_w = size
        if (out_d, out_h, out_w) != (d * scale_factor, h * scale_factor, w * scale_factor):
            # Non-integer or uneven scaling is uncommon in this HRNet. Fall back to PyTorch for safety.
            return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)

    out = torch.empty((b, c, out_d, out_h, out_w), device=x.device, dtype=x.dtype)
    tile = max(1, block_size - 2 * halo)
    nd, nh, nw = _ceil_div(d, tile), _ceil_div(h, tile), _ceil_div(w, tile)

    for ii, jj, kk in itertools.product(range(nd), range(nh), range(nw)):
        (vd0, vh0, vw0, vd1, vh1, vw1), (pd0, ph0, pw0, pd1, ph1, pw1), (rd0, rh0, rw0, rd1, rh1, rw1), _ = _tile_index_3d(
            ii, jj, kk, tile, (d, h, w), halo
        )
        patch = x[:, :, pd0:pd1, ph0:ph1, pw0:pw1]
        patch = F.interpolate(patch, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
        out[:, :, vd0 * scale_factor:vd1 * scale_factor, vh0 * scale_factor:vh1 * scale_factor, vw0 * scale_factor:vw1 * scale_factor] = patch[
            :, :, rd0 * scale_factor:rd1 * scale_factor, rh0 * scale_factor:rh1 * scale_factor, rw0 * scale_factor:rw1 * scale_factor
        ]
    return out


class NearestUpsample3d(nn.Module):
    """A drop-in nearest upsample layer with optional chunked execution."""

    def __init__(self, scale_factor: int = 2, chunk_size: Optional[int] = None):
        super().__init__()
        self.scale_factor = int(scale_factor)
        self.chunk_size = chunk_size

    def forward(self, x: Tensor) -> Tensor:
        if self.chunk_size is None:
            return F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        return interpolate3d_chunked(x, self.chunk_size, self.scale_factor, mode="nearest")


@torch.no_grad()
def conv3d_chunked(x: Tensor, conv: nn.Conv3d, block_size: int = 128) -> Tensor:
    """Chunk a Conv3d operator with correct halo padding.

    This is intended for inference. It is numerically equivalent to conv(x) for common Conv3d
    settings used here: dilation=1, groups inherited from conv, and integer downsampling.
    """
    assert not conv.training, "Chunked Conv3d is for eval/inference mode."
    assert x.ndim == 5

    b, _, d, h, w = x.shape
    kernel = _to_3tuple(conv.kernel_size)
    stride = _to_3tuple(conv.stride)
    padding = _to_3tuple(conv.padding)
    dilation = _to_3tuple(conv.dilation)
    assert dilation == (1, 1, 1), "This helper currently assumes dilation=1."

    out_d = _conv_out_size_1d(d, kernel[0], stride[0], padding[0], dilation[0])
    out_h = _conv_out_size_1d(h, kernel[1], stride[1], padding[1], dilation[1])
    out_w = _conv_out_size_1d(w, kernel[2], stride[2], padding[2], dilation[2])

    # The HRNet stem/transition convolutions use either stride 1 or 2 with exact scale.
    scale_d, scale_h, scale_w = d // out_d, h // out_h, w // out_w
    assert (d % out_d, h % out_h, w % out_w) == (0, 0, 0)

    old_padding = conv.padding
    old_padding_mode = conv.padding_mode
    conv.padding = (0, 0, 0)
    if old_padding_mode == "zeros":
        pad_mode = "constant"
        pad_value = 0.0
    else:
        pad_mode = old_padding_mode
        pad_value = None
        conv.padding_mode = "zeros"

    out = torch.empty((b, conv.out_channels, out_d, out_h, out_w), device=x.device, dtype=x.dtype)
    tile = max(stride[0], block_size)
    nd, nh, nw = _ceil_div(d, tile), _ceil_div(h, tile), _ceil_div(w, tile)

    try:
        for ii, jj, kk in itertools.product(range(nd), range(nh), range(nw)):
            vd0, vh0, vw0 = ii * tile, jj * tile, kk * tile
            vd1, vh1, vw1 = min(vd0 + tile, d), min(vh0 + tile, h), min(vw0 + tile, w)

            pd0, ph0, pw0 = max(0, vd0 - padding[0]), max(0, vh0 - padding[1]), max(0, vw0 - padding[2])
            pd1, ph1, pw1 = min(d, vd1 + padding[0]), min(h, vh1 + padding[1]), min(w, vw1 + padding[2])

            padlist = [0, 0, 0, 0, 0, 0]
            if vd0 == pd0:
                padlist[4] = padding[0]
            if vh0 == ph0:
                padlist[2] = padding[1]
            if vw0 == pw0:
                padlist[0] = padding[2]
            if vd1 == pd1:
                padlist[5] = padding[0]
            if vh1 == ph1:
                padlist[3] = padding[1]
            if vw1 == pw1:
                padlist[1] = padding[2]

            patch = x[:, :, pd0:pd1, ph0:ph1, pw0:pw1]
            if sum(padlist) > 0:
                patch = F.pad(patch, padlist, mode=pad_mode, value=pad_value)
            patch = conv(patch)

            od0, oh0, ow0 = vd0 // scale_d, vh0 // scale_h, vw0 // scale_w
            od1, oh1, ow1 = vd1 // scale_d, vh1 // scale_h, vw1 // scale_w
            out[:, :, od0:od1, oh0:oh1, ow0:ow1] = patch
    finally:
        conv.padding = old_padding
        conv.padding_mode = old_padding_mode
    return out


@torch.no_grad()
def conv_transpose3d_chunked(x: Tensor, deconv: nn.ConvTranspose3d, block_size: int = 64) -> Tensor:
    """Chunk a ConvTranspose3d operator used in the decoder."""
    assert not deconv.training, "Chunked ConvTranspose3d is for eval/inference mode."
    assert x.ndim == 5

    b, _, d, h, w = x.shape
    stride = _to_3tuple(deconv.stride)
    padding = _to_3tuple(deconv.padding)
    kernel = _to_3tuple(deconv.kernel_size)
    dilation = _to_3tuple(deconv.dilation)
    output_padding = _to_3tuple(deconv.output_padding)

    out_d = (d - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel[0] - 1) + output_padding[0] + 1
    out_h = (h - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel[1] - 1) + output_padding[1] + 1
    out_w = (w - 1) * stride[2] - 2 * padding[2] + dilation[2] * (kernel[2] - 1) + output_padding[2] + 1

    scale_d, scale_h, scale_w = out_d // d, out_h // h, out_w // w
    assert (out_d % d, out_h % h, out_w % w) == (0, 0, 0)

    halo = max(padding)
    tile = max(1, block_size - 2 * halo)
    out = torch.empty((b, deconv.out_channels, out_d, out_h, out_w), device=x.device, dtype=x.dtype)
    nd, nh, nw = _ceil_div(d, tile), _ceil_div(h, tile), _ceil_div(w, tile)

    for ii, jj, kk in itertools.product(range(nd), range(nh), range(nw)):
        vd0, vh0, vw0 = ii * tile, jj * tile, kk * tile
        vd1, vh1, vw1 = min(vd0 + tile, d), min(vh0 + tile, h), min(vw0 + tile, w)

        pd0, ph0, pw0 = max(0, vd0 - halo), max(0, vh0 - halo), max(0, vw0 - halo)
        pd1, ph1, pw1 = min(d, vd1 + halo), min(h, vh1 + halo), min(w, vw1 + halo)

        rd0, rh0, rw0 = (vd0 - pd0) * scale_d, (vh0 - ph0) * scale_h, (vw0 - pw0) * scale_w
        rd1 = rd0 + (vd1 - vd0) * scale_d
        rh1 = rh0 + (vh1 - vh0) * scale_h
        rw1 = rw0 + (vw1 - vw0) * scale_w

        od0, oh0, ow0 = vd0 * scale_d, vh0 * scale_h, vw0 * scale_w
        od1, oh1, ow1 = vd1 * scale_d, vh1 * scale_h, vw1 * scale_w

        patch = deconv(x[:, :, pd0:pd1, ph0:ph1, pw0:pw1])
        out[:, :, od0:od1, oh0:oh1, ow0:ow1] = patch[:, :, rd0:rd1, rh0:rh1, rw0:rw1]
    return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm3d(planes * self.expansion)
        self.act_fun = nn.SiLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.act_fun(self.norm1(self.conv1(x)))
        out = self.act_fun(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.act_fun(out + residual)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.BatchNorm3d(planes)
        self.act_fun = nn.SiLU(inplace=True)
        # Important fix: the second convolution receives `planes` channels, not `inplanes`.
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm3d(planes)
        if downsample is None and (stride != 1 or inplanes != planes):
            downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.act_fun(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.act_fun(out + residual)


class StageModule(nn.Module):
    def __init__(self, stage, output_branches, c, upsample_chunk_size: Optional[int] = None):
        super().__init__()
        self.stage = stage
        self.output_branches = output_branches
        self.upsample_chunk_size = upsample_chunk_size

        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = c * (2 ** i)
            self.branches.append(nn.Sequential(BasicBlock(w, w), BasicBlock(w, w), BasicBlock(w, w), BasicBlock(w, w)))

        self.fuse_layers = nn.ModuleList()
        for i in range(self.output_branches):
            fuse_i = nn.ModuleList()
            for j in range(self.stage):
                if i == j:
                    fuse_i.append(nn.Sequential())
                elif i < j:
                    fuse_i.append(nn.Sequential(
                        nn.Conv3d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                        nn.BatchNorm3d(c * (2 ** i)),
                        NearestUpsample3d(scale_factor=2 ** (j - i), chunk_size=upsample_chunk_size),
                    ))
                else:
                    ops = []
                    for _ in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv3d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm3d(c * (2 ** j)),
                            nn.SiLU(inplace=True),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv3d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm3d(c * (2 ** i)),
                    ))
                    fuse_i.append(nn.Sequential(*ops))
            self.fuse_layers.append(fuse_i)
        self.act_fun = nn.SiLU(inplace=True)

    def set_upsample_chunk_size(self, chunk_size: Optional[int]):
        self.upsample_chunk_size = chunk_size
        for m in self.modules():
            if isinstance(m, NearestUpsample3d):
                m.chunk_size = chunk_size

    def forward(self, x):
        assert len(self.branches) == len(x)
        x = [branch(b) for branch, b in zip(self.branches, x)]

        x_fused = []
        for i in range(len(self.fuse_layers)):
            y = None
            target_size = None
            for j in range(len(self.branches)):
                z = self.fuse_layers[i][j](x[j])
                if y is None:
                    y = z
                    target_size = z.shape[2:]
                else:
                    if z.shape[2:] != target_size:
                        if self.upsample_chunk_size is None:
                            z = F.interpolate(z, size=target_size, mode="nearest")
                        else:
                            z = interpolate3d_chunked(
                                z,
                                block_size=self.upsample_chunk_size,
                                mode="nearest",
                                size=target_size,
                            )
                    y = y + z
            x_fused.append(self.act_fun(y))
        return x_fused


class HRNet(nn.Module):
    """3D HRNet with an inference-only memory-efficient path.

    Usage:
        model = HRNet(c=32).eval()
        y0 = model(x)                         # original forward
        y1 = model(x, rank=1, chunk_size=64)  # chunk high-risk operators
        y2 = model(x, rank=2, chunk_size=64)  # plus chunk/fuse final decoder
        y3 = model(x, rank=3, chunk_size=64)  # chunk final decoder and recompute skip_x1 at output
        y4 = model(x, rank=4, chunk_size=64)  # lowest-memory path: never materialize skip_x1 or full-resolution dec_block2 feature

    `rank>0` is for inference only. Keep `rank=0` during training.
    """

    def __init__(self, c=32, out_channels: int = 1):
        super().__init__()
        self.base = c
        self.out_channels = out_channels

        self.skip_layer1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.SiLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.SiLU(inplace=True),
        )
        self.skip_layer2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.SiLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.SiLU(inplace=True),
        )

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm2 = nn.BatchNorm3d(64)
        self.act_fun = nn.SiLU(inplace=True)

        downsample = nn.Sequential(
            nn.Conv3d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(256),
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
        )

        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(256, c, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(c),
                nn.SiLU(inplace=True),
            ),
            nn.Sequential(nn.Sequential(
                nn.Conv3d(256, c * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(c * 2),
                nn.SiLU(inplace=True),
            )),
        ])

        self.stage2 = nn.Sequential(
            StageModule(stage=2, output_branches=2, c=c),
            StageModule(stage=2, output_branches=2, c=c),
        )

        self.transition2 = nn.ModuleList([
            nn.Sequential(),
            nn.Sequential(),
            nn.Sequential(nn.Sequential(
                nn.Conv3d(c * 2, c * 4, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(c * 4),
                nn.SiLU(inplace=True),
            )),
        ])

        self.stage3 = nn.Sequential(
            StageModule(stage=3, output_branches=3, c=c),
            StageModule(stage=3, output_branches=3, c=c),
            StageModule(stage=3, output_branches=3, c=c),
            StageModule(stage=3, output_branches=3, c=c),
            StageModule(stage=3, output_branches=3, c=c),
            StageModule(stage=3, output_branches=2, c=c),
        )

        self.dec_block1 = nn.Sequential(
            nn.ConvTranspose3d(c + c * 2, c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(c),
            nn.SiLU(inplace=True),
            BasicBlock(c, c),
        )
        self.dec_block2 = nn.Sequential(
            nn.ConvTranspose3d(c + 32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.SiLU(inplace=True),
            BasicBlock(16, 16),
        )
        self.dec_out = nn.Sequential(
            nn.Conv3d(16 + 16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.SiLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.SiLU(inplace=True),
            nn.Conv3d(16, out_channels, kernel_size=3, padding=1),
        )

    def set_upsample_chunk_size(self, chunk_size: Optional[int]):
        for m in self.modules():
            if isinstance(m, StageModule):
                m.set_upsample_chunk_size(chunk_size)
            elif isinstance(m, NearestUpsample3d):
                m.chunk_size = chunk_size

    def _encode_backbone(self, skip_x2: Tensor, chunk_size: Optional[int] = None) -> list[Tensor]:
        x = self.conv2(skip_x2)
        x = self.act_fun(self.norm2(x))
        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]
        x = self.stage2(x)
        x = [self.transition2[0](x[0]), self.transition2[1](x[1]), self.transition2[2](x[-1])]
        x = self.stage3(x)
        if chunk_size is None:
            x = torch.cat([x[0], F.interpolate(x[1], size=x[0].shape[2:], mode="nearest")], dim=1)
        else:
            x = torch.cat([x[0], interpolate3d_chunked(x[1], chunk_size, 2, mode="nearest", size=x[0].shape[2:])], dim=1)
        return x

    def forward_raw(self, x: Tensor) -> Tensor:
        skip_x1 = self.skip_layer1(x)
        skip_x2 = self.skip_layer2(skip_x1)
        x = self._encode_backbone(skip_x2, chunk_size=None)
        x = self.dec_block1(x)
        x = torch.cat([x, skip_x2], dim=1)
        x = self.dec_block2(x)
        x = torch.cat([x, skip_x1], dim=1)
        return self.dec_out(x)

    @torch.no_grad()
    def _skip1_region(self, inp: Tensor, patch: Tuple[int, int, int, int, int, int], halo: int = 2) -> Tensor:
        """Recompute skip_layer1 only for a requested full-resolution region."""
        b, c, d, h, w = inp.shape
        pd0, ph0, pw0, pd1, ph1, pw1 = patch

        qd0, qh0, qw0 = max(0, pd0 - halo), max(0, ph0 - halo), max(0, pw0 - halo)
        qd1, qh1, qw1 = min(d, pd1 + halo), min(h, ph1 + halo), min(w, pw1 + halo)
        pad_d0 = max(0, halo - (pd0 - qd0))
        pad_h0 = max(0, halo - (ph0 - qh0))
        pad_w0 = max(0, halo - (pw0 - qw0))
        pad_d1 = max(0, halo - (qd1 - pd1))
        pad_h1 = max(0, halo - (qh1 - ph1))
        pad_w1 = max(0, halo - (qw1 - pw1))
        padlist = [pad_w0, pad_w1, pad_h0, pad_h1, pad_d0, pad_d1]

        q = inp[:, :, qd0:qd1, qh0:qh1, qw0:qw1]
        if sum(padlist) > 0:
            q = F.pad(q, padlist, mode="constant", value=0.0)

        # Manually run skip_layer1 and zero the artificial halo after each local block.
        y = self.skip_layer1[0](q)
        y = self.skip_layer1[1](y)
        y = self.skip_layer1[2](y)
        y = set_pad_as_zero(y, padlist)
        y = self.skip_layer1[3](y)
        y = self.skip_layer1[4](y)
        y = self.skip_layer1[5](y)
        y = set_pad_as_zero(y, padlist)

        rd0 = pd0 - qd0 + pad_d0
        rh0 = ph0 - qh0 + pad_h0
        rw0 = pw0 - qw0 + pad_w0
        rd1 = rd0 + (pd1 - pd0)
        rh1 = rh0 + (ph1 - ph0)
        rw1 = rw0 + (pw1 - pw0)
        return y[:, :, rd0:rd1, rh0:rh1, rw0:rw1]


    @torch.no_grad()
    def _skip2_from_input_chunked(self, inp: Tensor, chunk_size: int) -> Tensor:
        """Compute skip_layer2 without ever materializing the full-resolution skip_x1.

        The output is identical to ``self.skip_layer2(self.skip_layer1(inp))`` up to
        floating-point roundoff. It tiles over the half-resolution output grid, recomputes
        the required full-resolution skip_layer1 context for each tile, and crops away
        regions influenced by artificial patch boundaries.
        """
        assert not self.training, "_skip2_from_input_chunked is only valid in eval mode."
        b, _, d, h, w = inp.shape
        d2, h2, w2 = d // 2, h // 2, w // 2
        out = torch.empty((b, 32, d2, h2, w2), device=inp.device, dtype=inp.dtype)

        # A conservative halo in the half-resolution grid. This covers:
        # skip_layer2 conv(stride=2, k=3) + skip_layer2 conv(stride=1, k=3)
        # and the two stride-1 convolutions in skip_layer1.
        half_halo = 6
        nd, nh, nw = _ceil_div(d2, chunk_size), _ceil_div(h2, chunk_size), _ceil_div(w2, chunk_size)

        for ii, jj, kk in itertools.product(range(nd), range(nh), range(nw)):
            hd0, hh0, hw0 = ii * chunk_size, jj * chunk_size, kk * chunk_size
            hd1, hh1, hw1 = min(hd0 + chunk_size, d2), min(hh0 + chunk_size, h2), min(hw0 + chunk_size, w2)

            eh0, eh1 = max(0, hd0 - half_halo), min(d2, hd1 + half_halo)
            ew0, ew1 = max(0, hh0 - half_halo), min(h2, hh1 + half_halo)
            ev0, ev1 = max(0, hw0 - half_halo), min(w2, hw1 + half_halo)

            # Convert the expanded half-resolution region to a full-resolution input patch.
            # Make starts even so the stride-2 grid remains aligned with the global grid.
            qd0 = max(0, 2 * eh0 - 6)
            qh0 = max(0, 2 * ew0 - 6)
            qw0 = max(0, 2 * ev0 - 6)
            qd0 -= qd0 % 2
            qh0 -= qh0 % 2
            qw0 -= qw0 % 2
            qd1 = min(d, 2 * eh1 + 6)
            qh1 = min(h, 2 * ew1 + 6)
            qw1 = min(w, 2 * ev1 + 6)

            patch = inp[:, :, qd0:qd1, qh0:qh1, qw0:qw1]
            y = self.skip_layer1(patch)
            y = self.skip_layer2(y)

            # Since q*0 are even, the local half-resolution origin maps to q*0 // 2.
            od0, oh0, ow0 = qd0 // 2, qh0 // 2, qw0 // 2
            cd0, ch0, cw0 = hd0 - od0, hh0 - oh0, hw0 - ow0
            cd1, ch1, cw1 = cd0 + (hd1 - hd0), ch0 + (hh1 - hh0), cw0 + (hw1 - hw0)
            out[:, :, hd0:hd1, hh0:hh1, hw0:hw1] = y[:, :, cd0:cd1, ch0:ch1, cw0:cw1]
        return out

    @torch.no_grad()
    def _conv_transpose3d_region(
        self,
        x: Tensor,
        deconv: nn.ConvTranspose3d,
        out_region: Tuple[int, int, int, int, int, int],
        safety: int = 3,
    ) -> Tensor:
        """Return ``deconv(x)`` cropped to a requested full-resolution output region.

        This avoids materializing the full transposed-convolution output. The helper is
        specialized for the decoder convolutions used here, where stride=2 and output
        size is exactly twice the input size along each spatial dimension.
        """
        assert not deconv.training, "_conv_transpose3d_region is only valid in eval mode."
        stride = _to_3tuple(deconv.stride)
        assert stride == (2, 2, 2), "This helper is specialized for stride-2 decoder deconvs."
        fd0, fh0, fw0, fd1, fh1, fw1 = out_region
        _, _, d, h, w = x.shape

        id0 = max(0, fd0 // 2 - safety)
        ih0 = max(0, fh0 // 2 - safety)
        iw0 = max(0, fw0 // 2 - safety)
        id1 = min(d, _ceil_div(fd1, 2) + safety)
        ih1 = min(h, _ceil_div(fh1, 2) + safety)
        iw1 = min(w, _ceil_div(fw1, 2) + safety)

        patch = deconv(x[:, :, id0:id1, ih0:ih1, iw0:iw1])
        rd0, rh0, rw0 = fd0 - id0 * 2, fh0 - ih0 * 2, fw0 - iw0 * 2
        rd1, rh1, rw1 = fd1 - id0 * 2, fh1 - ih0 * 2, fw1 - iw0 * 2
        return patch[:, :, rd0:rd1, rh0:rh1, rw0:rw1]

    def _run_dec_block2_post(self, patch: Tensor) -> Tensor:
        """Run BN + SiLU + BasicBlock after the second transposed convolution."""
        y = self.dec_block2[1](patch)
        y = self.dec_block2[2](y)
        y = self.dec_block2[3](y)
        return y


    def _run_dec_out_padded(self, patch: Tensor, padlist: Sequence[int]) -> Tensor:
        """Run dec_out on a padded patch and keep artificial halos zero between convs."""
        y = self.dec_out[0](patch)
        y = self.dec_out[1](y)
        y = self.dec_out[2](y)
        y = set_pad_as_zero(y, padlist)
        y = self.dec_out[3](y)
        y = self.dec_out[4](y)
        y = self.dec_out[5](y)
        y = set_pad_as_zero(y, padlist)
        y = self.dec_out[6](y)
        return y

    @torch.no_grad()
    def _dec_out_chunked(
        self,
        x: Tensor,
        skip_x1: Optional[Tensor],
        inp: Optional[Tensor],
        chunk_size: int,
        halo: int = 3,
    ) -> Tensor:
        """Fused and chunked final decoder.

        This avoids materializing `torch.cat([x, skip_x1], dim=1)` and the high-resolution
        intermediate feature maps inside `dec_out`. If `skip_x1 is None`, it is recomputed on demand.
        """
        assert not self.training, "_dec_out_chunked is only valid in eval mode."
        if skip_x1 is None:
            assert inp is not None, "Input tensor is required for on-demand skip recomputation."
        b, _, d, h, w = x.shape
        out = torch.empty((b, self.out_channels, d, h, w), device=x.device, dtype=x.dtype)
        nd, nh, nw = _ceil_div(d, chunk_size), _ceil_div(h, chunk_size), _ceil_div(w, chunk_size)

        for ii, jj, kk in itertools.product(range(nd), range(nh), range(nw)):
            (vd0, vh0, vw0, vd1, vh1, vw1), (pd0, ph0, pw0, pd1, ph1, pw1), (rd0, rh0, rw0, rd1, rh1, rw1), padlist = _tile_index_3d(
                ii, jj, kk, chunk_size, (d, h, w), halo
            )
            x_patch = x[:, :, pd0:pd1, ph0:ph1, pw0:pw1]
            if skip_x1 is None:
                skip_patch = self._skip1_region(inp, (pd0, ph0, pw0, pd1, ph1, pw1), halo=2)
            else:
                skip_patch = skip_x1[:, :, pd0:pd1, ph0:ph1, pw0:pw1]
            patch = torch.cat([x_patch, skip_patch], dim=1)
            if sum(padlist) > 0:
                patch = F.pad(patch, padlist, mode="constant", value=0.0)
            patch = self._run_dec_out_padded(patch, padlist)
            out[:, :, vd0:vd1, vh0:vh1, vw0:vw1] = patch[:, :, rd0:rd1, rh0:rh1, rw0:rw1]
        return out


    @torch.no_grad()
    def _decode_tail_rank4(self, z_half: Tensor, inp: Tensor, chunk_size: int) -> Tensor:
        """Lowest-memory tail decoder.

        This fuses the following path into tiled execution:

            dec_block2[0] -> dec_block2[1:] -> cat(skip_x1) -> dec_out

        It avoids storing both the full-resolution 16-channel ``dec_block2`` output and
        the full-resolution 16-channel ``skip_x1`` feature. Only a small tile, its halo,
        and the final output tensor are kept on GPU.
        """
        assert not self.training, "_decode_tail_rank4 is only valid in eval mode."
        b, _, dh, hh, wh = z_half.shape
        d, h, w = dh * 2, hh * 2, wh * 2
        out = torch.empty((b, self.out_channels, d, h, w), device=z_half.device, dtype=z_half.dtype)

        dec_out_halo = 3      # three 3x3x3 convolutions in dec_out
        basic_halo = 2        # two 3x3x3 convolutions in BasicBlock
        total_halo = dec_out_halo + basic_halo

        nd, nh, nw = _ceil_div(d, chunk_size), _ceil_div(h, chunk_size), _ceil_div(w, chunk_size)
        for ii, jj, kk in itertools.product(range(nd), range(nh), range(nw)):
            # Final output valid region and the real patch needed by dec_out.
            (vd0, vh0, vw0, vd1, vh1, vw1), dec_patch, dec_crop, dec_padlist = _tile_index_3d(
                ii, jj, kk, chunk_size, (d, h, w), dec_out_halo
            )
            dpd0, dph0, dpw0, dpd1, dph1, dpw1 = dec_patch

            # dec_block2 post-processing must produce the dec_out patch. Therefore its
            # input, i.e. the deconv2 output, needs an additional BasicBlock halo.
            pre_pd0 = max(0, dpd0 - basic_halo)
            pre_ph0 = max(0, dph0 - basic_halo)
            pre_pw0 = max(0, dpw0 - basic_halo)
            pre_pd1 = min(d, dpd1 + basic_halo)
            pre_ph1 = min(h, dph1 + basic_halo)
            pre_pw1 = min(w, dpw1 + basic_halo)
            pre_region = (pre_pd0, pre_ph0, pre_pw0, pre_pd1, pre_ph1, pre_pw1)

            deconv_pre = self._conv_transpose3d_region(z_half, self.dec_block2[0], pre_region, safety=3)
            z2_pre = self._run_dec_block2_post(deconv_pre)

            # Crop the dec_block2 result to the real dec_out patch.
            c0, c1 = dpd0 - pre_pd0, dpd1 - pre_pd0
            c2, c3 = dph0 - pre_ph0, dph1 - pre_ph0
            c4, c5 = dpw0 - pre_pw0, dpw1 - pre_pw0
            z2_patch = z2_pre[:, :, c0:c1, c2:c3, c4:c5]

            skip_patch = self._skip1_region(inp, dec_patch, halo=2)
            patch = torch.cat([z2_patch, skip_patch], dim=1)
            if sum(dec_padlist) > 0:
                patch = F.pad(patch, dec_padlist, mode="constant", value=0.0)
            patch = self._run_dec_out_padded(patch, dec_padlist)
            rd0, rh0, rw0, rd1, rh1, rw1 = dec_crop
            out[:, :, vd0:vd1, vh0:vh1, vw0:vw1] = patch[:, :, rd0:rd1, rh0:rh1, rw0:rw1]
        return out


    @torch.no_grad()
    def forward_efficient(self, x: Tensor, rank: int = 1, chunk_size: int = 64) -> Tensor:
        """Inference-only optimized forward.

        rank = 1: chunk high-risk operators/interpolations but keep the normal final decoder.
        rank = 2: additionally fuse and chunk the final output decoder.
        rank = 3: additionally discard and recompute the full-resolution skip feature at output.
        rank = 4: lowest-memory path. Do not materialize skip_x1 and do not materialize
                  the full-resolution dec_block2 feature.
        """
        assert not self.training, "Use model.eval() before rank>0 inference."
        assert rank in (1, 2, 3, 4)

        old_chunk = []
        for m in self.modules():
            if isinstance(m, NearestUpsample3d):
                old_chunk.append((m, m.chunk_size))
                m.chunk_size = chunk_size

        try:
            if rank >= 4:
                # True on-demand path: skip_x1 is never materialized as a full-volume tensor.
                skip_x1_for_output = None
                skip_x2 = self._skip2_from_input_chunked(x, chunk_size=chunk_size)
            else:
                skip_x1 = self.skip_layer1(x)
                skip_x2 = self.skip_layer2(skip_x1)
                if rank >= 3:
                    skip_x1_for_output = None
                    del skip_x1
                else:
                    skip_x1_for_output = skip_x1

            z = self._encode_backbone(skip_x2, chunk_size=chunk_size)

            z = conv_transpose3d_chunked(z, self.dec_block1[0], block_size=chunk_size)
            z = self.dec_block1[1:](z)
            z = torch.cat([z, skip_x2], dim=1)

            if rank >= 4:
                return self._decode_tail_rank4(z, x, chunk_size=chunk_size)

            z = conv_transpose3d_chunked(z, self.dec_block2[0], block_size=chunk_size)
            z = self.dec_block2[1:](z)

            if rank == 1:
                z = torch.cat([z, skip_x1_for_output], dim=1)
                return self.dec_out(z)
            return self._dec_out_chunked(z, skip_x1_for_output, x if rank >= 3 else None, chunk_size=chunk_size)
        finally:
            for m, old in old_chunk:
                m.chunk_size = old

    def forward(self, x: Tensor, rank: int = 4, chunk_size: int = 64) -> Tensor:
        if rank == 0:
            return self.forward_raw(x)
        return self.forward_efficient(x, rank=rank, chunk_size=chunk_size)


if __name__ == "__main__":
    # Keep this self-check small. Large-volume tests should be run on GPU with model.eval().
    torch.set_num_threads(1)
    model = HRNet(c=1).eval()
    inp = torch.randn(1, 1, 8, 8, 8)
    with torch.no_grad():
        out0 = model(inp, rank=0)
        out4 = model(inp, rank=4, chunk_size=8)
    print("input :", tuple(inp.shape))
    print("output:", tuple(out0.shape))
    print("rank4 max abs error:", (out0 - out4).abs().max().item())
