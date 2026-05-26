import torch
from torch import nn
from functools import partial
from math import gcd
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Norm layer factory
# ---------------------------------------------------------------------------
def get_norm_layer(norm_type='gn', gn_groups=16):
    """
    Return a callable `norm(num_channels)` that builds a normalization layer.

    Args:
        norm_type (str): one of {'bn', 'in', 'gn'} (case-insensitive).
            - 'bn': nn.BatchNorm3d
            - 'in': nn.InstanceNorm3d  (affine=True, track_running_stats=False)
            - 'gn': nn.GroupNorm with `gn_groups` groups.
                    Falls back to gcd(num_channels, gn_groups) if the channel
                    count is not divisible by `gn_groups`, so it never crashes.
        gn_groups (int): number of groups for GroupNorm. Default 16.

    Returns:
        Callable[[int], nn.Module]
    """
    nt = norm_type.lower()
    if nt == 'bn':
        return lambda num_channels: nn.BatchNorm3d(num_channels)
    elif nt == 'in':
        # affine=True so it has learnable params like BN/GN;
        # track_running_stats=False is the typical IN setting.
        return lambda num_channels: nn.InstanceNorm3d(
            num_channels, affine=True, track_running_stats=False
        )
    elif nt == 'gn':
        def _gn(num_channels):
            g = gn_groups if num_channels % gn_groups == 0 else gcd(num_channels, gn_groups)
            g = max(g, 1)
            return nn.GroupNorm(g, num_channels)
        return _gn
    else:
        raise ValueError(
            f"Unknown norm_type '{norm_type}'. Expected one of 'bn', 'in', 'gn'."
        )


class PixelShuffle3d(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3
        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale
        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)
        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return output.view(batch_size, nOut, out_depth, out_height, out_width)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = get_norm_layer('gn', 16)
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.norm1 = norm_layer(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = norm_layer(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.norm3 = norm_layer(planes * self.expansion)
        self.act_fun = nn.SiLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act_fun(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act_fun(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act_fun(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = get_norm_layer('gn', 16)
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = norm_layer(planes)
        self.act_fun = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act_fun(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act_fun(out)

        return out


class StageModule(nn.Module):
    def __init__(self, stage, output_branches, c, norm_layer=None):
        super(StageModule, self).__init__()
        if norm_layer is None:
            norm_layer = get_norm_layer('gn', 16)
        self.stage = stage
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = c * (2 ** i)
            branch = nn.Sequential(
                BasicBlock(w, w, norm_layer=norm_layer),
                BasicBlock(w, w, norm_layer=norm_layer),
                BasicBlock(w, w, norm_layer=norm_layer),
                BasicBlock(w, w, norm_layer=norm_layer),
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        # for each output_branches (i.e. each branch in all cases but the very last one)
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.stage):  # for each branch
                if i == j:
                    self.fuse_layers[-1].append(nn.Sequential())  # Used in place of "None" because it is callable
                elif i < j:
                    self.fuse_layers[-1].append(nn.Sequential(
                        nn.Conv3d(c * (2 ** j), c * (2 ** i), kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
                        norm_layer(c * (2 ** i)),
                        nn.Upsample(scale_factor=(2.0 ** (j - i)), mode='nearest'),
                    ))
                elif i > j:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv3d(c * (2 ** j), c * (2 ** j), kernel_size=(3, 3, 3), stride=(2, 2, 2),
                                      padding=(1, 1, 1),
                                      bias=False),
                            norm_layer(c * (2 ** j)),
                            nn.SiLU(inplace=True),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv3d(c * (2 ** j), c * (2 ** i), kernel_size=(3, 3, 3), stride=(2, 2, 2),
                                  padding=(1, 1, 1),
                                  bias=False),
                        norm_layer(c * (2 ** i)),
                    ))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.act_fun = nn.SiLU(inplace=True)

    def forward(self, x):
        assert len(self.branches) == len(x)

        x = [branch(b) for branch, b in zip(self.branches, x)]

        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(0, len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](x[j])

        for i in range(len(x_fused)):
            x_fused[i] = self.act_fun(x_fused[i])

        return x_fused


class HRNet(nn.Module):
    def __init__(self, c=48, norm_type='gn', gn_groups=16):
        """
        Args:
            c (int): base channel width.
            norm_type (str): 'bn' | 'in' | 'gn' (default 'gn').
            gn_groups (int): number of groups for GroupNorm (default 16).
                             Ignored when norm_type is not 'gn'.
        """
        super(HRNet, self).__init__()

        norm_layer = get_norm_layer(norm_type, gn_groups)
        self.norm_type = norm_type
        self.gn_groups = gn_groups

        # Input (stem net)
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
        self.norm1 = norm_layer(64)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
        self.norm2 = norm_layer(64)
        self.act_fun = nn.SiLU(inplace=True)

        # Stage 1 (layer1)      - First group of bottleneck (resnet) modules
        downsample = nn.Sequential(
            nn.Conv3d(64, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
            norm_layer(256),
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample, norm_layer=norm_layer),
            Bottleneck(256, 64, norm_layer=norm_layer),
            Bottleneck(256, 64, norm_layer=norm_layer),
            Bottleneck(256, 64, norm_layer=norm_layer),
        )

        # Fusion layer 1 (transition1)      - Creation of the first two branches (one full and one half resolution)
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(256, c, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
                norm_layer(c),
                nn.SiLU(inplace=True),
            ),
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv3d(256, c * (2 ** 1), kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False),
                norm_layer(c * (2 ** 1)),
                nn.SiLU(inplace=True),
            )),
        ])

        # Stage 2 (stage2)      - Second module with 1 group of bottleneck (resnet) modules. This has 2 branches
        self.stage2 = nn.Sequential(
            StageModule(stage=2, output_branches=2, c=c, norm_layer=norm_layer),
            StageModule(stage=2, output_branches=2, c=c, norm_layer=norm_layer),
        )

        # Fusion layer 2 (transition2)      - Creation of the third branch (1/4 resolution)
        self.transition2 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv3d(c * (2 ** 1), c * (2 ** 2), kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                          bias=False),
                norm_layer(c * (2 ** 2)),
                nn.SiLU(inplace=True),
            )),  # ToDo Why the new branch derives from the "upper" branch only?
        ])

        # Stage 3 (stage3)      - Third module with 4 groups of bottleneck (resnet) modules. This has 3 branches
        self.stage3 = nn.Sequential(
            StageModule(stage=3, output_branches=3, c=c, norm_layer=norm_layer),
            StageModule(stage=3, output_branches=3, c=c, norm_layer=norm_layer),
            StageModule(stage=3, output_branches=3, c=c, norm_layer=norm_layer),
            StageModule(stage=3, output_branches=3, c=c, norm_layer=norm_layer),
            StageModule(stage=3, output_branches=3, c=c, norm_layer=norm_layer),
            StageModule(stage=3, output_branches=2, c=c, norm_layer=norm_layer),
        )

        self.decoder = nn.Sequential(
            nn.Conv3d(c + c * 2, c * 8, kernel_size=3, padding=1),
            norm_layer(c * 8),
            nn.SiLU(inplace=True),
            PixelShuffle3d(2),
            nn.Conv3d(c, 16 * 8, kernel_size=3, padding=1),
            norm_layer(16 * 8),
            nn.SiLU(inplace=True),
            PixelShuffle3d(2),
            nn.Conv3d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act_fun(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act_fun(x)

        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]  # Since now, x is a list (# == nof branches)

        x = self.stage2(x)
        # x = [trans(x[-1]) for trans in self.transition2]    # New branch derives from the "upper" branch only
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage3(x)
        x = torch.cat([x[0], F.interpolate(x[1], scale_factor=2)], dim=1)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    # Examples:
    #   model = HRNet(c=32, norm_type='bn')
    #   model = HRNet(c=32, norm_type='in')
    #   model = HRNet(c=32, norm_type='gn', gn_groups=8)
    model = HRNet(c=32, norm_type='gn', gn_groups=16).cuda()

    y = model(torch.ones(1, 3, 384, 288, 288).cuda())
    print(y.shape)
    print(torch.min(y).item(), torch.mean(y).item(), torch.max(y).item())
