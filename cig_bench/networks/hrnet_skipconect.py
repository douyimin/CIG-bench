import torch
from torch import nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm3d(planes * self.expansion)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.BatchNorm3d(planes)
        self.act_fun = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm3d(planes)
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
    def __init__(self, stage, output_branches, c):
        super(StageModule, self).__init__()
        self.stage = stage
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = c * (2 ** i)
            branch = nn.Sequential(
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w),
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
                        nn.BatchNorm3d(c * (2 ** i)),
                        nn.Upsample(scale_factor=(2.0 ** (j - i)), mode='nearest'),
                    ))
                elif i > j:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv3d(c * (2 ** j), c * (2 ** j), kernel_size=(3, 3, 3), stride=(2, 2, 2),
                                      padding=(1, 1, 1),
                                      bias=False),
                            nn.BatchNorm3d(c * (2 ** j)),
                            nn.SiLU(inplace=True),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv3d(c * (2 ** j), c * (2 ** i), kernel_size=(3, 3, 3), stride=(2, 2, 2),
                                  padding=(1, 1, 1),
                                  bias=False),
                        nn.BatchNorm3d(c * (2 ** i)),
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
    def __init__(self, c=32):
        super(HRNet, self).__init__()

        # ===== UNet 风格的 skip 分支 (替代原 stem 的 conv1) =====
        # skip_layer1: 全分辨率, 1 -> 16 通道
        self.skip_layer1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.SiLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.SiLU(inplace=True),
        )
        # skip_layer2: 1/2 分辨率, 16 -> 64 通道 (用 stride=2 替代原 conv1 的下采样)
        self.skip_layer2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.SiLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.SiLU(inplace=True),
        )

        # Input (stem net) - conv1 已被 skip_layer1+skip_layer2 替代, 这里只保留 conv2
        # self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
        # self.norm1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
        self.norm2 = nn.BatchNorm3d(64)
        self.act_fun = nn.SiLU(inplace=True)

        # Stage 1 (layer1)      - First group of bottleneck (resnet) modules
        downsample = nn.Sequential(
            nn.Conv3d(64, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
            nn.BatchNorm3d(256),
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
        )

        # Fusion layer 1 (transition1)      - Creation of the first two branches (one full and one half resolution)
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(256, c, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
                nn.BatchNorm3d(c),
                nn.SiLU(inplace=True),
            ),
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv3d(256, c * (2 ** 1), kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False),
                nn.BatchNorm3d(c * (2 ** 1)),
                nn.SiLU(inplace=True),
            )),
        ])

        # Stage 2 (stage2)      - Second module with 1 group of bottleneck (resnet) modules. This has 2 branches
        self.stage2 = nn.Sequential(
            StageModule(stage=2, output_branches=2, c=c),
            StageModule(stage=2, output_branches=2, c=c),
        )

        # Fusion layer 2 (transition2)      - Creation of the third branch (1/4 resolution)
        self.transition2 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv3d(c * (2 ** 1), c * (2 ** 2), kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                          bias=False),
                nn.BatchNorm3d(c * (2 ** 2)),
                nn.SiLU(inplace=True),
            )),  # ToDo Why the new branch derives from the "upper" branch only?
        ])

        # Stage 3 (stage3)      - Third module with 4 groups of bottleneck (resnet) modules. This has 3 branches
        self.stage3 = nn.Sequential(
            StageModule(stage=3, output_branches=3, c=c),
            StageModule(stage=3, output_branches=3, c=c),
            StageModule(stage=3, output_branches=3, c=c),
            StageModule(stage=3, output_branches=3, c=c),
            StageModule(stage=3, output_branches=3, c=c),
            StageModule(stage=3, output_branches=2, c=c),
        )

        # ----- 解码器 -----
        # 主干输出 cat 后通道数 = c + c*2 = 3c (1/4 分辨率)
        # 第一次反卷积上采样: (c+c*2) -> c, 1/4 -> 1/2 -> cat skip_x2 (64ch, 1/2)
        # 第二次反卷积上采样: (c+64) -> 16, 1/2 -> 全分辨率 -> cat skip_x1 (16ch, 全)
        self.dec_block1 = nn.Sequential(
            nn.ConvTranspose3d(c + c * 2, c, kernel_size=4, stride=2,padding=1, bias=False),
            nn.BatchNorm3d(c),
            nn.SiLU(inplace=True),
            BasicBlock(c, c)
        )

        self.dec_block2 = nn.Sequential(
            nn.ConvTranspose3d(c + 32, 16, kernel_size=4, stride=2,padding=1, bias=False),  # cat skip_x2 后通道 = c + 64
            nn.BatchNorm3d(16),
            nn.SiLU(inplace=True),
            BasicBlock(16, 16)
        )

        self.dec_out = nn.Sequential(
            nn.Conv3d(16 + 16, 16, kernel_size=3, padding=1),  # cat skip_x1 后通道 = 16 + 16
            nn.BatchNorm3d(16),
            nn.SiLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),  # cat skip_x1 后通道 = 16 + 16
            nn.BatchNorm3d(16),
            nn.SiLU(inplace=True),
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
        )

        # 兼容旧名 (避免外部引用断裂)
        # self.reg_decoder = nn.ModuleList([self.dec_block1, self.dec_block2, self.dec_out])

    def forward(self, x):
        # ===== UNet 风格 skip 分支 (替代原 stem 的 conv1, 同时保存供解码器使用) =====
        skip_x1 = self.skip_layer1(x)  # 全分辨率, 16 通道
        skip_x2 = self.skip_layer2(skip_x1)  # 1/2 分辨率, 64 通道  (注意: 这里调用 skip_layer2 而非 skip_layer1)

        # ===== 主干: 从 skip_x2 接入原来的 conv2 =====
        x = self.conv2(skip_x2)
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

        # ===== 解码器: 在每次上采样后 cat 对应分辨率的 skip 特征 =====
        x = self.dec_block1(x)  # -> 1/2 分辨率
        x = torch.cat([x, skip_x2], dim=1)  # cat 1/2 分辨率 skip 特征

        x = self.dec_block2(x)  # -> 全分辨率
        x = torch.cat([x, skip_x1], dim=1)  # cat 全分辨率 skip 特征

        x = self.dec_out(x)
        return x


if __name__ == '__main__':
    # CPU 形状自检 (输入通道改为1, 与 skip_layer1 的 in_channels=1 一致)
    model = HRNet(c=32)
    inp = torch.ones(1, 1, 128, 128, 128)
    y = model(inp)
    print('input :', inp.shape)
    print('output:', y.shape)
    print(torch.min(y).item(), torch.mean(y).item(), torch.max(y).item())
