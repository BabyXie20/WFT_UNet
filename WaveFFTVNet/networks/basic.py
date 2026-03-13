import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out):
        super().__init__()
        ops = []
        for i in range(n_stages):
            in_ch = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(in_ch, n_filters_out, kernel_size=3, padding=1))
            ops.append(nn.InstanceNorm3d(n_filters_out, affine=True))
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out):
        super().__init__()

        layers = []
        for i in range(n_stages):
            in_ch = n_filters_in if i == 0 else n_filters_out
            layers += [
                nn.Conv3d(in_ch, n_filters_out, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(n_filters_out, affine=True),
            ]
            if i != n_stages - 1:
                layers.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*layers)
        self.proj = (
            nn.Sequential(
                nn.Conv3d(n_filters_in, n_filters_out, kernel_size=1, bias=False),
                nn.InstanceNorm3d(n_filters_out, affine=True),
            )
            if n_filters_in != n_filters_out else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.proj(x))


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(n_filters_in, n_filters_out, kernel_size=stride, padding=0, stride=stride),
            nn.InstanceNorm3d(n_filters_out, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(n_filters_in, n_filters_out, kernel_size=stride, padding=0, stride=stride),
            nn.InstanceNorm3d(n_filters_out, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, n_channels=1, n_filters=16, has_residual=False):
        super().__init__()
        conv = ResidualConvBlock if has_residual else ConvBlock

        stages = [1, 2, 3, 3, 3]
        chs = [n_filters * (2 ** i) for i in range(5)]  # [f, 2f, 4f, 8f, 16f]

        self.blocks = nn.ModuleList(
            [conv(stages[0], n_channels, chs[0])] +
            [conv(stages[i], chs[i], chs[i]) for i in range(1, 5)]
        )
        self.downs = nn.ModuleList([DownsamplingConvBlock(chs[i], chs[i + 1]) for i in range(4)])

    def forward(self, x):
        feats = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            feats.append(x)
            if i < len(self.downs):
                x = self.downs[i](x)
        return feats  # [x1, x2, x3, x4, x5]


class Decoder(nn.Module):
    def __init__(self, n_classes=14, n_filters=16, has_residual=False):
        super().__init__()
        conv = ResidualConvBlock if has_residual else ConvBlock

        chs = [n_filters * (2 ** i) for i in range(5)]  
        dec_stages = [3, 3, 2, 1]  

        self.ups = nn.ModuleList([UpsamplingDeconvBlock(chs[i], chs[i - 1]) for i in range(4, 0, -1)])

        self.blocks = nn.ModuleList([
            conv(dec_stages[0], 2 * chs[3], chs[3]),
            conv(dec_stages[1], 2 * chs[2], chs[2]),
            conv(dec_stages[2], 2 * chs[1], chs[1]),
            conv(dec_stages[3], 2 * chs[0], chs[0]),
        ])

        self.out_conv = nn.Conv3d(chs[0], n_classes, kernel_size=1)

    def forward(self, feats):
        x = feats[-1]  
        for i, (up, blk) in enumerate(zip(self.ups, self.blocks)):
            x = up(x)
            skip = feats[-2 - i]  
            x = torch.cat([x, skip], dim=1)
            x = blk(x)
        return self.out_conv(x)


class VNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=14, n_filters=16, has_residual=False):
        super().__init__()
        self.encoder = Encoder(n_channels=n_channels, n_filters=n_filters, has_residual=has_residual)
        self.decoder = Decoder(n_classes=n_classes, n_filters=n_filters, has_residual=has_residual)

    def forward(self, x):
        return self.decoder(self.encoder(x))