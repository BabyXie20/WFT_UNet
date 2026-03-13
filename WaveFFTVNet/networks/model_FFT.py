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


def _valid_num_groups(num_channels: int, max_groups: int = 8) -> int:
    max_groups = min(max_groups, num_channels)
    for g in range(max_groups, 0, -1):
        if num_channels % g == 0:
            return g
    return 1


class MagnitudeOnlyNorm3D(nn.Module):
    def __init__(self, channels: int, max_gn_groups: int = 8, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        complex_channels = channels // 2
        self.mag_norm = nn.GroupNorm(
            _valid_num_groups(complex_channels, max_gn_groups),
            complex_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = torch.chunk(x, 2, dim=1)
        mag = torch.sqrt(xr * xr + xi * xi + self.eps)
        gain = 1.0 + torch.tanh(self.mag_norm(mag))
        return torch.cat([xr * gain, xi * gain], dim=1)


class FreqConditionalPE3D(nn.Module):
    def __init__(self, channels: int, max_gn_groups: int = 8):
        super().__init__()
        self.norm = MagnitudeOnlyNorm3D(channels, max_gn_groups)
        self.dw_conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dw_conv(self.norm(x))


class FreqDynamicConv3D(nn.Module):
    def __init__(
        self,
        channels: int,
        num_experts: int = 4,
        kernel_size: int = 3,
        max_gn_groups: int = 8,
    ):
        super().__init__()
        self.channels = channels
        self.num_experts = num_experts
        self.complex_channels = channels // 2
        self.norm = MagnitudeOnlyNorm3D(channels, max_gn_groups)

        pad = kernel_size // 2
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=pad,
                    groups=channels,
                    bias=True,
                ),
                nn.Conv3d(
                    channels,
                    channels,
                    kernel_size=1,
                    groups=1,  
                    bias=True,
                ),
            )
            for _ in range(num_experts)
        ])
        self.band_gate = nn.Conv3d(
            self.complex_channels * 3,
            num_experts * 3,
            kernel_size=1,
            bias=True,
        )

    @staticmethod
    def _band_masks(
        d: int,
        h: int,
        wf: int,
        orig_w: int,
        device,
        dtype,
    ) -> torch.Tensor:
        fz = torch.fft.fftfreq(d, device=device, dtype=dtype)
        fy = torch.fft.fftfreq(h, device=device, dtype=dtype)
        fx = torch.fft.rfftfreq(orig_w, device=device, dtype=dtype)  

        zz, yy, xx = torch.meshgrid(fz, fy, fx, indexing="ij")
        r = torch.sqrt(zz * zz + yy * yy + xx * xx)
        r = r / (r.max() + 1e-6)

        low = (r < 1.0 / 3.0).to(dtype)
        mid = ((r >= 1.0 / 3.0) & (r < 2.0 / 3.0)).to(dtype)
        high = (r >= 2.0 / 3.0).to(dtype)

        return torch.stack([low, mid, high], dim=0).unsqueeze(0).unsqueeze(2)

    def forward(self, x: torch.Tensor, orig_w: int) -> torch.Tensor:
        x_n = self.norm(x)
        b, c2, d, h, wf = x_n.shape

        expert_outs = torch.stack([expert(x_n) for expert in self.experts], dim=1)

        xr, xi = torch.chunk(x_n, 2, dim=1)  # [B, C, D, H, Wf]
        mag = torch.sqrt(xr * xr + xi * xi + 1e-6)

        band_masks = self._band_masks(d, h, wf, orig_w, x_n.device, x_n.dtype)  # [1, 3, 1, D, H, Wf]

        masked_mag = mag.unsqueeze(1) * band_masks
        denom = band_masks.sum(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-6)  # [1, 3, 1, 1, 1, 1]

        band_stats = masked_mag.sum(dim=(-3, -2, -1), keepdim=True) / denom

        band_stats = band_stats.reshape(b, self.complex_channels * 3, 1, 1, 1)

        band_logits = self.band_gate(band_stats).view(b, 3, self.num_experts, 1, 1, 1, 1)
        band_weights = torch.softmax(band_logits, dim=2)

        weights = (band_weights * band_masks.unsqueeze(2)).sum(dim=1)

        y = (weights * expert_outs).sum(dim=1)
        return y


class FrequencyBranch(nn.Module):
    def __init__(self, channels: int, num_experts: int = 4, max_gn_groups: int = 8):
        super().__init__()
        self.channels = channels
        self.freq_channels = channels * 2

        self.freq_pe = FreqConditionalPE3D(
            channels=self.freq_channels,
            max_gn_groups=max_gn_groups,
        )

        self.freq_dyn = FreqDynamicConv3D(
            channels=self.freq_channels,
            num_experts=num_experts,
            kernel_size=3,
            max_gn_groups=max_gn_groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        b, c, d, h, w = x.shape

        fft_in = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x

        x_fft = torch.fft.rfftn(fft_in, dim=(-3, -2, -1), norm="ortho")  # [B, C, D, H, Wf]

        xr = x_fft.real
        xi = x_fft.imag
        x_freq = torch.cat([xr, xi], dim=1)  # [B, 2C, D, H, Wf]

        x_freq = self.freq_pe(x_freq)
        x_freq = self.freq_dyn(x_freq, orig_w=w)

        yr, yi = torch.chunk(x_freq, 2, dim=1)
        y_fft = torch.complex(yr.contiguous(), yi.contiguous())

        y = torch.fft.irfftn(y_fft, s=(d, h, w), dim=(-3, -2, -1), norm="ortho")

        y = y.to(x_in.dtype)
        y = y + x_in
        return y


class CAFM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv1_spatial = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1)
        self.conv2_spatial = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1)
        self.fusion = nn.Conv3d(channels * 2, channels, kernel_size=1, stride=1, padding=0)

    def _spatial_attn(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        n = d * h * w
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        sa = torch.cat([avg_out, max_out], dim=1)
        sa = F.relu(self.conv1_spatial(sa))
        sa = torch.sigmoid(self.conv2_spatial(sa))
        return sa.view(b, 1, n)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = f1.shape
        n = d * h * w

        f1_flat = f1.view(b, c, n)
        f2_flat = f2.view(b, c, n)

        f1n = F.normalize(f1_flat, p=2, dim=-1, eps=1e-6)
        f2n = F.normalize(f2_flat, p=2, dim=-1, eps=1e-6)

        cross = torch.matmul(f1n, f2n.transpose(1, 2))              
        attn12 = F.softmax(cross, dim=-1)
        attn21 = F.softmax(cross.transpose(1, 2), dim=-1)

        a1_feat = torch.matmul(attn12, f2_flat).view(b, c, d, h, w)  
        a2_feat = torch.matmul(attn21, f1_flat).view(b, c, d, h, w)

        sa1 = self._spatial_attn(a1_feat)                           
        sa2 = self._spatial_attn(a2_feat)                           

        f1_enh = (f1_flat * (1.0 + sa1)).view(b, c, d, h, w)
        f2_enh = (f2_flat * (1.0 + sa2)).view(b, c, d, h, w)

        return self.fusion(torch.cat([f1_enh, f2_enh], dim=1))
    

class DualDomainBlock(nn.Module):
    def __init__(self, n: int, channels: int, num_experts: int = 4, max_gn_groups: int = 8):
        super().__init__()
        self.fre = FrequencyBranch(channels=channels,num_experts=num_experts, max_gn_groups=max_gn_groups)
        self.spa = ConvBlock(n_stages=n,n_filters_in=channels,n_filters_out=channels)
        self.fuse = CAFM(channels=channels)

    def forward(self, x):
        x_spa = self.spa(x)     # [B, C, D, H, W]
        x_fre = self.fre(x)     # [B, C, D, H, W]
        y = self.fuse(x_spa, x_fre)  # [B, C, D, H, W]
        return y
    

class Encoder(nn.Module):
    def __init__(self, n_channels=1, n_filters=16, has_residual=False):
        super().__init__()
        conv = ResidualConvBlock if has_residual else ConvBlock

        stages = [1, 2, 3, 3, 3]
        chs = [n_filters * (2 ** i) for i in range(5)]  # [16,32,64,128,256]
        self.blk0 = conv(stages[0], n_channels, chs[0])  
        self.down0 = DownsamplingConvBlock(chs[0], chs[1])  # 16->32
        self.blk1 = DualDomainBlock(n=stages[1], channels=chs[1],max_gn_groups=4)

        self.down1 = DownsamplingConvBlock(chs[1], chs[2])  # 32->64
        self.blk2 = DualDomainBlock(n=stages[2], channels=chs[2])

        self.down2 = DownsamplingConvBlock(chs[2], chs[3])  # 64->128
        self.blk3 = DualDomainBlock(n=stages[3], channels=chs[3])

        self.down3 = DownsamplingConvBlock(chs[3], chs[4])  # 128->256
        self.blk4 = conv(stages[4], chs[4], chs[4])         # 256

    def forward(self, x):
        x0 = self.blk0(x)          
        x1 = self.blk1(self.down0(x0))   # (B,32, ...)
        x2 = self.blk2(self.down1(x1))   # (B,64, ...)
        x3 = self.blk3(self.down2(x2))   # (B,128,...)
        x4 = self.blk4(self.down3(x3))   # (B,256,...)

        feats = [x0, x1, x2, x3, x4]
        return feats


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