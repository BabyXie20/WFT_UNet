import torch
from torch import nn
import torch.nn.functional as F
from .DWT_IDWT_layer import DWT_3D, IDWT_3D
from .swin_unetr import window_partition, window_reverse, get_window_size, compute_mask
from monai.networks.layers import trunc_normal_


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


class CrossWindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(7, 7, 7), qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        wd, wh, ww = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * wd - 1) * (2 * wh - 1) * (2 * ww - 1), num_heads)
        )

        coords_d = torch.arange(wd)
        coords_h = torch.arange(wh)
        coords_w = torch.arange(ww)
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        relative_coords[:, :, 0] += wd - 1
        relative_coords[:, :, 1] += wh - 1
        relative_coords[:, :, 2] += ww - 1
        relative_coords[:, :, 0] *= (2 * wh - 1) * (2 * ww - 1)
        relative_coords[:, :, 1] *= (2 * ww - 1)
        relative_position_index = relative_coords.sum(-1)

        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_q, x_kv, mask=None):
        b, n, c = x_q.shape
        h = self.num_heads
        d = c // h

        q = self.q(x_q).reshape(b, n, h, d).permute(0, 2, 1, 3)
        kv = self.kv(x_kv).reshape(b, n, 2, h, d).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        rpb = self.relative_position_bias_table[
            self.relative_position_index[:n, :n].reshape(-1)
        ].reshape(n, n, -1).permute(2, 0, 1).contiguous()
        attn = attn + rpb.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, h, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, h, n, n)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn).to(v.dtype)

        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class CrossShiftWindowAttn3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=(7, 7, 7), shift=True, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(w // 2 for w in window_size) if shift else (0, 0, 0)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = CrossWindowAttention(dim, num_heads, window_size, qkv_bias, attn_drop, proj_drop)

    def forward(self, x_q, x_kv):
        b, c, d, h, w = x_q.shape

        xq = x_q.permute(0, 2, 3, 4, 1).contiguous()
        xkv = x_kv.permute(0, 2, 3, 4, 1).contiguous()

        xq = self.norm_q(xq)
        xkv = self.norm_kv(xkv)

        ws, ss = get_window_size((d, h, w), self.window_size, self.shift_size)

        pad_d1 = (ws[0] - d % ws[0]) % ws[0]
        pad_b = (ws[1] - h % ws[1]) % ws[1]
        pad_r = (ws[2] - w % ws[2]) % ws[2]

        xq = F.pad(xq, (0, 0, 0, pad_r, 0, pad_b, 0, pad_d1))
        xkv = F.pad(xkv, (0, 0, 0, pad_r, 0, pad_b, 0, pad_d1))
        _, dp, hp, wp, _ = xq.shape

        if any(i > 0 for i in ss):
            xq = torch.roll(xq, shifts=(-ss[0], -ss[1], -ss[2]), dims=(1, 2, 3))
            xkv = torch.roll(xkv, shifts=(-ss[0], -ss[1], -ss[2]), dims=(1, 2, 3))
            attn_mask = compute_mask([dp, hp, wp], ws, ss, xq.device)
        else:
            attn_mask = None

        q_win = window_partition(xq, ws)
        kv_win = window_partition(xkv, ws)

        out_win = self.attn(q_win, kv_win, mask=attn_mask)
        out_win = out_win.view(-1, ws[0], ws[1], ws[2], c)
        out = window_reverse(out_win, ws, [b, dp, hp, wp])

        if any(i > 0 for i in ss):
            out = torch.roll(out, shifts=(ss[0], ss[1], ss[2]), dims=(1, 2, 3))

        out = out[:, :d, :h, :w, :].contiguous()
        out = out.permute(0, 4, 1, 2, 3).contiguous()

        return x_q + out


class BandGraphFusion(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, topk: int = 3, attn_drop: float = 0.0, proj_drop: float = 0.0, eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.topk = topk
        self.eps = eps

        self.desc_proj = nn.Linear(4 * channels, channels, bias=True)
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(channels, channels, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Linear(4 * channels, channels),
        )
        self.gate = nn.Linear(channels, channels, bias=True)
        self.fuse = nn.Conv3d(7 * channels, channels, kernel_size=1, bias=True)

    def _node_descriptors(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3, 4))
        var = x.var(dim=(2, 3, 4), unbiased=False)
        std = torch.sqrt(var + self.eps)
        abs_mean = x.abs().mean(dim=(2, 3, 4))
        energy = (x * x).mean(dim=(2, 3, 4))
        desc = torch.cat([mean, std, abs_mean, energy], dim=-1)
        return desc

    def _build_topk_mask(self, raw_desc: torch.Tensor) -> torch.Tensor:
        b, n, _ = raw_desc.shape

        e = F.normalize(raw_desc, p=2, dim=-1, eps=self.eps)
        sim = torch.matmul(e, e.transpose(1, 2))

        eye = torch.eye(n, device=raw_desc.device, dtype=torch.bool).unsqueeze(0)
        sim = sim.masked_fill(eye, float("-inf"))

        k = min(self.topk, n - 1)
        idx = sim.topk(k=k, dim=-1).indices

        mask = torch.full(
            (b, n, n),
            float("-inf"),
            device=raw_desc.device,
            dtype=raw_desc.dtype,
        )
        mask.scatter_(-1, idx, 0.0)

        return mask.unsqueeze(1)

    def forward(self, detail_bands):
        assert isinstance(detail_bands, (list, tuple)) and len(detail_bands) == 7

        b, c, _, _, _ = detail_bands[0].shape
        n = 7

        raw_nodes = torch.stack(
            [self._node_descriptors(x) for x in detail_bands], dim=1
        )

        mask = self._build_topk_mask(raw_nodes)

        nodes0 = self.desc_proj(raw_nodes)
        nodes = self.norm(nodes0)

        qkv = self.qkv(nodes)
        q, k, v = qkv.chunk(3, dim=-1)

        h = self.num_heads
        d = c // h

        q = q.view(b, n, h, d).transpose(1, 2)
        k = k.view(b, n, h, d).transpose(1, 2)
        v = v.view(b, n, h, d).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (d ** -0.5)
        attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, c)
        out = self.proj_drop(self.proj(out))

        nodes = nodes0 + out
        nodes = nodes + self.ffn(self.ffn_norm(nodes))

        g = torch.sigmoid(self.gate(nodes))

        modulated = []
        for i, x in enumerate(detail_bands):
            gi = g[:, i, :].view(b, c, 1, 1, 1)
            modulated.append(x * gi)

        x_cat = torch.cat(modulated, dim=1)
        f_fused = self.fuse(x_cat)

        return f_fused


class WaveletCrossBandBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, topk: int, window_size=(7, 7, 7), shift=True):
        super().__init__()

        self.fusion = BandGraphFusion(
            channels=channels,
            num_heads=num_heads,
            topk=topk,
        )

        self.attn_l = CrossShiftWindowAttn3D(
            dim=channels, num_heads=num_heads, window_size=window_size, shift=shift
        )

        self.attn_h_1H = CrossShiftWindowAttn3D(
            dim=channels, num_heads=num_heads, window_size=window_size, shift=shift
        )
        self.attn_h_2H = CrossShiftWindowAttn3D(
            dim=channels, num_heads=num_heads, window_size=window_size, shift=shift
        )
        self.attn_h_3H = CrossShiftWindowAttn3D(
            dim=channels, num_heads=num_heads, window_size=window_size, shift=shift
        )

        self.group_1H = ["LLH", "LHL", "HLL"]
        self.group_2H = ["LHH", "HLH", "HHL"]
        self.group_3H = ["HHH"]
        self.detail_keys = self.group_1H + self.group_2H + self.group_3H

    def forward(self, bands: dict):
        lll = bands["LLL"]
        details = [bands[k] for k in self.detail_keys]

        fused = self.fusion(details)
        lll_new = self.attn_l(lll, fused)

        out = {"LLL": lll_new}

        for k in self.group_1H:
            out[k] = self.attn_h_1H(bands[k], lll_new)

        for k in self.group_2H:
            out[k] = self.attn_h_2H(bands[k], lll_new)

        for k in self.group_3H:
            out[k] = self.attn_h_3H(bands[k], lll_new)

        return out


_KEYS = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]


def tuple_to_bands(t):
    return {k: v for k, v in zip(_KEYS, t)}


def bands_to_tuple(bands: dict):
    return tuple(bands[k] for k in _KEYS)


class WaveletDWT_Cross_IDWT(nn.Module):
    def __init__(self, wavename: str, channels: int, num_heads: int, topk: int, window_size=(6, 6, 6), shift=True, n_blocks: int = 1):
        super().__init__()
        self.dwt = DWT_3D(wavename)
        self.idwt = IDWT_3D(wavename)

        self.blocks = nn.ModuleList([
            WaveletCrossBandBlock(
                channels=channels,
                num_heads=num_heads,
                topk=topk,
                window_size=window_size,
                shift=shift,
            )
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        bands_tuple = self.dwt(x)
        bands = tuple_to_bands(bands_tuple)

        for blk in self.blocks:
            bands = blk(bands)

        lll, llh, lhl, lhh, hll, hlh, hhl, hhh = bands_to_tuple(bands)
        y = self.idwt(lll, llh, lhl, lhh, hll, hlh, hhl, hhh)

        return y


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
    def __init__(self, channels: int, num_experts: int = 4, kernel_size: int = 3, max_gn_groups: int = 8):
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
    def _band_masks(d: int, h: int, wf: int, orig_w: int, device, dtype) -> torch.Tensor:
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
        b, _, d, h, wf = x_n.shape

        expert_outs = torch.stack([expert(x_n) for expert in self.experts], dim=1)

        xr, xi = torch.chunk(x_n, 2, dim=1)
        mag = torch.sqrt(xr * xr + xi * xi + 1e-6)

        band_masks = self._band_masks(d, h, wf, orig_w, x_n.device, x_n.dtype)
        masked_mag = mag.unsqueeze(1) * band_masks
        denom = band_masks.sum(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-6)
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
        _, _, d, h, w = x.shape

        fft_in = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        x_fft = torch.fft.rfftn(fft_in, dim=(-3, -2, -1), norm="ortho")

        xr = x_fft.real
        xi = x_fft.imag
        x_freq = torch.cat([xr, xi], dim=1)

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
        b, _, d, h, w = x.shape
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


class DualDomainBlockv1(nn.Module):
    def __init__(self, n: int, channels: int, cfg: dict):
        super().__init__()
        self.fre = WaveletDWT_Cross_IDWT(
            wavename="bior2.2",
            channels=channels,
            num_heads=cfg["num_heads"],
            topk=cfg["topk"],
        )
        self.spa = ConvBlock(n_stages=n, n_filters_in=channels, n_filters_out=channels)
        self.fuse = CAFM(channels=channels)

    def forward(self, x):
        x_spa = self.spa(x)
        x_fre = self.fre(x)
        y = self.fuse(x_spa, x_fre)
        return y


class DualDomainBlockV2(nn.Module):
    def __init__(self, n: int, channels: int, cfg: dict):
        super().__init__()
        self.fre1 = WaveletDWT_Cross_IDWT(
            wavename="bior2.2",
            channels=channels,
            num_heads=cfg["num_heads"],
            topk=cfg["topk"],
        )
        self.fre2 = FrequencyBranch(
            channels=channels,
            num_experts=cfg["num_experts"],
            max_gn_groups=cfg["max_gn_groups"],
        )
        self.spa = ConvBlock(n_stages=n, n_filters_in=channels, n_filters_out=channels)
        self.fuse = CAFM(channels=channels)

    def forward(self, x):
        x_spa = self.spa(x)
        x_fre1 = self.fre1(x)
        x_fre2 = self.fre2(x_fre1)
        y = self.fuse(x_spa, x_fre2)
        return y


class Encoder(nn.Module):
    def __init__(self, n_channels=1, n_filters=16, has_residual=False):
        super().__init__()
        conv = ResidualConvBlock if has_residual else ConvBlock

        stages = [1, 2, 3, 3, 3]
        chs = [n_filters * (2 ** i) for i in range(5)]

        self.stem = conv(stages[0], n_channels, chs[0])

        self.blk0 = DualDomainBlockv1(n=stages[0], channels=chs[0], cfg=self._dual_domain_block_cfg(0))
        self.down0 = DownsamplingConvBlock(chs[0], chs[1])

        self.blk1 = DualDomainBlockV2(n=stages[1], channels=chs[1], cfg=self._dual_domain_block_cfg(1))
        self.down1 = DownsamplingConvBlock(chs[1], chs[2])

        self.blk2 = DualDomainBlockV2(n=stages[2], channels=chs[2], cfg=self._dual_domain_block_cfg(2))
        self.down2 = DownsamplingConvBlock(chs[2], chs[3])

        self.blk3 = DualDomainBlockV2(n=stages[3], channels=chs[3], cfg=self._dual_domain_block_cfg(3))
        self.down3 = DownsamplingConvBlock(chs[3], chs[4])

        self.blk4 = conv(stages[4], chs[4], chs[4])

    def _dual_domain_block_cfg(self, stage_idx: int) -> dict:
        cfgs = [
            {"num_experts": 4, "topk": 3, "num_heads": 2, "max_gn_groups": 4},
            {"num_experts": 4, "topk": 3, "num_heads": 4, "max_gn_groups": 8},
            {"num_experts": 4, "topk": 3, "num_heads": 8, "max_gn_groups": 8},
            {"num_experts": 4, "topk": 3, "num_heads": 8, "max_gn_groups": 8},
        ]
        return cfgs[stage_idx]

    def forward(self, x):
        stem_feat = self.stem(x)

        x0 = self.blk0(stem_feat)
        x1 = self.blk1(self.down0(x0))
        x2 = self.blk2(self.down1(x1))
        x3 = self.blk3(self.down2(x2))
        x4 = self.blk4(self.down3(x3))

        feats = [x0, x1, x2, x3, x4]
        return stem_feat, feats


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
            conv(dec_stages[3], 3 * chs[0], chs[0]),
        ])

        self.out_conv = nn.Conv3d(chs[0], n_classes, kernel_size=1)

    def forward(self, stem_feat, feats):
        x = feats[-1]

        for i, (up, blk) in enumerate(zip(self.ups, self.blocks)):
            x = up(x)
            skip = feats[-2 - i]

            if i == len(self.blocks) - 1:
                x = torch.cat([x, skip, stem_feat], dim=1)
            else:
                x = torch.cat([x, skip], dim=1)

            x = blk(x)

        return self.out_conv(x)


class VNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=14, n_filters=16, has_residual=False):
        super().__init__()
        self.encoder = Encoder(n_channels=n_channels, n_filters=n_filters, has_residual=has_residual)
        self.decoder = Decoder(n_classes=n_classes, n_filters=n_filters, has_residual=has_residual)

    def forward(self, x):
        stem, feats = self.encoder(x)
        return self.decoder(stem, feats)