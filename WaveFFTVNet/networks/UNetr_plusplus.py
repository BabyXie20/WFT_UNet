import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Sequence, Tuple, Union
from einops import rearrange

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from timm.models.layers import trunc_normal_


def to_3tuple(x: Union[int, Sequence[int]]) -> Tuple[int, int, int]:
    if isinstance(x, int):
        return (x, x, x)
    return tuple(x)


def prod(x: Sequence[int]) -> int:
    return int(np.prod(x))


def valid_num_groups(channels: int, preferred: int = 8) -> int:
    preferred = min(preferred, channels)
    for g in range(preferred, 0, -1):
        if channels % g == 0:
            return g
    return 1


def compute_stage_shapes(img_size: Union[int, Sequence[int]]) -> list[Tuple[int, int, int]]:
    img_size = to_3tuple(img_size)

    s0 = (img_size[0] // 2, img_size[1] // 4, img_size[2] // 4)
    s1 = (s0[0] // 2, s0[1] // 2, s0[2] // 2)
    s2 = (s1[0] // 2, s1[1] // 2, s1[2] // 2)
    s3 = (s2[0] // 2, s2[1] // 2, s2[2] // 2)

    return [s0, s1, s2, s3]


def get_padding(
    kernel_size: Union[Sequence[int], int],
    stride: Union[Sequence[int], int],
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    padding = tuple(int(p) for p in padding_np)
    return padding if len(padding) > 1 else padding[0]


def get_output_padding(
    kernel_size: Union[Sequence[int], int],
    stride: Union[Sequence[int], int],
    padding: Union[Sequence[int], int],
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)
    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    out_padding = tuple(int(p) for p in out_padding_np)
    return out_padding if len(out_padding) > 1 else out_padding[0]


def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Sequence[int], int] = 3,
    stride: Union[Sequence[int], int] = 1,
    act: Optional[Union[Tuple, str]] = Act.PRELU,
    norm: Union[Tuple, str] = Norm.INSTANCE,
    dropout: Optional[Union[Tuple, str, float]] = None,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)

    return Convolution(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )


class UnetResBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            conv_only=True,
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

        self.downsample = (in_channels != out_channels) or (np.any(np.atleast_1d(stride) != 1))
        if self.downsample:
            self.conv3 = get_conv_layer(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                dropout=dropout,
                conv_only=True,
            )
            self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.lrelu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample:
            residual = self.conv3(residual)
            residual = self.norm3(residual)

        out = out + residual
        out = self.lrelu(out)
        return out


class UnetOutBlock(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=True,
            conv_only=True,
        )

    def forward(self, x):
        return self.conv(x)


class EPA(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        proj_size: int,
        num_heads: int = 4,
        qkv_bias: bool = False,
        channel_attn_drop: float = 0.1,
        spatial_attn_drop: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)
        self.E = nn.Linear(input_size, proj_size)
        self.F = nn.Linear(input_size, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, hidden_size // 2)
        self.out_proj2 = nn.Linear(hidden_size, hidden_size // 2)

    def forward(self, x):
        b, n, c = x.shape

        qkvv = self.qkvv(x).reshape(b, n, 4, self.num_heads, c // self.num_heads)
        qkvv = qkvv.permute(2, 0, 3, 1, 4)
        q_shared, k_shared, v_ca, v_sa = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_ca = v_ca.transpose(-2, -1)
        v_sa = v_sa.transpose(-2, -1)

        k_shared_projected = self.E(k_shared)
        v_sa_projected = self.F(v_sa)

        q_shared = F.normalize(q_shared, dim=-1)
        k_shared = F.normalize(k_shared, dim=-1)

        attn_ca = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature
        attn_ca = self.attn_drop(attn_ca.softmax(dim=-1))
        x_ca = (attn_ca @ v_ca).permute(0, 3, 1, 2).reshape(b, n, c)

        attn_sa = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2
        attn_sa = self.attn_drop_2(attn_sa.softmax(dim=-1))
        x_sa = (attn_sa @ v_sa_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(b, n, c)

        x_sa = self.out_proj(x_sa)
        x_ca = self.out_proj2(x_ca)
        x = torch.cat((x_sa, x_ca), dim=-1)

        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        proj_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        pos_embed: bool = True,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)

        self.epa_block = EPA(
            input_size=input_size,
            hidden_size=hidden_size,
            proj_size=proj_size,
            num_heads=num_heads,
            channel_attn_drop=dropout_rate,
            spatial_attn_drop=dropout_rate,
        )

        self.conv51 = UnetResBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=3,
            stride=1,
            norm_name="batch",
        )
        self.conv8 = nn.Sequential(
            nn.Dropout3d(0.1, inplace=False),
            nn.Conv3d(hidden_size, hidden_size, kernel_size=1),
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size)) if pos_embed else None

    def forward(self, x):
        b, c, d, h, w = x.shape

        x = x.reshape(b, c, d * h * w).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        attn = x + self.gamma * self.epa_block(self.norm(x))

        attn_skip = attn.reshape(b, d, h, w, c).permute(0, 4, 1, 2, 3)
        attn = self.conv51(attn_skip)
        out = attn_skip + self.conv8(attn)

        return out


class UnetrPPEncoder(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Sequence[int]] = (96, 96, 96),
        dims: Sequence[int] = (32, 64, 128, 256),
        proj_size: Sequence[int] = (64, 64, 64, 32),
        depths: Sequence[int] = (3, 3, 3, 3),
        num_heads: int = 4,
        spatial_dims: int = 3,
        in_channels: int = 1,
        dropout: float = 0.0,
        transformer_dropout_rate: float = 0.15,
    ):
        super().__init__()

        self.img_size = to_3tuple(img_size)
        self.stage_shapes = compute_stage_shapes(self.img_size)
        self.input_sizes = [prod(s) for s in self.stage_shapes]

        self.downsample_layers = nn.ModuleList()

        stem_layer = nn.Sequential(
            get_conv_layer(
                spatial_dims,
                in_channels,
                dims[0],
                kernel_size=(2, 4, 4),
                stride=(2, 4, 4),
                dropout=dropout,
                conv_only=True,
            ),
            get_norm_layer(
                name=("group", {"num_groups": valid_num_groups(dims[0])}),
                spatial_dims=spatial_dims,
                channels=dims[0],
            ),
        )
        self.downsample_layers.append(stem_layer)

        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(
                    spatial_dims,
                    dims[i],
                    dims[i + 1],
                    kernel_size=(2, 2, 2),
                    stride=(2, 2, 2),
                    dropout=dropout,
                    conv_only=True,
                ),
                get_norm_layer(
                    name=("group", {"num_groups": valid_num_groups(dims[i + 1])}),
                    spatial_dims=spatial_dims,
                    channels=dims[i + 1],
                ),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(4):
            blocks = [
                TransformerBlock(
                    input_size=self.input_sizes[i],
                    hidden_size=dims[i],
                    proj_size=proj_size[i],
                    num_heads=num_heads,
                    dropout_rate=transformer_dropout_rate,
                    pos_embed=True,
                )
                for _ in range(depths[i])
            ]
            self.stages.append(nn.Sequential(*blocks))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        hidden_states = []

        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i == 3:
                x = rearrange(x, "b c d h w -> b (d h w) c")
            hidden_states.append(x)

        return x, hidden_states


class UnetrUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        proj_size: int,
        num_heads: int,
        out_size: int,
        depth: int = 3,
        conv_decoder: bool = False,
    ):
        super().__init__()

        self.transp_conv = get_conv_layer(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_kernel_size,
            conv_only=True,
            is_transposed=True,
        )

        if conv_decoder:
            self.decoder_block = UnetResBlock(
                spatial_dims=spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.decoder_block = nn.Sequential(
                *[
                    TransformerBlock(
                        input_size=out_size,
                        hidden_size=out_channels,
                        proj_size=proj_size,
                        num_heads=num_heads,
                        dropout_rate=0.15,
                        pos_embed=True,
                    )
                    for _ in range(depth)
                ]
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, skip):
        x = self.transp_conv(x)
        x = x + skip
        x = self.decoder_block(x)
        return x


class UNETR_PP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Union[int, Sequence[int]] = (96, 96, 96),
        feature_size: int = 16,
        dims: Sequence[int] = (32, 64, 128, 256),
        depths: Sequence[int] = (3, 3, 3, 3),
        proj_size: Sequence[int] = (64, 64, 64, 32),
        num_heads: int = 4,
        norm_name: Union[Tuple, str] = "instance",
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.img_size = to_3tuple(img_size)
        self.stage_shapes = compute_stage_shapes(self.img_size)
        self.feat_size = self.stage_shapes[-1]
        self.hidden_size = dims[-1]
        self.num_classes = out_channels

        self.unetr_pp_encoder = UnetrPPEncoder(
            img_size=self.img_size,
            dims=dims,
            proj_size=proj_size,
            depths=depths,
            num_heads=num_heads,
            spatial_dims=3,
            in_channels=in_channels,
            dropout=dropout_rate,
            transformer_dropout_rate=0.15,
        )

        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=dims[3],
            out_channels=dims[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            proj_size=proj_size[2],
            num_heads=num_heads,
            out_size=prod(self.stage_shapes[2]),
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=dims[2],
            out_channels=dims[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            proj_size=proj_size[1],
            num_heads=num_heads,
            out_size=prod(self.stage_shapes[1]),
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=dims[1],
            out_channels=dims[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            proj_size=proj_size[0],
            num_heads=num_heads,
            out_size=prod(self.stage_shapes[0]),
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=dims[0],
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(2, 4, 4),
            norm_name=norm_name,
            proj_size=proj_size[0],
            num_heads=num_heads,
            out_size=prod(self.img_size),
            conv_decoder=True,
        )

        self.out = UnetOutBlock(
            spatial_dims=3,
            in_channels=feature_size,
            out_channels=out_channels,
        )

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x):
        _, hidden_states = self.unetr_pp_encoder(x)

        conv_block = self.encoder1(x)

        enc1, enc2, enc3, enc4 = hidden_states

        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)
        out = self.decoder2(dec1, conv_block)

        logits = self.out(out)
        return logits

