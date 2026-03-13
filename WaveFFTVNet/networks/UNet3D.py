import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        f_maps=64,
        num_levels: int = 4,
        kernel_size: int = 3,
        padding: int = 1,
        pool_kernel_size: int = 2,
        upsample_mode: str = "nearest",
    ):
        super().__init__()

        if isinstance(f_maps, int):
            f_maps = [f_maps * (2**k) for k in range(num_levels)]
        if len(f_maps) < 2:
            raise ValueError("f_maps 至少需要包含 2 个层级")

        self.upsample_mode = upsample_mode
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel_size)

        # encoder
        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for channels in f_maps:
            self.encoders.append(self._double_conv(prev_channels, channels, kernel_size, padding))
            prev_channels = channels

        # decoder
        self.decoders = nn.ModuleList()
        rev_f_maps = f_maps[::-1]
        for i in range(len(rev_f_maps) - 1):
            in_ch = rev_f_maps[i] + rev_f_maps[i + 1]   # upsample 后与 skip concat
            out_ch = rev_f_maps[i + 1]
            self.decoders.append(self._double_conv(in_ch, out_ch, kernel_size, padding))

        self.final_conv = nn.Conv3d(f_maps[0], out_channels, kernel_size=1)

    def _conv_block(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def _double_conv(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        return nn.Sequential(
            self._conv_block(in_channels, out_channels, kernel_size, padding),
            self._conv_block(out_channels, out_channels, kernel_size, padding),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder
        features = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            features.append(x)
            if i != len(self.encoders) - 1:
                x = self.pool(x)

        # decoder
        x = features[-1]
        skip_connections = features[:-1][::-1]

        for decoder, skip in zip(self.decoders, skip_connections, strict=False):
            x = F.interpolate(x, size=skip.shape[2:], mode=self.upsample_mode)
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        logits = self.final_conv(x)
        return logits