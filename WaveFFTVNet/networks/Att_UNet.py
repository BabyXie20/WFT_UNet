import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpConv3D(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class AttentionBlock3D(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttU_Net3D(nn.Module):
    def __init__(self, input_channel: int = 1, num_classes: int = 14, base_channels: int = 16):
        super().__init__()

        c1 = base_channels
        c2, c3, c4, c5 = c1 * 2, c1 * 4, c1 * 8, c1 * 16

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock3D(ch_in=input_channel, ch_out=c1)
        self.Conv2 = ConvBlock3D(ch_in=c1, ch_out=c2)
        self.Conv3 = ConvBlock3D(ch_in=c2, ch_out=c3)
        self.Conv4 = ConvBlock3D(ch_in=c3, ch_out=c4)
        self.Conv5 = ConvBlock3D(ch_in=c4, ch_out=c5)

        self.Up5 = UpConv3D(ch_in=c5, ch_out=c4)
        self.Att5 = AttentionBlock3D(F_g=c4, F_l=c4, F_int=c4 // 2)
        self.Up_conv5 = ConvBlock3D(ch_in=c5, ch_out=c4)

        self.Up4 = UpConv3D(ch_in=c4, ch_out=c3)
        self.Att4 = AttentionBlock3D(F_g=c3, F_l=c3, F_int=c3 // 2)
        self.Up_conv4 = ConvBlock3D(ch_in=c4, ch_out=c3)

        self.Up3 = UpConv3D(ch_in=c3, ch_out=c2)
        self.Att3 = AttentionBlock3D(F_g=c2, F_l=c2, F_int=c2 // 2)
        self.Up_conv3 = ConvBlock3D(ch_in=c3, ch_out=c2)

        self.Up2 = UpConv3D(ch_in=c2, ch_out=c1)
        self.Att2 = AttentionBlock3D(F_g=c1, F_l=c1, F_int=c1 // 2)
        self.Up_conv2 = ConvBlock3D(ch_in=c2, ch_out=c1)

        self.Conv_1x1 = nn.Conv3d(c1, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        return self.Conv_1x1(d2)
    
#from networks.model import AttU_Net3D
#model = AttU_Net3D(input_channel=1, num_classes=num_classes, base_channels=16).to(device)