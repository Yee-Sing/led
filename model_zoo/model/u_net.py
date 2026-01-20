import torch
import torch.nn as nn
import torch.nn.functional as F

class residualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(residualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class UpSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm=None):
        super(UpSampleLayer, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm_layer = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x_upsampled = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.conv2d(x_upsampled)
        out = self.norm_layer(out)
        if self.activation is not None:
            out = self.activation(out)

        return out


class ED_U_shape(nn.Module):
    def __init__(self):
        super(ED_U_shape, self).__init__()

        self.static_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.residualBlock = nn.Sequential(
            residualBlock(256, 256))

        self.up1 = UpSampleLayer(in_channels=512, out_channels=128, kernel_size=5, stride=1, padding=2)

        self.up2 = UpSampleLayer(in_channels=256, out_channels=64, kernel_size=5, stride=1, padding=2)

        self.up3 = UpSampleLayer(in_channels=128, out_channels=32, kernel_size=5, stride=1, padding=2)

        self.temporalflat = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        x_in = self.static_conv(x)

        x1 = self.down1(x_in)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        r1 = self.residualBlock(x3)

        u1 = self.up1(torch.cat([r1, x3], 1))
        u2 = self.up2(torch.cat([u1, x2], 1))
        u3 = self.up3(torch.cat([u2, x1], 1))

        output = self.temporalflat(torch.cat([u3, x_in], 1))

        return output

