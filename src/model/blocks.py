import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, dilation = 1):
        super().__init__()

        self.padding_left = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            dilation = dilation
        )

    def forward(self, x):
        x = F.pad(x, (self.padding_left, 0))
        return self.conv(x)
    
class CausalConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1):
        super().__init__()

        self.conv = nn.ConvTranspose1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride
        )

        self.crop_right = kernel_size - stride

    def forward(self, x):
        out = self.conv(x)
        if self.crop_right > 0:
            out = out[..., :-self.crop_right]
        return out

class ResidualUnit(nn.Module):
    def __init__(self, n_channels, dilation):
        super().__init__()

        self.net = nn.Sequential(
            CausalConv1d(
                in_channels = n_channels,
                out_channels = n_channels,
                kernel_size = 7,
                dilation = dilation
            ),
            nn.ELU(),
            CausalConv1d(
                in_channels = n_channels,
                out_channels = n_channels,
                kernel_size = 1
            )
        )

    def forward(self, x):
        return self.net(x) + x
    

class EncoderBlock(nn.Module):
    def __init__(self, n_channels, stride):
        super().__init__()
        
        self.net = nn.Sequential(
            ResidualUnit(n_channels = n_channels // 2, dilation = 1),
            nn.ELU(),
            ResidualUnit(n_channels = n_channels // 2, dilation = 3),
            nn.ELU(),
            ResidualUnit(n_channels = n_channels // 2, dilation = 9),
            nn.ELU(),
            CausalConv1d(
                in_channels = n_channels // 2, 
                out_channels = n_channels, 
                kernel_size = 2 * stride, 
                stride = stride
            )
        )
    
    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, n_channels, stride):
        super().__init__()

        self.net = nn.Sequential(
            CausalConvTranspose1d(
                in_channels = n_channels, 
                out_channels = n_channels // 2, 
                kernel_size = 2 * stride, 
                stride = stride
            ),
            nn.ELU(),
            ResidualUnit(n_channels = n_channels // 2, dilation = 1),
            nn.ELU(),
            ResidualUnit(n_channels = n_channels // 2, dilation = 3),
            nn.ELU(),
            ResidualUnit(n_channels = n_channels // 2, dilation = 9),
        )
    
    def forward(self, x):
        return self.net(x)
    
class ResidualUnit2d(nn.Module):
    def __init__(self, in_channels, out_channels, m, s: Tuple[int, int]):
        self.net = nn.Sequential(
            nn.Conv2d(kernel_size = (3, 3), in_channels = in_channels, out_channels = out_channels),
            nn.ELU(),
            nn.Conv2d(kernel_size = (s[0] + 2, s[1] + 2), in_channels = out_channels, out_channels = out_channels * m, stride = s)
        )

        self.skip_connection = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels * m,
            kernel_size = (1, 1),
            stride = s
        )

    def forward(self, x):
        return self.skip_connection(x) + self.net(x)

def NormalizedConv1d(**kwargs):
    return F.instance_norm(nn.Conv1d(**kwargs))

class WaveDiscriminatorBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            NormalizedConv1d(in_channels = 1, out_channels = 16, kernel_size = 15),
            nn.LeakyReLU(0.2),
            NormalizedConv1d(in_channels = 16, out_channels = 64, kernel_size = 41, padding = 20, stride = 4, groups = 4),
            nn.LeakyReLU(0.2),
            NormalizedConv1d(in_channels = 64, out_channels = 256, kernel_size = 41, padding = 20, stride = 4, groups = 16),
            nn.LeakyReLU(0.2),
            NormalizedConv1d(in_channels = 256, out_channels = 1024, kernel_size = 41, padding = 20, stride = 4, groups = 64),
            nn.LeakyReLU(0.2),
            NormalizedConv1d(in_channels = 1024, out_channels = 1024, kernel_size = 41, padding = 20, stride = 4, groups = 256),
            nn.LeakyReLU(0.2),
            NormalizedConv1d(in_channels = 1024, out_channels = 1024, kernel_size = 5, padding = 2, stride = 1),
            nn.LeakyReLU(0.2),
            NormalizedConv1d(in_channels = 1024, out_channels = 1, kernel_size = 3, padding = 1, stride = 1)
        )

    def forward(self, x):
        return self.net(x)