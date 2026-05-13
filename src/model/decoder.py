import torch.nn as nn

from src.model.blocks import DecoderBlock, CausalConv1d

class Decoder(nn.Module):
    def __init__(self, encoder_channels, target_channels):
        super().__init__()

        self.net = nn.Sequential(
            CausalConv1d(in_channels = target_channels, out_channels = 16 * encoder_channels, kernel_size = 7),
            nn.ELU(),
            DecoderBlock(n_channels = 16 * encoder_channels, stride = 5),
            DecoderBlock(n_channels = 8 * encoder_channels, stride = 5),
            DecoderBlock(n_channels = 4 * encoder_channels, stride = 4),
            DecoderBlock(n_channels = 2 * encoder_channels, stride = 2),
            nn.ELU(),
            CausalConv1d(in_channels = encoder_channels, out_channels = 1, kernel_size = 7)
        )

    def forward(self, x):
        return self.net(x)