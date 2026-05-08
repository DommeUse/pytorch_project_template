import torch.nn as nn

from src.model.blocks import EncoderBlock, CausalConv1d

class Encoder(nn.Module):
    def __init__(self, encoder_channels, target_channels):
        super().__init__()

        self.net = nn.Sequential(
            CausalConv1d(in_channels = 1, out_channels = encoder_channels, kernel_size = 7),
            nn.ELU(),
            EncoderBlock(n_channels = 2 * encoder_channels, stride = 2),
            nn.ELU(),
            EncoderBlock(n_channels = 4 * encoder_channels, stride = 4),
            nn.ELU(),
            EncoderBlock(n_channels = 8 * encoder_channels, stride = 5),
            nn.ELU(),
            EncoderBlock(n_channels = 16 * encoder_channels, stride = 5),
            nn.ELU(),
            CausalConv1d(in_channels = 16 * encoder_channels, out_channels = target_channels, kernel_size = 3)
        )

    def forward(self, x):
        return self.net(x)