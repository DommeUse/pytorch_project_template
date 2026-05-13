import torch
import torch.nn as nn

from src.model.blocks import ResidualUnit2d, WaveDiscriminatorBlock

class STFTDiscriminator(nn.Module):
    def __init__(self, hidden_dim, n_fft, hop_length, n_bins):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.net = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(kernel_size = (7, 7), in_channels = 2, out_channels = 32, padding = 3),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels = 32, out_channels = hidden_dim, m = 2, s = (1, 2)),
                nn.ELU(),
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels = 2 * hidden_dim, out_channels = 2 * hidden_dim, m = 2, s = (2, 2)),
                nn.ELU(),
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels = 4 * hidden_dim, out_channels = 4 * hidden_dim, m = 1, s = (1, 2)),
                nn.ELU(),
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels = 4 * hidden_dim, out_channels = 4 * hidden_dim, m = 2, s = (2, 2)),
                nn.ELU(),
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels = 8 * hidden_dim, out_channels = 8 * hidden_dim, m = 1, s = (1, 2)),
                nn.ELU(),
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels = 8 * hidden_dim, out_channels = 8 * hidden_dim, m = 2, s = (2, 2)),
                nn.ELU(),
            ),
            nn.Conv2d(kernel_size = (n_bins // 2**6, 1), in_channels = 16 * hidden_dim, out_channels = 1)
        ])

    def forward(self, x):
        spec = torch.stft(
            x.squeeze(1), 
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            win_length = 1024,
            return_complex = True,
            center = True
        )

        spec = spec[:, 1:]
        spec = torch.view_as_real(spec).permute(0, 3, 1, 2)

        feature_map = []
        for i in range(len(self.net)):
            spec = self.net[i](spec)
            feature_map.append(spec)
        return feature_map

class WaveDiscriminator(nn.Module):
    def __init__(self, n_blocks = 3):
        super().__init__()

        self.n_blocks = n_blocks

        self.net = nn.ModuleList([
            WaveDiscriminatorBlock() for i in range(n_blocks)
        ])

        self.downsampler = nn.AvgPool1d(kernel_size = 4, stride = 2, padding = 1)

    def forward(self, x):
        feature_map = []

        for i in range(self.n_blocks):
            feature_map.append(self.net[i](x))
            x = self.downsampler(x)

        return feature_map
    
class MultiScaleDiscriminator(nn.Module):
    def __init__(self, hidden_dim = 32, n_fft = 1024, hop_length = 256, n_bins = 512, n_wave_blocks = 3):
        super().__init__()

        self.stft_disc = STFTDiscriminator(hidden_dim, n_fft, hop_length, n_bins)
        self.wave_disc = WaveDiscriminator(n_wave_blocks)

    def forward(self, x):
        stft_features = self.stft_disc(x)
        wave_features = self.wave_disc(x)

        return stft_features + list(wave_features)