import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import math

class MultiScaleMelLoss(nn.Module):
    def __init__(self, sample_rate = 16000, window_sizes = [2**s for s in range(6, 12)], mel_bins = 64, eps = 1e-5):
        super().__init__()

        self.window_sizes = window_sizes
        self.eps = eps

        self.mel_transforms = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate = sample_rate,
                n_fft = s,
                hop_length = s // 4,
                win_length = s,
                n_mels = mel_bins,
                power = 1.0,
                center = True,
            )
            for s in window_sizes
        ])

        alphas = torch.tensor([(math.log2(s) / 2) ** 0.5 for s in window_sizes])
        self.register_buffer("alphas", alphas)

    def forward(self, output, audio):
        if output.dim() == 3:
            output = output.squeeze(1)
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        loss = 0

        for i in range(len(self.alphas)):
            transform = self.mel_transforms[i]

            mel_output = transform(output)
            mel_audio = transform(audio)

            l1 = F.l1_loss(mel_output, mel_audio)

            log_output = torch.log(mel_output + self.eps)
            log_audio = torch.log(mel_audio + self.eps)
            
            l2 = F.mse_loss(log_output, log_audio)

            loss = loss + l1 + self.alphas[i] * l2

        return loss