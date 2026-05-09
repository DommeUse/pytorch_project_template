import torch.nn as nn

from src.model.encoder import Encoder
from src.model.decoder import Decoder
from src.model.rvq import ResidualVQ
class SoundStream(nn.Module):
    def __init__(self, encoder_channels, target_channels, n_quantizers, codebook_size):
        super().__init__()

        self.encoder = Encoder(encoder_channels = encoder_channels, target_channels = target_channels)
        self.rvq = ResidualVQ(
            n_quantizers = n_quantizers, 
            codebook_size = codebook_size, 
            embedding_dim = target_channels
        )
        self.decoder = Decoder(encoder_channels = encoder_channels, target_channels = target_channels)

    def forward(self, x):
        z = self.encoder(x)
        z_hat, all_idx, all_commitment, all_ppls = self.rvq(z)
        out = self.decoder(z_hat)
        return {
            "output" : out, 
            "indices" : all_idx, 
            "commitment_loss" : all_commitment,
            "perplexities" : all_ppls
        }