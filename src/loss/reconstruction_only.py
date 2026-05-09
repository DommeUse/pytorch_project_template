import torch.nn as nn
from src.loss.reconstruction import MultiScaleMelLoss

class ReconstructionOnlyLoss(nn.Module):
    def __init__(self, sample_rate = 16000, lambda_rec = 1.0, lambda_commit = 1.0):
        super().__init__()
        self.rec_loss = MultiScaleMelLoss(sample_rate = sample_rate)
        self.lambda_rec = lambda_rec
        self.lambda_commit = lambda_commit

    def forward(self, audio, output, commitment_loss, fake_logits, real_features, fake_features):
        rec = self.rec_loss(output, audio)

        total = self.lambda_rec * rec + self.lambda_commit * commitment_loss

        return {
            "loss" : total,
            "g_rec": rec,
            "g_commit": commitment_loss
        }