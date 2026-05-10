import torch.nn as nn
from src.loss.reconstruction import MultiScaleMelLoss
from src.loss.adversarial import generator_adversarial_loss, feature_matching_loss

class GeneratorLoss(nn.Module):
    def __init__(self, sample_rate = 16000, lambda_adv = 1.0, lambda_feat = 100.0, lambda_rec = 1.0, lambda_commit = 1.0):
        super().__init__()
        self.rec_loss = MultiScaleMelLoss(sample_rate = sample_rate)
        self.lambda_adv = lambda_adv
        self.lambda_feat = lambda_feat
        self.lambda_rec = lambda_rec
        self.lambda_commit = lambda_commit

    def forward(self, audio, output, commitment_loss, fake_logits, real_features, fake_features, **kwargs):
        rec = self.rec_loss(output, audio)
        adv = generator_adversarial_loss(fake_logits)
        feat = feature_matching_loss(real_features, fake_features)

        generator_loss = self.lambda_rec * rec + self.lambda_adv * adv + self.lambda_feat * feat + self.lambda_commit * commitment_loss

        return {
            "loss" : generator_loss,
            "g_rec": rec,
            "g_adv": adv,
            "g_feat": feat,
            "g_commit": commitment_loss
        }