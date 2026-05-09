import torch.nn as nn
from src.loss.adversarial import discriminator_hinge_loss

class DiscriminatorLoss(nn.Module):
    def forward(self, real_logits, fake_logits):
        loss = discriminator_hinge_loss(real_logits, fake_logits)
        return {"loss": loss}