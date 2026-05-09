import torch
import torch.nn.functional as F

def discriminator_hinge_loss(real_logits_list, fake_logits_list):
    loss = 0.0

    for real_logits, fake_logits in zip(real_logits_list, fake_logits_list):
        real_term = F.relu(1.0 - real_logits).mean()
        fake_term = F.relu(1.0 + fake_logits).mean()
        loss = loss + real_term + fake_term

    return loss / len(real_logits_list)

def generator_adversarial_loss(fake_logits_list):
    loss = 0.0

    for fake_logits in fake_logits_list:
        loss = loss + F.relu(1.0 - fake_logits).mean()

    return loss / len(fake_logits_list)

def feature_matching_loss(real_features_list, fake_features_list):
    loss = 0.0
    count = 0
    for real_feats, fake_feats in zip(real_features_list, fake_features_list):
        for r, f in zip(real_feats, fake_feats):
            loss = loss + F.l1_loss(f, r.detach())
            count += 1
    return loss / max(count, 1)