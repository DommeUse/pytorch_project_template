import torch
import torch.nn as nn

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, embedding_dim, decay = 0.99, eps = 1e-5, dead_code_threshold = 2, k_mean_iters = 10):
        super().__init__()

        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps
        self.dead_code_threshold = dead_code_threshold
        self.k_means_iters = k_mean_iters

        self.register_buffer("codebook", torch.zeros(codebook_size, embedding_dim))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embedding_sum", torch.zeros(codebook_size, embedding_dim))
        self.register_buffer("k_means_inited", torch.tensor(False))

    @torch.no_grad()
    def _k_means_init(self, z_flat):
        batch_size = z_flat.shape[0]

        if batch_size >= self.codebook_size:
            idx = torch.randperm(batch_size, device = z_flat.device)[:self.codebook_size]
        else:
            idx = torch.randint(0, batch_size, (self.codebook_size, ), device = z_flat.device)

        centroids = z_flat[idx].clone()

        for i in range(self.k_means_iters):
            distances = (z_flat ** 2).sum(-1, keepdim = True) - 2 * z_flat @ centroids.T + (centroids ** 2).sum(-1)
            nearest = distances.argmin(dim = -1)

            one_hot = torch.nn.functional.one_hot(nearest, self.codebook_size).type_as(z_flat)
            frequency = one_hot.sum(dim = 0)
            new_centroids = one_hot.T @ z_flat / frequency.unsqueeze(-1).clamp(min = 1)

            not_empty = frequency != 0
            centroids[not_empty] = new_centroids[not_empty]

        self.codebook.copy_(centroids)
        self.cluster_size.fill_(1)
        self.embedding_sum.copy_(centroids)

    @torch.no_grad()
    def _update_ema(self, z_flat, idx):
        one_hot = torch.nn.functional.one_hot(idx, self.codebook_size).type_as(z_flat)
        cluster_batch_size = one_hot.sum(dim = 0)
        embedding_batch_sum = one_hot.T @ z_flat

        self.cluster_size = self.cluster_size * self.decay + cluster_batch_size * (1 - self.decay)
        self.embedding_sum = self.embedding_sum * self.decay + embedding_batch_sum * (1 - self.decay)

        n = self.cluster_size.sum()
        normalized_cluster_size = (self.cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n

        self.codebook.copy_(self.embedding_sum / normalized_cluster_size.unsqueeze(-1))

    @torch.no_grad()
    def _reset_dead_codes(self, z_flat):
        dead_idx = self.cluster_size < self.dead_code_threshold

        if dead_idx.sum() == 0:
            return
        
        replace_idx = torch.randint(0, z_flat.shape[0], (int(dead_idx.sum()),), device = z_flat.device)

        self.codebook[dead_idx] = z_flat[replace_idx]
        self.cluster_size[dead_idx] = 1
        self.embedding_sum[dead_idx] = z_flat[replace_idx]

    @torch.no_grad()
    def _compute_ppl(self, idx):
        one_hot = torch.nn.functional.one_hot(idx, self.codebook_size).float()
        avg_probs = one_hot.mean(dim = 0)
        return torch.exp(-(avg_probs * (avg_probs + self.eps).log()).sum())

    def forward(self, z):
        B, D, T = z.shape

        z_flat = z.permute(0, 2, 1).contiguous().view(-1, D)

        if self.training and not self.k_means_inited.item():
            self._k_means_init(z_flat)
            self.k_means_inited.fill_(True)

        distances = (z_flat ** 2).sum(-1, keepdim = True) - 2 * z_flat @ self.codebook.T + (self.codebook ** 2).sum(-1)
        idx = distances.argmin(dim = -1)
        z_hat_flat = self.codebook[idx]
        z_hat = z_hat_flat.view(B, T, D).permute(0, 2, 1)

        if self.training:
            self._update_ema(z_flat, idx)
            self._reset_dead_codes(z_flat)

        perplexity = self._compute_ppl(idx)
        
        return z_hat, idx.view(B, T), perplexity


class ResidualVQ(nn.Module):
    def __init__(self, n_quantizers, codebook_size, embedding_dim, **kwargs):
        super().__init__()

        self.n_quantizers = n_quantizers

        self.quantizers = nn.ModuleList([
            VectorQuantizer(codebook_size, embedding_dim, **kwargs) for i in range(n_quantizers)
        ])

    def forward(self, z):
        residual = z.clone()
        z_hat = torch.zeros_like(z)

        all_idx = []
        all_ppls = []
        total_commitment = 0

        for quantizer in self.quantizers:
            q, idx, ppl = quantizer(residual)

            z_hat = z_hat + q
            residual = residual - q

            all_idx.append(idx)
            all_ppls.append(ppl)

        commitment_loss = torch.nn.functional.mse_loss(z, z_hat.detach())
        z_hat = z + (z_hat - z).detach()

        return z_hat, all_idx, commitment_loss, all_ppls
