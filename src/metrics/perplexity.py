import torch
from src.metrics.base_metric import BaseMetric

class MeanPerplexityMetric(BaseMetric):
    def __call__(self, perplexities, **kwargs):
        if isinstance(perplexities, list):
            return torch.stack(perplexities).mean().item()
        return perplexities