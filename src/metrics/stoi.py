import torch
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from src.metrics.base_metric import BaseMetric

class STOIMetric(BaseMetric):
    def __init__(self, sample_rate = 16000, device = "auto", *args, **kwargs):
        super().__init__(*args, **kwargs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.metric = ShortTimeObjectiveIntelligibility(fs = sample_rate, extended = False).to(device)

    @torch.no_grad()
    def __call__(self, output, audio):
        return self.metric(output, audio).item()