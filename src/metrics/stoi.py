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
    def __call__(self, output, audio, **kwargs):
        if output.dim() == 3:
            output = output.squeeze(1)
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        min_len = min(output.shape[-1], audio.shape[-1])
        output = output[..., :min_len]
        audio = audio[..., :min_len]

        return self.metric(output, audio).item()