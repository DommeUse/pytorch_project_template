import torch
from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment
from src.metrics.base_metric import BaseMetric


class NISQAMetric(BaseMetric):
    def __init__(self, sample_rate = 16000, device = "auto", *args, **kwargs):
        super().__init__(*args, **kwargs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.metric = NonIntrusiveSpeechQualityAssessment(fs = sample_rate).to(device)

    @torch.no_grad()
    def __call__(self, output, **kwargs):
        if output.dim() == 3:
            output = output.squeeze(1)

        return self.metric(output)[0].item()