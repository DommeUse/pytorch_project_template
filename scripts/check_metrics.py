import torch
from src.metrics.nisqa_v2 import NISQAMetric
from src.metrics.stoi import STOIMetric

m1 = NISQAMetric(sample_rate = 16000)
m2 = STOIMetric(sample_rate = 16000)

audio = torch.randn(2, 16000)

score1 = m1(audio)
score2 = m2(audio, audio)
print(f"(NISQA, STOI): ({score1}, {score2})")