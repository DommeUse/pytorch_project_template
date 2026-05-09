import torch
from src.model.encoder import Encoder
from src.model.decoder import Decoder

"""Encoder and Decoder check"""

enc = Encoder(encoder_channels = 32, target_channels = 256)
dec = Decoder(encoder_channels = 32, target_channels = 256)

x = torch.randn(2, 1, 16000)

z = enc(x)
print(f"Latent:{tuple(z.shape)}")

x_hat = dec(z)
print(f"Output:{tuple(x_hat.shape)}")

loss = (x - x_hat).abs().mean()
loss.backward()

"""SoundStream check"""

from src.model.soundstream import SoundStream

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SoundStream(
    encoder_channels = 32,
    target_channels = 256,
    n_quantizers = 8,
    codebook_size = 1024,
).to(device)

model.train()

x = torch.randn(2, 1, 16000, device = device)
out = model(x)

print(f"Reconstructed: {out['output'].shape}")
print(f"Indices: {len(out['indices'])}, {out['indices'][0].shape}")
print(f"Commitment loss: {out['commitment_loss'].item()}")
print(f"Perplexities: {[f'{p.item()}' for p in out['perplexities']]}")

loss = (x - out['output']).abs().mean() + out['commitment_loss']
loss.backward()

print("Backward OK")
print()
print(f"First quantizer codebook initialized: {model.rvq.quantizers[0].k_means_inited.item()}")