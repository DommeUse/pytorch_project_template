import torch
from src.loss.reconstruction import MultiScaleMelLoss
from src.loss.generator import GeneratorLoss
from src.loss.discriminator import DiscriminatorLoss

device = "cuda" if torch.cuda.is_available() else "cpu"

''' MultiScale mel loss test '''

loss_fn = MultiScaleMelLoss().to(device)
x = torch.randn(2, 16000, device = device)

print(f"Same audio: {loss_fn(x, x).item()}")
print(f"Tiny noise: {loss_fn(x + 0.01 * torch.randn_like(x), x).item()}")
print(f"Big noise: {loss_fn(x + 0.5 * torch.randn_like(x), x).item()}")

''' Generator loss test '''


gen_loss = GeneratorLoss().to(device)

B = 2
T = 16000

audio = torch.randn(B, 1, T, device = device)
output = audio + 0.01 * torch.randn_like(audio)

output = output.detach().requires_grad_(True)

n_disc = 5
fake_logits = [
    torch.randn(B, 1, 50, device=device, requires_grad = True)
    for _ in range(n_disc)
]

n_layers = 7
real_features = [
    [torch.randn(B, 32, 50, device = device) for _ in range(n_layers)]
    for _ in range(n_disc)
]
fake_features = [
    [torch.randn(B, 32, 50, device = device, requires_grad=True) for _ in range(n_layers)]
    for _ in range(n_disc)
]

commit_loss = torch.tensor(0.1, device = device, requires_grad = True) # Random value

losses = gen_loss(
    audio = audio,
    output = output,
    commitment_loss = commit_loss,
    fake_logits = fake_logits,
    real_features = real_features,
    fake_features = fake_features,
)

print(f"Total loss: {losses['loss'].item()}")
print(f"rec: {losses['g_rec'].item()}")
print(f"adv: {losses['g_adv'].item()}")
print(f"feat: {losses['g_feat'].item()}")
print(f"commit: {losses['g_commit'].item()}")

losses["loss"].backward()
print()
print(f"Backward OK")


''' Discriminator Loss '''

disc_loss = DiscriminatorLoss().to(device)

real_logits = [
    torch.randn(B, 1, 50, device = device, requires_grad = True)
    for _ in range(n_disc)
]
fake_logits = [
    torch.randn(B, 1, 50, device = device, requires_grad = True)
    for _ in range(n_disc)
]

losses_d = disc_loss(real_logits = real_logits, fake_logits = fake_logits)
print(f"Discriminator loss: {losses_d['loss'].item()}")
losses_d["loss"].backward()
print(f"Backward OK")