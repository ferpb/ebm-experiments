import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import tasks


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


device = torch.device("cuda")
model = MLP().to(device)
model.train()

sample_data = tasks.samplers["three_discs"]


# training with contrastive divergence

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-8)

def langevin_step(x, step=0.05):
    x = x.detach().requires_grad_(True).to(device)
    energy = model(x).sum()
    grad = torch.autograd.grad(energy, x)[0]
    noise = torch.randn_like(x)
    x_new = x - 0.5 * step * grad + (step ** 0.5) * noise
    return x_new.detach()


batch_size = 256
steps = 3000

cd_steps = 20
langevin_step_size = 0.05

# replay buffer
buffer_size = 10000
x_neg_buffer = torch.randn(buffer_size, 2).to(device)


for step in range(steps):
    # positive data
    x_pos = sample_data(batch_size).to(device)

    # take a batch from the replay buffer
    idx = torch.randint(0, buffer_size, (batch_size,))
    x_neg = x_neg_buffer[idx]

    for _ in range(cd_steps):
        x_neg = langevin_step(x_neg, step=langevin_step_size)

    # update the buffer with new samples
    x_neg_buffer[idx] = x_neg

    energy_pos = model(x_pos) # we want it to be low
    energy_neg = model(x_neg) # we want it to be high

    reg = 1e-5 * (energy_pos.pow(2).mean() + energy_neg.pow(2).mean())
    loss = energy_pos.mean() - energy_neg.mean() + reg

    margin = energy_neg.mean() - energy_pos.mean()

    optimizer.zero_grad()
    loss.backward()

    # gradient clipping
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

    optimizer.step()

    if (step + 1) % 100 == 0:
        print(f"step {step+1}/{steps}, loss {loss.item():.4f}, margin {margin.item():.4f}")


# plotting

model.eval()
samples = sample_data(1000)

# vector field

grid_x, grid_y = torch.meshgrid(
    torch.linspace(-4, 4, 25, dtype=torch.float32),
    torch.linspace(-4, 4, 25, dtype=torch.float32),
    indexing="ij",
)

grid_points = torch.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
grid_points = grid_points.to(device).requires_grad_(True)

energy_grid = model(grid_points).sum()
grad_grid = torch.autograd.grad(energy_grid, grid_points)[0]
vec_field = -grad_grid.detach().cpu()

u = vec_field[:, 0].reshape(grid_x.shape)
v = vec_field[:, 1].reshape(grid_y.shape)

plt.figure(figsize=(6, 6))
plt.quiver(grid_x, grid_y, u, v)
plt.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5)
plt.xlabel("x1")
plt.ylabel("x2")
plt.savefig("vector_field.png")

# energy landscape

grid_x, grid_y = torch.meshgrid(
    torch.linspace(-4, 4, 100, dtype=torch.float32),
    torch.linspace(-4, 4, 100, dtype=torch.float32),
    indexing="ij",
)

grid_points = torch.stack([grid_x.ravel(), grid_y.ravel()], dim=-1)
grid_points = grid_points.to(device)

with torch.no_grad():
    energy_vals = model(grid_points)

energy_vals = energy_vals.reshape(grid_x.shape).cpu()

plt.figure(figsize=(8, 6))
cs = plt.contourf(grid_x, grid_y, energy_vals, levels=30)
plt.colorbar(cs, label="E(x)")
plt.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5, color="k")
plt.xlabel("x1")
plt.xlabel("x2")
plt.axis("equal")
plt.savefig("energy_landscape.png")

# discrete sampling

with torch.no_grad():
    energy = model(grid_points)
    logits = -energy # p propto exp(-E)
    probs = torch.softmax(logits, dim=0)

idx = torch.multinomial(probs, num_samples=1000, replacement=True)
new_samples = grid_points[idx].cpu()

plt.figure(figsize=(6, 6))
plt.scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.5, color="k", label="train")
plt.scatter(new_samples[:, 0], new_samples[:, 1], s=4, alpha=0.5, color="r", label="generated")
plt.xlabel("x1")
plt.xlabel("x2")
plt.legend()
plt.savefig("samples_discrete.png")

# langevin sampling

new_samples = torch.randn(1000, 2, device=device)

for _ in range(300):
    new_samples = langevin_step(new_samples, langevin_step_size)

new_samples = new_samples.cpu()

plt.figure(figsize=(6, 6))
plt.scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.5, color="k", label="train")
plt.scatter(new_samples[:, 0], new_samples[:, 1], s=4, alpha=0.5, color="r", label="generated")
plt.xlabel("x1")
plt.xlabel("x2")
plt.legend()
plt.savefig("samples_langevin.png")
