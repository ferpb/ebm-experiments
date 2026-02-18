import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision


def sample_gaussian_mixture(n):
    p, std = 0.3, 0.2
    out = torch.randn(n, 1) * std
    out = out + torch.sign(torch.rand(n, 1) - p) / 2
    return out


def sample_ramp(n):
    out = torch.min(torch.rand(n, 1), torch.rand(n, 1))
    return out


def sample_three_discs(n):
    c = torch.tensor([[2, 2], [-2, 2], [0, -2]])
    idx = torch.randint(0, len(c), (n,))
    out = c[idx] + 0.3 * torch.randn(n, 2)
    return out


def sample_swiss_roll(n):
    u = torch.rand(1000)
    t = u * 4 * torch.pi
    s = u + 0.1 * torch.rand(1000)
    x = torch.cos(t) * s
    y = torch.sin(t) * s
    out = torch.stack([x, y], dim=1)
    return out


def sample_mnist(n):
    train_data = torchvision.datasets.mnist(root="./data", train=True, download=True)
    out = train_data.data[:n] / 255.0
    return out


samplers = {
    f.__name__.removeprefix("sample_"): f
    for f in [
        sample_gaussian_mixture,
        sample_ramp,
        sample_three_discs,
        sample_swiss_roll,
        sample_mnist
    ]
}
