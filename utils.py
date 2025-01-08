import torch


def noise(batch_size, latent_dim):
    return torch.randn(batch_size, latent_dim)
