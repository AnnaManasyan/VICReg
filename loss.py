import torch.nn as nn
import torch
import torch.nn.functional as F

# invariance loss
sim_loss = nn.MSELoss()

# variance loss
def std_loss(z_a, z_b):
    std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
    std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
    std_loss = torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))
    return std_loss


#function taken from https://github.com/facebookresearch/barlowtwins/tree/a655214c76c97d0150277b85d16e69328ea52fd9
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

# covariance loss
def cov_loss(z_a, z_b):
    N = z_a.shape[0]
    D = z_a.shape[1]
    z_a = z_a - z_a.mean(dim=0)
    z_b = z_b - z_b.mean(dim=0)
    cov_z_a = (z_a.T @ z_a) / (N - 1)
    cov_z_b = (z_b.T @ z_b) / (N - 1)
    cov_loss = off_diagonal(cov_z_a).pow_(2).sum() / D + off_diagonal(cov_z_b).pow_(2).sum() / D
    return cov_loss



