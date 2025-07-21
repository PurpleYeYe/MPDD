import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        # 稀疏损失项
        sparsity_loss = torch.mean(torch.abs(z))
        return x_hat, sparsity_loss
