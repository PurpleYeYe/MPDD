import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardModule(nn.Module):
    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion_factor),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ConvolutionModule(nn.Module):
    def __init__(self, dim, kernel_size=31, dropout=0.1):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.batch_norm = nn.BatchNorm1d(dim)
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GLU(dim=1)

    def forward(self, x):
        # x: [B, T, D]
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.pointwise_conv1(x)
        x = self.activation(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)  # [B, T, D]
        return self.dropout(x)

class ConformerBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=4, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForwardModule(dim, dropout=dropout)
        self.ff2 = FeedForwardModule(dim, dropout=dropout)
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.ln_attn = nn.LayerNorm(dim)
        self.conv_module = ConvolutionModule(dim, dropout=dropout)
        self.ln_conv = nn.LayerNorm(dim)
        self.ln_final = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        residual = x
        x = self.ln_attn(x)
        x, _ = self.self_attn(x, x, x)
        x = residual + x
        x = x + self.conv_module(self.ln_conv(x))
        x = x + 0.5 * self.ff2(x)
        return self.ln_final(x)

class ConformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            ConformerBlock(dim=hidden_dim, dim_head=hidden_dim//nhead, heads=nhead, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.input_proj(x)  # [B, T, H]
        for layer in self.layers:
            x = layer(x)        # [B, T, H]
        return x
