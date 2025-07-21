import torch
import torch.nn as nn
import torch.nn.functional as F

class LightFrameAttention(nn.Module):
    def __init__(self, input_dim, dropout=0.2):
        super(LightFrameAttention, self).__init__()
        self.attn_proj = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        attn_score = self.attn_proj(x).squeeze(-1)  # [B, T]
        attn_weights = F.softmax(attn_score, dim=-1)  # [B, T]
        attn_weights = self.dropout(attn_weights.unsqueeze(-1))  # [B, T, 1]

        out = torch.sum(attn_weights * x, dim=1)  # [B, D]
        return out, attn_weights.squeeze(-1)
