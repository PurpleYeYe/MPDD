import torch
import torch.nn as nn
import torch.nn.functional as F

class LightVisualFrameAttention(nn.Module):
    def __init__(self, input_dim, dropout=0.2):
        super(LightVisualFrameAttention, self).__init__()
        self.attn_proj = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Tensor [B, T, D] - 多帧视频特征输入
        Returns:
            pooled: Tensor [B, D] - 权重平均后的视频特征
            attn_weights: Tensor [B, T] - 每帧注意力权重
        """
        attn_score = self.attn_proj(x).squeeze(-1)  # [B, T]
        attn_weights = F.softmax(attn_score, dim=1)  # [B, T]
        attn_weights = self.dropout(attn_weights.unsqueeze(-1))  # [B, T, 1]
        pooled = torch.sum(attn_weights * x, dim=1)  # [B, D]
        return pooled, attn_weights.squeeze(-1)
