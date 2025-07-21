# attention_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(HybridAttention, self).__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        x: Tensor [T, D] — 一段音频或视频的帧级特征
        return: Tensor [D] — attention 加权后的融合表示
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # 变成 [1, T, D]
        attn_scores = self.attn_mlp(x)              # [1, T, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # [1, T, 1]
        weighted_sum = torch.sum(attn_weights * x, dim=1)  # [1, D]
        return weighted_sum.squeeze(0)  # [D]
