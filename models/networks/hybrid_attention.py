import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
        """
        半软 + 半硬注意力机制模块
        Args:
            input_dim (int): 输入特征维度 D
            hidden_dim (int): 注意力 MLP 的隐藏层维度
            dropout (float): 注意力分数的 dropout，用于正则化
        """
        super(HybridAttention, self).__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, T, D]，B=batch size, T=帧数, D=特征维度

        Returns:
            enhanced: [B, T, D] — 每一帧特征加入全局信息
            attn_weights: [B, T] — 每一帧的注意力权重（可用于可视化）
        """
        B, T, D = x.size()

        attn_score = self.attn_mlp(x)         # [B, T, 1]
        attn_weights = F.softmax(attn_score, dim=1)  # [B, T, 1]
        attn_weights = self.dropout(attn_weights)

        global_feat = torch.sum(attn_weights * x, dim=1, keepdim=True)  # [B, 1, D]

        enhanced = x + global_feat  # 残差增强 [B, T, D]

        return enhanced, attn_weights.squeeze(-1)  # [B, T, D], [B, T]
