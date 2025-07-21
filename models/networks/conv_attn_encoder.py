import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAttnEncoder(nn.Module):
    def __init__(self, input_dim, embd_size, kernel_size=3, num_layers=2, pool_type='attention'):
        super(ConvAttnEncoder, self).__init__()

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_dim, embd_size, kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(embd_size))
            layers.append(nn.ReLU())
            in_dim = embd_size

        self.conv = nn.Sequential(*layers)
        self.pool_type = pool_type

        if pool_type == 'attention':
            self.attn_fc = nn.Linear(embd_size, 1)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        x = x.permute(0, 2, 1)  # [B, D, T]
        x = self.conv(x)         # [B, embd_size, T]
        x = x.permute(0, 2, 1)   # [B, T, embd_size]

        if self.pool_type == 'attention':
            attn_score = self.attn_fc(x)  # [B, T, 1]
            attn_weights = F.softmax(attn_score, dim=1)  # [B, T, 1]
            out = torch.sum(attn_weights * x, dim=1)  # [B, embd_size]
            out = out.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, T, embd_size]
        elif self.pool_type == 'mean':
            out = x.mean(dim=1, keepdim=True).expand(-1, x.size(1), -1)
        else:
            raise NotImplementedError

        return out