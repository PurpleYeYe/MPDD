import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioAttNet(nn.Module):
    def __init__(self, dim_aud=64, seq_len=9):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = nn.Sequential(
            nn.Conv1d(self.dim_aud, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.02, inplace=True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(self.seq_len, self.seq_len),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [B, seq_len, dim_aud]
        y = x.permute(0, 2, 1)  # [B, dim_aud, seq_len]
        y = self.attentionConvNet(y)  # [B, 1, seq_len]
        y = self.attentionNet(y.view(x.size(0), self.seq_len))  # [B, seq_len]
        y = y.view(x.size(0), self.seq_len, 1)  # [B, seq_len, 1]
        return torch.sum(y * x, dim=1)  # [B, dim_aud]
