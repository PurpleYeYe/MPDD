import torch
import torch.nn as nn

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=32, bias=True):
        super(LowRankLinear, self).__init__()
        self.rank = rank
        self.low1 = nn.Linear(in_features, rank, bias=False)
        self.low2 = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x):
        return self.low2(self.low1(x))