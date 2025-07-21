import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, features, labels):
        """
        features: (B, D) - batch_size x embedding_dim
        labels: (B,) - int class labels

        同类为正样本，不同类为负样本。
        """
        device = features.device
        batch_size = features.shape[0]

        sim_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)  # (B, B)
        sim_matrix = sim_matrix / self.temperature

        labels = labels.contiguous().view(-1, 1)  # (B, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # (B, B) 同类为1，异类为0

        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()  # 稳定性优化

        exp_logits = torch.exp(logits) * (1 - torch.eye(batch_size, device=device))  # 去掉对角线
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

        loss = -mean_log_prob_pos.mean()
        return loss
