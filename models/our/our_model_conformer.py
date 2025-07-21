import torch
import os
import json
from collections import OrderedDict
import torch.nn.functional as F
from models.base_model import BaseModel


from models.networks.conformer_encoder import ConformerEncoder
from models.networks.classifier import FcClassifier
from models.utils.config import OptConfig
import math
import torch.nn as nn

# ✅ 新增：用于最终融合的中间 MLP 投影模块（三模型预测拼接 -> 再预测）
class EnsembleRefiner(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# ✅ Mixer 融合模块（保留）
class MixerFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.token_mixers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Linear(input_dim, input_dim)
            ) for _ in range(num_layers)
        ])
        self.channel_mixers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(num_layers)
        ])

    def forward(self, x):  # x: [B, T, D]
        for token_mixer, channel_mixer in zip(self.token_mixers, self.channel_mixers):
            x = x + token_mixer(x)
            x = x + channel_mixer(x)
        return x

# ✅ Gated Adapter（保留）
class GatedAdapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=32):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.activation = nn.ReLU()
        self.gate = nn.Sigmoid()
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        return residual + self.gate(x)

# ✅ 低秩线性层（保留）
class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=32):
        super().__init__()
        self.linear1 = nn.Linear(in_features, rank, bias=False)
        self.linear2 = nn.Linear(rank, out_features, bias=True)

    def forward(self, x):
        return self.linear2(self.linear1(x))

# ✅ LiteFusion 保留
class LiteFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_lowrank=False, lowrank_rank=32):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.depthwise_conv = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1, groups=input_dim)
        self.use_lowrank = use_lowrank

        if use_lowrank:
            self.linear1 = nn.Linear(input_dim, lowrank_rank)
            self.linear2 = nn.Linear(lowrank_rank, hidden_dim)
        else:
            self.pointwise = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x_norm = self.layer_norm(x)
        x_conv = x_norm.transpose(1, 2)
        x_conv = self.depthwise_conv(x_conv)

        if self.use_lowrank:
            B, D, T = x_conv.size()
            x_conv = x_conv.permute(0, 2, 1).contiguous().view(-1, D)
            x_conv = self.linear2(self.linear1(x_conv))
            x_conv = x_conv.view(B, T, -1).transpose(1, 2)
        else:
            x_conv = self.pointwise(x_conv)

        x_conv = self.activation(x_conv)
        x_conv = self.dropout(x_conv)
        x_conv = x_conv.transpose(1, 2)
        return self.output_proj(x_conv)

class ourModel_conformer(BaseModel, nn.Module):
    def __init__(self, opt):
        nn.Module.__init__(self)
        super().__init__(opt)

        self.loss_names = []
        self.model_names = []

        self.netEmoA = ConformerEncoder(opt.input_dim_a, opt.embd_size_a)
        self.netEmoV = ConformerEncoder(opt.input_dim_v, opt.embd_size_v)
        self.model_names += ['EmoA', 'EmoV']

        self.fusion_method = getattr(opt, 'fusion_method', 'concat')
        self.use_lowrank = getattr(opt, 'use_lowrank', False)
        self.lowrank_rank = getattr(opt, 'lowrank_rank', 32)

        fusion_input_size = opt.embd_size_a + opt.embd_size_v
        fusion_hidden_size = opt.hidden_size

        if self.fusion_method == 'gated':
            self.gate_net = nn.Sequential(
                nn.Linear(fusion_input_size, fusion_input_size),
                nn.ReLU(),
                nn.Dropout(opt.dropout_rate),
                nn.Linear(fusion_input_size, opt.embd_size_a),
                nn.Sigmoid()
            )
            fusion_out_size = opt.embd_size_a
        elif self.fusion_method == 'attention':
            self.cross_attn = nn.MultiheadAttention(embed_dim=opt.embd_size_a, num_heads=4, batch_first=True)
            fusion_out_size = opt.embd_size_a
        else:
            fusion_out_size = fusion_input_size

        self.netEmoFusion = LiteFusion(fusion_out_size, fusion_hidden_size,
                                       use_lowrank=self.use_lowrank, lowrank_rank=self.lowrank_rank)
        self.model_names.append('EmoFusion')

        cls_layers = list(map(int, opt.cls_layers.split(',')))
        cls_input_size = opt.feature_max_len * fusion_out_size + 1024

        self.netEmoC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.netEmoCF = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.model_names += ['EmoC', 'EmoCF']
        self.loss_names += ['emo_CE', 'EmoF_CE']

        # ✅ 新增 ensemble-refiner 模块：用于后期中间模型训练
        self.ensemble_refiner = EnsembleRefiner(input_dim=3 * opt.emo_output_dim, hidden_dim=64, output_dim=opt.emo_output_dim)

        self.temperature = opt.temperature
        self.criterion_ce = torch.nn.CrossEntropyLoss()

        if self.isTrain:
            if not opt.use_ICL:
                self.criterion_focal = torch.nn.CrossEntropyLoss()
            else:
                self.criterion_focal = Focal_Loss()

            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            paremeters += [{'params': self.ensemble_refiner.parameters()}]  # 加上新结构

            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.ce_weight = opt.ce_weight
            self.focal_weight = opt.focal_weight

        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)

    def set_input(self, input):
        self.acoustic = input['A_feat'].float().to(self.device)
        self.visual = input['V_feat'].float().to(self.device)
        self.emo_label = input['emo_label'].to(self.device)
        self.personalized = input.get('personalized_feat', None)
        if self.personalized is not None:
            self.personalized = self.personalized.float().to(self.device)

    def forward(self, acoustic_feat=None, visual_feat=None):
        if acoustic_feat is not None:
            self.acoustic = acoustic_feat.float().to(self.device)
            self.visual = visual_feat.float().to(self.device)

        emo_feat_A = self.netEmoA(self.acoustic)
        emo_feat_V = self.netEmoV(self.visual)

        if self.fusion_method == 'gated':
            fusion_input = torch.cat((emo_feat_A, emo_feat_V), dim=-1)
            gate = self.gate_net(fusion_input)
            emo_fusion_feat = gate * emo_feat_A + (1 - gate) * emo_feat_V
        elif self.fusion_method == 'attention':
            emo_fusion_feat, _ = self.cross_attn(emo_feat_A, emo_feat_V, emo_feat_V)
        else:
            emo_fusion_feat = torch.cat((emo_feat_V, emo_feat_A), dim=-1)

        emo_fusion_feat = self.netEmoFusion(emo_fusion_feat)
        batch_size = emo_fusion_feat.size(0)
        emo_fusion_feat = emo_fusion_feat.permute(1, 0, 2).reshape(batch_size, -1)

        if self.personalized is not None:
            emo_fusion_feat = torch.cat((emo_fusion_feat, self.personalized), dim=-1)

        self.emo_logits_fusion, _ = self.netEmoCF(emo_fusion_feat)
        self.emo_logits, _ = self.netEmoC(emo_fusion_feat)
        self.emo_pred = F.softmax(self.emo_logits, dim=-1)

        # # ✅ 新增：用于训练中间融合器模型的输出（拼接三模型概率）
        # if hasattr(self, 'external_logits_all'):
        #     # 期望格式：List[Tensor shape=(B, num_cls)]，拼接后 (B, 3*num_cls)
        #     concat_probs = torch.cat(self.external_logits_all, dim=1)  # [B, 3*C]
        #     self.ensemble_logits = self.ensemble_refiner(concat_probs)
        #     self.ensemble_pred = F.softmax(self.ensemble_logits, dim=-1)

    def backward(self):
        self.loss_emo_CE = self.criterion_ce(self.emo_logits, self.emo_label)
        self.loss_EmoF_CE = self.focal_weight * self.criterion_focal(self.emo_logits_fusion, self.emo_label)
        loss = self.loss_emo_CE + self.loss_EmoF_CE

        if hasattr(self, 'ensemble_logits'):
            self.loss_ensemble = self.criterion_ce(self.ensemble_logits, self.emo_label)
            loss += self.loss_ensemble

        loss.backward()

        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 1.0)

    def optimize_parameters(self, epoch):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()


class ActivateFun(torch.nn.Module):
    def __init__(self, opt):
        super(ActivateFun, self).__init__()
        self.activate_fun = opt.activate_fun

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == 'relu':
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            return self._gelu(x)


class Focal_Loss(torch.nn.Module):
    def __init__(self, weight=0.5, gamma=3, reduction='mean'):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.alpha = weight
        self.reduction = reduction

    def forward(self, preds, targets):
        ce_loss = F.cross_entropy(preds, targets, reduction='mean')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            raise NotImplementedError("Invalid reduction mode. Please choose 'none', 'mean', or 'sum'.")
