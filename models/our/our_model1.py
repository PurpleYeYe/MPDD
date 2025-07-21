import torch
import os
import json
from collections import OrderedDict
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.lstm import LSTMEncoder
from models.networks.classifier import FcClassifier
from models.networks.audio_attention import AudioAttNet
from models.networks.conv_attn_encoder import ConvAttnEncoder
from models.utils.config import OptConfig
import math
import torch.nn as nn


class ourModel(BaseModel, nn.Module):
    """
    这段代码是你项目的情感识别模型类 ourModel 的构造函数 __init__()，定义了模型的完整结构和训练配置。
    它整合了音频、视觉、个性化特征、Transformer 融合模块和多个分类器，并配置了损失函数与优化器。
    BaseModel：你项目中定义的基础模型类（封装了训练框架）；
    nn.Module：PyTorch 的基础模型类。
    """
    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        nn.Module.__init__(self)# 显式调用 nn.Module 的构造函数，注册模块；
        super().__init__(opt)# 调用 BaseModel.__init__(opt)，传入配置，初始化模型基类逻辑（如 self.device, self.isTrain 等）。

        
        self.loss_names = []# loss_names：损失函数的名字；
        self.model_names = []# model_names：模型组件的名字（用于保存/加载模型）。

        # acoustic model
        """
        创建音频编码器
        LSTMEncoder 是音频特征的时序编码器（使用 LSTM）；
        输入维度 input_dim_a 来自 .npy 文件 shape；
        输出维度 embd_size_a；
        使用方法如 mean, last, attn 等聚合策略。
        """
        self.netEmoA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
        self.model_names.append('EmoA')

        # visual model
        """
        创建视觉编码器
        同理，用于编码视频特征；
        输入维度 input_dim_v，输出维度 embd_size_v，使用 embd_method_v 聚合方式。
        """
        self.netEmoV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
        self.model_names.append('EmoV')
        self.audio_attention = AudioAttNet(
            dim_aud=opt.input_dim_a,  # 例如 64
            seq_len=5  # 对应你对齐的音频帧窗口
        ).to(self.device)
        # Transformer Fusion model
        """
        创建 Transformer 融合模块
        用于融合音频和视频的时序特征；
        参数来自配置文件：
        hidden_size：每个时间步的特征维度；
        Transformer_head：多头注意力数量；
        Transformer_layers：Transformer 层数；
        输出维度依旧为 batch_size × seq_len × hidden_size。
        """
        emo_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=opt.hidden_size, nhead=int(opt.Transformer_head), batch_first=True)
        self.netEmoFusion = torch.nn.TransformerEncoder(emo_encoder_layer, num_layers=opt.Transformer_layers)
        self.model_names.append('EmoFusion')

        # Classifier
        """
        分类器网络
        分类器是一个多层全连接网络：
        层结构来自字符串 "128,64" → [128, 64]；
        输入特征维度 = seq_len × hidden_size + 1024（拼接个性化向量）；
        feature_max_len 是时间步数；
        1024 是个性化嵌入向量的维度。
        """
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        # cls_input_size = 5*opt.hidden_size, same with max-len
        cls_input_size = opt.feature_max_len * opt.hidden_size + 1024  # with personalized feature

        """
        主分类器（EmoC）
        输出维度是分类的类别数；
        使用交叉熵计算主要损失 emo_CE。
        """
        self.netEmoC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('EmoC')
        self.loss_names.append('emo_CE')
        """
        辅助分类器（EmoCF）
        通常与 Focal Loss 搭配使用，针对类别不平衡；
        是一种增强策略，也可能用于 ICL（Instance Contrastive Learning）结构。
        """
        self.netEmoCF = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('EmoCF')
        self.loss_names.append('EmoF_CE')
        """
        置温度参数
        通常用于 softmax 温度缩放（例如在 contrastive learning 或蒸馏中）；
        后续模型中可能用于调节 logits。
        """
        self.temperature = opt.temperature


        # self.device = 'cpu'
        # self.netEmoA = self.netEmoA.to(self.device)
        # self.netEmoV = self.netEmoV.to(self.device)
        # self.netEmoFusion = self.netEmoFusion.to(self.device)
        # self.netEmoC = self.netEmoC.to(self.device)
        # self.netEmoCF = self.netEmoCF.to(self.device)
        """
        设置损失函数
        默认使用标准交叉熵。
        """
        self.criterion_ce = torch.nn.CrossEntropyLoss()
        """
        训练模式下：初始化优化器、损失函数等
        如果启用了 ICL，则使用自定义的 Focal_Loss()；
        否则直接使用两次标准交叉熵；
        ce_weight, focal_weight 控制最终损失中的加权比例。
        """
        if self.isTrain:
            if not opt.use_ICL:
                self.criterion_ce = torch.nn.CrossEntropyLoss()
                self.criterion_focal = torch.nn.CrossEntropyLoss() 
            else:
                self.criterion_ce = torch.nn.CrossEntropyLoss()
                self.criterion_focal = Focal_Loss()
            """
            创建优化器
            收集所有子模块的参数；
            使用 Adam 优化器；
            参数：学习率、beta1 从配置文件中获取。
            """
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.ce_weight = opt.ce_weight
            self.focal_weight = opt.focal_weight

        # modify save_dir
        """
        设置模型保存路径
        设置模型权重保存路径；
        如果不存在则创建目录。
        """
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name) 
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)  


    def post_process(self):
        """
        从预训练模型中加载编码器（音频、视觉、融合）的参数，并适配到当前模型中。
        这个函数在模型搭建完毕后由外部流程（比如 BaseModel.setup()）调用；
        它并不是训练过程中每轮都会执行的，而是一次性的初始化辅助操作。
        """
        # called after model.setup()
        """
        嵌套函数：键名转换
        为什么加 'module.' 前缀？
        当使用 torch.nn.DataParallel 或 DistributedDataParallel 并行训练时，模型的参数名字会多一个 'module.' 前缀；
        所以加载单卡保存的模型到多卡模型，或相反时需要手动处理参数名对齐。
        这个函数就是把参数名加上 'module.'，适配这种情况。
        """
        def transform_key_for_parallel(state_dict):
            return OrderedDict([('module.' + key, value) for key, value in state_dict.items()])
        """
        条件判断：仅在训练模式下使用
        只有训练模式下（如在 train.py 中）才会加载预训练模型；
        推理/测试阶段则不需要重复初始化。
        """
        if self.isTrain:
            print('[ Init ] Load parameters from pretrained encoder network')
            """
            关键步骤：加载子网络的预训练参数
            这些模块分别是：
            netEmoA：音频 LSTM 编码器；
            netEmoV：视频 LSTM 编码器；
            netEmoFusion：Transformer 融合模块。
            这里的 self.pretrained_encoder 是在哪定义的？
            👉 它应当是你在 BaseModel 或其他初始化过程中指定的一个已加载的预训练模型实例，包含这些模块的参数。
            """
            f = lambda x: transform_key_for_parallel(x)
            self.netEmoA.load_state_dict(f(self.pretrained_encoder.netEmoA.state_dict()))
            self.netEmoV.load_state_dict(f(self.pretrained_encoder.netEmoV.state_dict()))
            self.netEmoFusion.load_state_dict(f(self.pretrained_encoder.netEmoFusion.state_dict()))

    def load_from_opt_record(self, file_path):
        """
        该方法用于从 JSON 文件中读取模型配置（opt）记录，并还原成 OptConfig 对象。
        使用场景：
        这个方法可以让你在训练后重新加载保存的超参数配置，比如用于推理、复现实验等；
        常与保存的模型权重搭配，用于构造模型的完整状态（结构 + 参数 + 配置）；
        OptConfig 应该是 models.utils.config 中定义的类，用于统一存储和管理配置参数。
        """
        opt_content = json.load(open(file_path, 'r'))# 载入以 JSON 格式保存的超参数配置；文件内容通常是 dict，记录了模型结构、维度、学习率等信息。
        opt = OptConfig()
        opt.load(opt_content)
        return opt

    def set_input(self, input):
        """
        该方法是训练/测试时模型接收一个 batch 数据的入口，主要是将输入数据放入模型的变量中，并移动到目标设备（GPU/CPU）。
        input 是一个 字典，来自 AudioVisualDataset 的 __getitem__()；
        包含键：
        'A_feat'：音频特征张量 (batch_size, T, D)
        'V_feat'：视频特征张量 (batch_size, T, D)
        'emo_label'：分类标签（整数）
        'personalized_feat'：个性化嵌入 (batch_size, 1024)，可选
        """
        self.acoustic = input['A_feat'].float().to(self.device)# 设置音频特征
        self.visual = input['V_feat'].float().to(self.device)# 设置视频特征

        self.emo_label = input['emo_label'].to(self.device)# 设置情感标签

        if 'personalized_feat' in input:# 处理个性化特征（可选）
            self.personalized = input['personalized_feat'].float().to(self.device)
        else:
            self.personalized = None  # if no personalized features given

    def forward(self, acoustic_feat=None, visual_feat=None):
        if acoustic_feat is not None:
            self.acoustic = acoustic_feat.float().to(self.device)
            self.visual = visual_feat.float().to(self.device)

        # acoustic: [B, T, 9, D]
        B, T, W, D = self.acoustic.shape
        acoustic_list = []

        for i in range(T):
            frame_audio = self.acoustic[:, i, :, :]  # [B, 9, D]
            fused = self.audio_attention(frame_audio)  # [B, D]
            acoustic_list.append(fused)

        acoustic_aligned = torch.stack(acoustic_list, dim=1)  # [B, T, D]

        # 编码音频和视频
        emo_feat_A = self.netEmoA(acoustic_aligned)  # [B, T, embd_size]
        emo_feat_V = self.netEmoV(self.visual)  # [B, T, embd_size]

        # 融合
        emo_fusion_feat = torch.cat((emo_feat_V, emo_feat_A), dim=-1)  # [B, T, 2*embd_size]
        emo_fusion_feat = self.netEmoFusion(emo_fusion_feat)  # [B, T, hidden]

        # reshape
        batch_size = emo_fusion_feat.size(0)
        emo_fusion_feat = emo_fusion_feat.permute(1, 0, 2).reshape(batch_size, -1)  # [B, T*hidden]

        # 拼接个性化特征
        if self.personalized is not None:
            emo_fusion_feat = torch.cat((emo_fusion_feat, self.personalized), dim=-1)  # [B, T*hidden + 1024]

        # 分类器输出
        self.emo_logits_fusion, _ = self.netEmoCF(emo_fusion_feat)
        self.emo_logits, _ = self.netEmoC(emo_fusion_feat)
        self.emo_pred = F.softmax(self.emo_logits, dim=-1)

    def backward(self):
        """
        这段 backward() 方法定义的是模型的反向传播阶段，它完成以下任务：
        ✅ 计算损失 → ✅ 合成总损失 → ✅ 反向传播 → ✅ 梯度裁剪（防止梯度爆炸）
        在 PyTorch 中，backward() 是训练循环中的核心步骤之一；
        通常会在调用 forward() 后执行，以便反向传播误差、更新模型权重。
        """
        """Calculate the loss for back propagation"""
        self.loss_emo_CE = self.criterion_ce(self.emo_logits, self.emo_label)
        """
        计算主损失：交叉熵损失（主分类器）
        self.emo_logits 是主分类器 netEmoC 的输出；
        self.emo_label 是 ground truth 标签；
        使用 CrossEntropyLoss() 计算分类损失；
        这是标准的多分类 loss。
        """
        self.loss_EmoF_CE = self.focal_weight * self.criterion_focal(self.emo_logits_fusion, self.emo_label)
        """
        计算辅助损失：Focal Loss（增强模型对小类/困难样本的关注）
        self.emo_logits_fusion 是辅助分类器 netEmoCF 的输出；
        self.criterion_focal 是 focal loss 实例（用于类别不平衡场景）；
        self.focal_weight 是一个超参数，用来调节 focal loss 在总损失中的权重；
        这部分增强模型对难以预测或小样本类别的学习能力。
        """
        loss = self.loss_emo_CE + self.loss_EmoF_CE
        loss.backward()
        """
        合成总损失并反向传播
        总损失是主分类器损失 + 辅助分类器损失；
        调用 .backward() 自动计算每个模型参数的梯度（autograd）。
        """
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 1.0)
        """
        梯度裁剪（防止梯度爆炸）
        遍历模型名称列表（['EmoA', 'EmoV', 'EmoFusion', 'EmoC', 'EmoCF']）；
        getattr(self, 'net' + model) 动态获取对应模型子模块；
        使用 clip_grad_norm_() 限制所有参数的梯度 L2 范数不超过 1.0；
        这是防止梯度爆炸的一种常见技术；
        特别是在使用 LSTM、Transformer 时容易遇到此问题。
        """

    def optimize_parameters(self, epoch):
        """
        这段 optimize_parameters() 是你模型训练流程中的核心执行函数，每次训练迭代（即每个 batch）都会调用它。
        它负责：前向传播 → 清零梯度 → 反向传播 → 更新权重
        即完成一整个 “前传 + 反传 + 优化” 过程。
        参数 epoch：当前训练的 epoch 编号（虽然在本函数内没有直接用，但可能用于内部记录、调试或日志打印）；
        这个函数通常由外部训练循环 for epoch in range(...) 每次调用。
        """
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward前向传播
        self.forward()
        """
        调用模型的 forward() 方法：
        编码音频+视频特征；
        Transformer 融合；
        拼接个性化向量；
        分类器输出 logits；
        计算 softmax 得到预测概率（self.emo_pred）；
        forward() 内部已经将 self.emo_logits, self.emo_logits_fusion 等准备好供后续使用。
        """
        # backward
        self.optimizer.zero_grad()# 梯度清零,清空上一步残留的梯度；
        self.backward()
        """
        反向传播
        调用 self.backward()：
        计算主损失（交叉熵）和辅助损失（Focal Loss）；
        合并两个损失；
        .backward() 执行 autograd 计算所有参数的梯度；
        使用 clip_grad_norm_() 对所有模块梯度裁剪，防止梯度爆炸。
        """
        self.optimizer.step()# 更新参数,使用之前注册的 Adam 优化器：将刚刚计算好的梯度用于更新模型中的参数。


class ActivateFun(torch.nn.Module):
    """
    你这段代码定义了两个类，分别封装了：
    ✅ 自定义激活函数模块 ActivateFun
    ✅ 改良版分类损失函数 Focal_Loss（处理类别不平衡问题）
    """
    def __init__(self, opt):
        super(ActivateFun, self).__init__()
        self.activate_fun = opt.activate_fun
    """
    ActivateFun：根据配置动态选择激活函数
    接收一个配置对象 opt；
    opt.activate_fun 是字符串（如 "relu"、"gelu"），指示使用哪种激活函数；
    将激活函数类型记录在 self.activate_fun 中。
    """
    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    """
    内部定义的 GELU 函数（不是默认 API）
    这是 GELU（Gaussian Error Linear Unit）的精确公式版本；
    比 ReLU 更平滑，在某些任务（如 NLP、BERT）中表现更优。
    """
    def forward(self, x):
        if self.activate_fun == 'relu':
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            return self._gelu(x)
    """
    forward()：根据配置选择激活函数执行
    根据配置返回 ReLU 或 GELU 激活；
    如果以后你想扩展更多激活函数（如 LeakyReLU、Swish），可以很方便加进来。
    """

class Focal_Loss(torch.nn.Module):
    def __init__(self, weight=0.5, gamma=3, reduction='mean'):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.alpha = weight
        self.reduction = reduction
    """
    Focal_Loss：处理类别不平衡的改良损失函数
    参数	            作用
    weight（或 α）	类别权重（对正类或困难样本的强调）
    gamma	        调节关注难易样本的程度
    reduction	    输出方式（'mean' / 'sum' / 'none'）
    Focal Loss 是为了解决类别不平衡问题，在分类任务中对容易分类的样本降低权重，对困难样本提高权重。
    """
    def forward(self, preds, targets):
        """
        preds:softmax output
        labels:true values
        """
        ce_loss = F.cross_entropy(preds, targets, reduction='mean')# 计算普通交叉熵损失
        pt = torch.exp(-ce_loss)# 计算 pt = softmax 概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss# 构造 Focal Loss

        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            raise NotImplementedError("Invalid reduction mode. Please choose 'none', 'mean', or 'sum'.")
        """
        none：返回每个样本的 loss；
        mean（默认）：求平均；
        sum：求和；
        """