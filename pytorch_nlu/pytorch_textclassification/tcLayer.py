# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/7/24 21:45
# @author  : Mo
# @function: Layer and Loss


from torch import nn
import numpy as np
import torch


__all__ = ["PriorMultiLabelSoftMarginLoss",
           "LabelSmoothingCrossEntropyV2",
           "LabelSmoothingCrossEntropy",
           "MultiLabelCircleLoss",
           "FocalLoss",
           "DiceLoss",
           "FCLayer",
           "Swish",
           "Mish",
           ]


class PriorMultiLabelSoftMarginLoss(nn.Module):
    def __init__(self, prior=None, num_labels=None, reduction="mean", eps=1e-9, tau=1.0):
        """PriorCrossEntropy
        categorical-crossentropy-with-prior
        urls: [通过互信息思想来缓解类别不平衡问题](https://spaces.ac.cn/archives/7615)
        args:
            prior: List<float>, prior of label, 先验知识.  eg. [0.6, 0.2, 0.1, 0.1]
            num_labels: int, num of labels, 类别数.  eg. 10
            reduction: str, Specifies the reduction to apply to the output, 输出形式. 
                            eg.``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``
            eps: float, Minimum of maths, 极小值.  eg. 1e-9
            tau: float, weight of prior in loss, 先验知识的权重, eg. ``1.0``
        returns:
            Tensor of loss.
        examples:
        >>> loss = PriorCrossEntropy(prior)(logits, label)
        """
        super(PriorMultiLabelSoftMarginLoss, self).__init__()
        self.loss_mlsm = torch.nn.MultiLabelSoftMarginLoss(reduction=reduction)
        if not prior: prior = np.array([1/num_labels for _ in range(num_labels)])  # 如果不存在就设置为num
        if type(prior) ==list: prior = np.array(prior)
        self.log_prior = torch.tensor(np.log(prior + eps)).unsqueeze(0)
        self.eps = eps
        self.tau = tau

    def forward(self, logits, labels):
        # 使用与输入label相同的device
        logits = logits + self.tau * self.log_prior.to(labels.device)
        loss = self.loss_mlsm(logits, labels)
        return loss


class LabelSmoothingCrossEntropyV2(nn.Module):
    """ 平滑的交叉熵, LabelSommth-CrossEntropy
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    url: https://github.com/CoinCheung/pytorch-loss
    examples:
        >>> criteria = LabelSmoothingCrossEntropyV2()
        >>> logits = torch.randn(8, 19, 384, 384)  # nchw, float/half
        >>> lbs = torch.randint(0, 19, (8, 384, 384))  # nhw, int64_t
        >>> loss = criteria(logits, lbs)
    """
    def __init__(self, lb_smooth=0.1, reduction="mean", ignore_index=-100):
        super(LabelSmoothingCrossEntropyV2, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.lb_ignore = ignore_index
        self.lb_smooth = lb_smooth
        self.reduction = reduction

    def forward(self, logits, label):
        # overcome ignored label
        logits = logits.float() # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            # b.fill_(0)就表示用0填充b，是in_place操作。  input.scatter_(dim, index, src)将src中数据根据index中的索引按照dim的方向填进input中。
            label_unsq = label.unsqueeze(1)
            lb_one_hot = torch.empty_like(logits).fill_(lb_neg).scatter_(1, label_unsq, lb_pos).detach()
        logs = self.log_softmax(logits)
        loss = - torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == "mean":
            loss = loss.sum() / n_valid
        if self.reduction == "sum":
            loss = loss.sum()
        return loss


class LabelSmoothingCrossEntropyV1(nn.Module):
    def __init__(self, eps=0.1, reduction="mean", ignore_index=-100):
        """【直接smooth输入logits效果不好】LabelSmoothingCrossEntropy, no-softmax-input
        eps==0-1, 通过控制ce权重、新增后置项来处理来平滑
        urls: [pytorch | labelSmooth](https://zhuanlan.zhihu.com/p/265704145)
        args:
            ignore_index: (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient. Default: -100
            reduction: str, Specifies the reduction to apply to the output, 输出形式. 
                            eg.``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``
            eps: float, Minimum of maths, 极小值.  eg. 0.1
        returns:
            Tensor of loss.
        examples:
        >>> loss = LabelSmoothingCrossEntropyV1()(logits, label)
        """
        super(LabelSmoothingCrossEntropyV1, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, labels):  # logits --- logistic unit
        V = max(logits.size()[-1] - 1, 1)
        logits_smooth = (1 - self.eps) * logits + self.eps / V
        logits_smooth_logsigmoid = torch.nn.functional.logsigmoid(logits_smooth)
        loss = -(labels * logits_smooth_logsigmoid + (1 - labels) * logits_smooth_logsigmoid)
        loss = loss.sum(dim=1)  # / logits.size(1)  # only return N loss values
        if  "mean" == self.reduction:
            loss = loss.mean()
        elif "sum" == self.reduction:
            loss = loss.sum()
        else:
            _
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction="mean", ignore_index=-100):
        """LabelSmoothingCrossEntropy, no-softmax-input
        对logits进行smoothing, 即log_softmax后进行操作
        args:
            ignore_index: (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient. Default: -100
            reduction: str, Specifies the reduction to apply to the output, 输出形式. 
                            eg.``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``
            eps: float, Minimum of maths, 极小值.  eg. 0.1
        returns:
            Tensor of loss.
        examples:
          >>> loss = LabelSmoothingCrossEntropy()(logits, label)
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, labels):
        V = max(logits.size()[-1] - 1, 1)
        loss = (1 - self.eps) * (-(labels * torch.nn.functional.logsigmoid(logits) +
                 (1 - labels) * torch.nn.functional.logsigmoid(-logits))) + self.eps / V
        loss = loss.sum(dim=1) / logits.size(1)  # only return N loss values
        if  "mean" == self.reduction:
            loss = loss.mean()
        elif "sum" == self.reduction:
            loss = loss.sum()
        else:
            _
        return loss


class MultiLabelCircleLoss(nn.Module):
    def __init__(self, reduction="mean", inf=1e12):
        """CircleLoss of MultiLabel, 多个目标类的多标签分类场景，希望“每个目标类得分都不小于每个非目标类的得分”
        多标签分类的交叉熵(softmax+crossentropy推广, N选K问题), LSE函数的梯度恰好是softmax函数
        让同类相似度与非同类相似度之间拉开一定的margin。
          - 使同类相似度比最大的非同类相似度更大。
          - 使最小的同类相似度比最大的非同类相似度更大。
          - 所有同类相似度都比所有非同类相似度更大。
        urls: [将“softmax+交叉熵”推广到多标签分类问题](https://spaces.ac.cn/archives/7359)
        args:
            reduction: str, Specifies the reduction to apply to the output, 输出形式. 
                            eg.``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``
            inf: float, Minimum of maths, 无穷大.  eg. 1e12
        returns:
            Tensor of loss.
        examples:
            >>> label, logits = [[1, 1, 1, 1], [0, 0, 0, 1]], [[0, 1, 1, 0], [1, 0, 0, 1],]
            >>> label, logits = torch.tensor(label).float(), torch.tensor(logits).float()
            >>> loss = MultiLabelCircleLoss()(logits, label)
        """
        super(MultiLabelCircleLoss, self).__init__()
        self.reduction = reduction
        self.inf = inf  # 无穷大

    def forward(self, logits, labels):
        logits = (1 - 2 * labels) * logits              # <3, 4>
        logits_neg = logits - labels * self.inf         # <3, 4>, 减去选中多标签的index
        logits_pos = logits - (1 - labels) * self.inf   # <3, 4>, 减去其他不需要的多标签Index
        zeros = torch.zeros_like(logits[..., :1])       # <3, 1>
        logits_neg = torch.cat([logits_neg, zeros], dim=-1)  # <3, 5>
        logits_pos = torch.cat([logits_pos, zeros], dim=-1)  # <3, 5>
        neg_loss = torch.logsumexp(logits_neg, dim=-1)       # <3, >
        pos_loss = torch.logsumexp(logits_pos, dim=-1)       # <3, >
        loss = neg_loss + pos_loss                           # pos比零大, neg比零小
        if "mean" == self.reduction:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction="mean"):
        """FocalLoss
        聚焦损失, 不确定的情况下alpha==0.5效果可能会好一点
        url: https://github.com/CoinCheung/pytorch-loss
        Usage is same as nn.BCEWithLogits:
          >>> loss = criteria(logits, lbs)
        """
        super(FocalLoss, self).__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)
        coeff = torch.abs(labels - probs).pow(self.gamma).neg()
        log_0_probs = torch.where(logits >= 0, -logits + nn.functional.softplus(logits, -1, 50), -nn.functional.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0, nn.functional.softplus(logits, -1, 50), logits - nn.functional.softplus(logits, 1, 50))
        loss = labels * self.alpha * log_1_probs + (1. - labels) * (1. - self.alpha) * log_0_probs
        loss = loss * coeff
        if self.reduction == "mean":
            loss = loss.mean()
        if self.reduction == "sum":
            loss = loss.sum()
        return loss


class DiceLossV1(nn.Module):
    def __init__(self, reduction="mean", epsilon=1e-9):
        """【ERROR, 不收敛-原因未知】Dice-Loss, 切块损失, 用于不均衡数据, 但是收敛困难
        paper: Dice Loss for Data-imbalanced NLP Tasks
        url: https://arxiv.org/pdf/1911.02855.pdf
        args:
            reduction: str, Specifies the reduction to apply to the output, 输出形式. 
                            eg.``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``
            epsilon: float, Minimum of maths, 无穷小.  eg. 1e-9
        returns:
            Tensor of loss.
        examples:
            >>> label, logits = [[1, 1, 1, 1], [0, 0, 0, 1]], [[0, 1, 1, 0], [1, 0, 0, 1],]
            >>> label, logits = torch.tensor(label).float(), torch.tensor(logits).float()
            >>> loss = DiceLoss()(logits, label)
        """
        super(DiceLossV1, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, logits, labels):
        prob = torch.sigmoid(logits)  # <2, 4>
        # logits: [N, C], index: [N, ]
        index = labels.unsqueeze(1).view(prob.size(0), -1)  # <2, 4>
        prob = torch.gather(prob, dim=1, index=index)
        dsc_i = 1 - ((1 - prob) * prob + self.epsilon) / ((1 - prob) * prob + 1 + self.epsilon)
        if "mean" == self.reduction:
            loss = dsc_i.mean()
        else:
            loss = dsc_i.sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-9):
        """Dice-Loss, 切块损失, 用于不均衡数据, 但是收敛困难, 不太稳定
        paper: Dice Loss for Data-imbalanced NLP Tasks
        url: https://arxiv.org/pdf/1911.02855.pdf
        args:
            reduction: str, Specifies the reduction to apply to the output, 输出形式. 
                            eg.``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``
            epsilon: float, Minimum of maths, 无穷小.  eg. 1e-9
        returns:
            Tensor of loss.
        examples:
            >>> label, logits = [[1, 1, 1, 1], [0, 0, 0, 1]], [[0, 1, 1, 0], [1, 0, 0, 1],]
            >>> label, logits = torch.tensor(label).long(), torch.tensor(logits).float()
            >>> loss = DiceLoss()(logits, label)
        """
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits, labels):  # 利用预测值与标签相乘当作交集
        predict = torch.sigmoid(logits)
        intersect =  predict * labels + self.epsilon
        unionset = predict + labels + self.epsilon
        loss = 1 - 2 * intersect.sum() / unionset.sum()
        return loss


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.1, is_active=True,
                 is_dropout=True, active_type="mish"):
        """
        FC-Layer, mostly last output of model
        args:
            input_dim: input dimension, 输入维度, eg. 768
            output_dim: output dimension, 输出维度, eg. 32
            dropout_rate: dropout rate, 随机失活, eg. 0.1
            is_dropout: use dropout or not, 是否使用随机失活dropout, eg. True
            is_active: use activation or not, 是否使用激活函数如tanh, eg. True
            active_type: type of activate function, 激活函数类型, eg. "tanh", "relu"
        Returns:
            Tensor of batch.
        """
        super(FCLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)  # probability of an element to be zeroed
        self.is_dropout = is_dropout
        self.active_type = active_type
        self.is_active = is_active
        self.softmax = nn.Softmax(1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()

    def forward(self, x):
        if self.is_dropout:
            x = self.dropout(x)
        x = self.linear(x)
        if self.is_active:
            if    self.active_type.upper() == "MISH":
                x = x * torch.tanh(nn.functional.softplus(x))
            elif self.active_type.upper() == "SWISH":
                x = x * torch.sigmoid(x)
            elif self.active_type.upper() == "TANH":
                x = self.tanh(x)
            elif self.active_type.upper() == "GELU":
                x = self.gelu(x)
            elif self.active_type.upper() == "RELU":
                x = self.relu(x)
            else:
                x = self.relu(x)
        return x


class Swish(nn.Module):
    def __init__(self):
        """ Swish函数可以看做是介于线性函数与ReLU函数之间的平滑函数.(sigmoid和Relu的拼凑)
        Searching for Activation Functions
        Applies the swish function element-wise:
            f(x)=x⋅sigmoid(βx)
        paper: https://arxiv.org/abs/1710.05941(2017)
        """
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    def __index__(self):
        """
        Script provides functional interface for Mish activation function.
        Applies the mish function element-wise:
            mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        See additional documentation for mish class.
        """
        super().__init__()

    def forword(self, x):
        x = x * torch.tanh(nn.functional.softplus(x))
        return x


if __name__ == '__main__':

    label, logits = [[1, 1, 1, 1], [0, 0, 0, 1]], [[0, 1, 1, 0], [1, 0, 0, 1], ]
    label, logits = torch.tensor(label).long(), torch.tensor(logits).float()

    dice = DiceLoss()
    loss = dice(logits, label)
    print(loss)

    dice2 = DiceLossV1()
    loss = dice2(logits, label)
    print(loss)

    lsce = LabelSmoothingCrossEntropy()
    loss = lsce(logits, label)
    print(loss)

    lsce = LabelSmoothingCrossEntropyV1()
    loss = lsce(logits, label)
    print(loss)

