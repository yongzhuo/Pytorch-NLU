# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/7/24 21:45
# @author  : Mo
# @function: Layer and Loss
# @reference: ResampleLoss<CB-loss> code from 'https://github.com/wutong16/DistributionBalancedLoss/blob/a3ecaa9021a920fcce9fdafbd7d83b51bf526af8/mllt/models/losses/resample_loss.py'
# @reference: ResampleLoss<DB-loss> code from 'https://github.com/Roche/BalancedLossNLP'


import torch.nn.functional as F
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
           "ResampleLoss",  # class-balance
           "partial_cross_entropy",
           "binary_cross_entropy",
           "cross_entropy",
           "weight_reduce_loss",
           "reduce_loss",
           ]


class PriorMultiLabelSoftMarginLoss(nn.Module):
    def __init__(self, prior=None, num_labels=None, reduction="mean", eps=1e-6, tau=1.0):
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
        if prior is None: prior = np.array([1/num_labels for _ in range(num_labels)])  # 如果不存在就设置为num
        if type(prior) == list: prior = np.array(prior)
        self.log_prior = torch.tensor(np.log(prior + eps)).unsqueeze(0)
        self.eps = eps
        self.tau = tau

    def forward(self, logits, labels):
        # 使用与输入label相同的device
        logits = logits + self.tau * self.log_prior.to(labels.device)
        loss = self.loss_mlsm(logits, labels)
        return loss


class LabelSmoothingCrossEntropyV3(nn.Module):
    """ 平滑的交叉熵, LabelSommth-CrossEntropy
        url: https://github.com/Tongjilibo/bert4torch/blob/master/bert4torch/losses.py
        examples:
            >>> criteria = LabelSmoothingCrossEntropyV2()
            >>> logits = torch.randn(8, 19, 384, 384)  # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384))  # nhw, int64_t
            >>> loss = criteria(logits, lbs)
    """
    def __init__(self, eps=0.1, reduction='mean',ignore_index=-100):
        super(LabelSmoothingCrossEntropyV3, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = nn.functional.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * nn.functional.nll_loss(log_preds, target, reduction=self.reduction,
                                                           ignore_index=self.ignore_index)


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
    def __init__(self, reduction="mean", epsilon=1e-6):
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
    def __init__(self, epsilon=1e-6):
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
        intersect = predict * labels + self.epsilon
        unionset = predict + labels + self.epsilon
        loss = 1 - 2 * intersect.sum() / unionset.sum()
        return loss


class NCELoss(nn.Module):
    def __init__(self):
        """NCE-Loss, 切块损失, 用于不均衡数据, 但是收敛困难, 不太稳定(可能有溢出)
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
            >>> loss = NCELoss()(logits, label)
        """
        super(NCELoss, self).__init__()
        self.loss_bce = torch.nn.BCELoss()
        self.sigmoid = torch.sigmoid

    def forward(self, logits, labels):  # 利用预测值与标签相乘当作交集
        """
        # input is batch_size*2 int Variable
        i = self.input_embeddings(in_out_pairs[:, 0])
        o = self.output_embeddings(in_out_pairs[:, 1])
        # raw activations, NCE_Loss handles the sigmoid (we need to know classes to know the sign to apply)
        return (i * o).sum(1).squeeze()

        loss_func(torch.sigmoid(logits * labels).sum())
        """
        # return torch.log(torch.sigmoid(logits * labels).sum()) * -1.0
        logits_sigmoid = self.sigmoid(logits * labels)  # 只关注正样本?
        return self.loss_bce(logits_sigmoid, labels.float())


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
        self.relu = nn.ReLU(inplace=False)
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()

    def forward(self, x):
        if self.is_dropout:
            x = self.dropout(x)
        x = self.linear(x)
        if self.is_active:
            if self.active_type.upper() ==   "MISH":
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



def partial_cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()

    mask = label == -1
    loss = F.binary_cross_entropy_with_logits(pred, label.float(), weight, reduction='none')
    if mask.sum() > 0:
        loss *= (1-mask).float()
        avg_factor = (1-mask).float().sum()
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)
    return loss


def binary_cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()

    loss = F.binary_cross_entropy_with_logits(pred, label.float(), weight, reduction='none')
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

    return loss


def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    # element-wise losses
    loss = F.cross_entropy(pred, label, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


class ResampleLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True, partial=False,
                 loss_weight=1.0, reduction='mean',
                 reweight_func=None,  # None, 'inv', 'sqrt_inv', 'rebalance', 'CB'
                 weight_norm=None,  # None, 'by_instance', 'by_batch'
                 focal=dict(
                     focal=True,
                     alpha=0.5,
                     gamma=2,
                 ),
                 map_param=dict(
                     alpha=10.0,
                     beta=0.2,
                     gamma=0.1
                 ),
                 CB_loss=dict(
                     CB_beta=0.9,
                     CB_mode='average_w'  # 'by_class', 'average_n', 'average_w', 'min_n'
                 ),
                 logit_reg=dict(
                     neg_scale=5.0,
                     init_bias=0.1
                 ),
                 class_freq=None,
                 train_num=None):
        super(ResampleLoss, self).__init__()

        assert (use_sigmoid is True) or (partial is False)
        self.use_sigmoid = use_sigmoid
        self.partial = partial
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.use_sigmoid:
            if self.partial:
                self.cls_criterion = partial_cross_entropy
            else:
                self.cls_criterion = binary_cross_entropy
        else:
            self.cls_criterion = cross_entropy

        # reweighting function
        self.reweight_func = reweight_func

        # normalization (optional)
        self.weight_norm = weight_norm

        # focal loss params
        self.focal = focal['focal']
        self.gamma = focal['gamma']
        self.alpha = focal['alpha']  # change to alpha

        # mapping function params
        self.map_alpha = map_param['alpha']
        self.map_beta = map_param['beta']
        self.map_gamma = map_param['gamma']

        # CB loss params (optional)
        self.CB_beta = CB_loss['CB_beta']
        self.CB_mode = CB_loss['CB_mode']

        # self.class_freq = torch.from_numpy(np.asarray(class_freq)).float()
        self.class_freq = torch.tensor([item for item in class_freq])
        self.num_classes = self.class_freq.shape[0]
        self.train_num = train_num  # only used to be divided by class_freq
        # regularization params
        self.logit_reg = logit_reg
        self.neg_scale = logit_reg['neg_scale'] if 'neg_scale' in logit_reg else 1.0
        init_bias = logit_reg['init_bias'] if 'init_bias' in logit_reg else 0.0
        self.init_bias = - torch.log(
            self.train_num / self.class_freq - 1) * init_bias  ########################## bug fixed https://github.com/wutong16/DistributionBalancedLoss/issues/8

        self.freq_inv = torch.ones(self.class_freq.shape) / self.class_freq
        self.propotion_inv = self.train_num / self.class_freq

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        """  cls_score is logits  """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        weight = self.reweight_functions(label)

        cls_score, weight = self.logit_reg_functions(label.float(), cls_score, weight)

        if self.focal:
            logpt = self.cls_criterion(cls_score.clone(), label, weight=None, reduction='none', avg_factor=avg_factor)
            # pt is sigmoid(logit) for pos or sigmoid(-logit) for neg
            pt = torch.exp(-logpt)
            wtloss = self.cls_criterion(cls_score, label.float(), weight=weight, reduction='none')
            alpha_t = torch.where(label == 1, self.alpha, 1 - self.alpha)
            loss = alpha_t * ((1 - pt) ** self.gamma) * wtloss  ####################### balance_param should be a tensor
            loss = reduce_loss(loss, reduction)  ############################ add reduction
        else:
            loss = self.cls_criterion(cls_score, label.float(), weight, reduction=reduction)

        loss = self.loss_weight * loss
        return loss

    def reweight_functions(self, label):
        if self.reweight_func is None:
            return None
        elif self.reweight_func in ['inv', 'sqrt_inv']:
            weight = self.RW_weight(label.float())
        elif self.reweight_func in 'rebalance':
            weight = self.rebalance_weight(label.float())
        elif self.reweight_func in 'CB':
            weight = self.CB_weight(label.float())
        else:
            return None

        if self.weight_norm is not None:
            if 'by_instance' in self.weight_norm:
                max_by_instance, _ = torch.max(weight, dim=-1, keepdim=True)
                weight = weight / max_by_instance
            elif 'by_batch' in self.weight_norm:
                weight = weight / torch.max(weight)

        return weight

    def logit_reg_functions(self, labels, logits, weight=None):
        if not self.logit_reg:
            return logits, weight
        if 'init_bias' in self.logit_reg:
            logits += self.init_bias.to(logits.device)
        if 'neg_scale' in self.logit_reg:
            logits = logits * (1 - labels) * self.neg_scale + logits * labels
            if weight is not None:
                weight = weight / self.neg_scale * (1 - labels) + weight * labels
        return logits, weight

    def rebalance_weight(self, gt_labels):
        self.freq_inv = self.freq_inv.to(gt_labels.device)
        repeat_rate = torch.sum(gt_labels.float() * self.freq_inv, dim=1, keepdim=True)
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate
        # pos and neg are equally treated
        weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha
        return weight

    def CB_weight(self, gt_labels):
        device = gt_labels.device
        if 'by_class' in self.CB_mode:
            weight = torch.tensor((1 - self.CB_beta)).to(device) / \
                     (1 - torch.pow(self.CB_beta, self.class_freq)).to(device)
        elif 'average_n' in self.CB_mode:
            avg_n = torch.sum(gt_labels * self.class_freq, dim=1, keepdim=True) / \
                    torch.sum(gt_labels, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)).to(device) / \
                     (1 - torch.pow(self.CB_beta, avg_n)).to(device)
        elif 'average_w' in self.CB_mode:
            weight_ = torch.tensor((1 - self.CB_beta)).to(device) / \
                      (1 - torch.pow(self.CB_beta, self.class_freq)).to(device)
            weight = torch.sum(gt_labels * weight_, dim=1, keepdim=True) / \
                     torch.sum(gt_labels, dim=1, keepdim=True)
        elif 'min_n' in self.CB_mode:
            min_n, _ = torch.min(gt_labels * self.class_freq +
                                 (1 - gt_labels) * 100000, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)).to(device) / \
                     (1 - torch.pow(self.CB_beta, min_n)).to(device)
        else:
            raise NameError
        return weight

    def RW_weight(self, gt_labels, by_class=True):
        if 'sqrt' in self.reweight_func:
            weight = torch.sqrt(self.propotion_inv)
        else:
            weight = self.propotion_inv
        if not by_class:
            sum_ = torch.sum(weight * gt_labels, dim=1, keepdim=True)
            weight = sum_ / torch.sum(gt_labels, dim=1, keepdim=True)
        return weight


if __name__ == '__main__':

    label, logits = [[1, 1, 1, 1], [0, 0, 0, 1]], [[0, 1, 1, 0], [1, 0, 0, 1], ]
    label, logits = torch.tensor(label).long().to("cpu"), torch.tensor(logits).float().to("cpu")
    # label, logits = torch.tensor(label).long().to("cuda:0"), torch.tensor(logits).float().to("cuda:0")

    # t1 = torch.tensor(label).long().to("cpu")
    # t2 = torch.tensor(label).long().to("cuda")

    dice = DiceLoss()
    loss = dice(logits, label)
    print("DiceLoss:")
    print(loss)

    dice2 = DiceLossV1()
    loss = dice2(logits, label)
    print("DiceLossV1:")
    print(loss)

    lsce = LabelSmoothingCrossEntropy()
    loss = lsce(logits, label)
    print("LabelSmoothingCrossEntropy:")
    print(loss)

    lsce = LabelSmoothingCrossEntropyV1()
    loss = lsce(logits, label)
    print("LabelSmoothingCrossEntropyV1:")
    print(loss)

    lnce = NCELoss()
    loss = lnce(logits, label)
    print("NCELoss:")
    print(loss)


    class_freq, train_num = [160, 110, 20, 10], 320
    rsl_cb = ResampleLoss(reweight_func='CB', loss_weight=10.0,
                          focal=dict(focal=True, alpha=0.5, gamma=2),
                          logit_reg=dict(),
                          CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                          class_freq=class_freq, train_num=train_num)

    rsl_cb_nf = ResampleLoss(reweight_func='CB', loss_weight=10.0,
                             focal=dict(focal=False, alpha=0.5, gamma=2),
                             logit_reg=dict(),
                             CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                             class_freq=class_freq, train_num=train_num)

    rsl_db = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                          focal=dict(focal=True, alpha=0.5, gamma=2),
                          logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                          map_param=dict(alpha=0.1, beta=10.0, gamma=0.9),
                          class_freq=class_freq, train_num=train_num)

    rsl_db_nf = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                             focal=dict(focal=False, alpha=0.5, gamma=2),
                             logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                             map_param=dict(alpha=0.1, beta=10.0, gamma=0.9),
                             class_freq=class_freq, train_num=train_num)
    loss_cb = rsl_cb(logits, label)
    loss_cb_nf = rsl_cb_nf(logits, label)
    loss_db = rsl_db(logits, label)
    loss_db_nf = rsl_db_nf(logits, label)
    print("CBLoss:")
    print(loss_cb)
    print("DBLoss:")
    print(loss_db)
    print("CBLoss_nf:")
    print(loss_cb_nf)
    print("DBLoss_nf:")
    print(loss_db_nf)

    pmlsm = PriorMultiLabelSoftMarginLoss(prior=np.array(class_freq)/sum(class_freq), num_labels=len(class_freq))
    loss = pmlsm(logits, label)
    print("PriorMultiLabelSoftMarginLoss:")
    print(loss)

    mlcl = MultiLabelCircleLoss()
    loss = mlcl(logits, label)
    print("MultiLabelCircleLoss:")
    print(loss)

    fl = FocalLoss()
    loss = fl(logits, label)
    print("FocalLoss:")
    print(loss)

    func_sigmoid = torch.sigmoid
    # loss_ce = torch.nn.CrossEntropyLoss(ignore_index=0)
    # loss = loss_ce(logits, label)
    # print("CrossEntropyLoss:")
    # print(loss)

    loss_mlsm = torch.nn.MultiLabelSoftMarginLoss()
    loss = loss_mlsm(func_sigmoid(logits), label)
    print("MultiLabelSoftMarginLoss:")
    print(loss)

    loss_bcelog = torch.nn.BCEWithLogitsLoss()
    label = label.float()  # 奇怪,直接放过去居然不行
    loss = loss_bcelog(logits, label)
    print("BCEWithLogitsLoss:")
    print(loss)

    loss_bce = torch.nn.BCELoss()
    loss = loss_bce(func_sigmoid(logits), label)
    print("BCELoss:")
    print(loss)

    loss_mse = torch.nn.MSELoss()
    loss = loss_mse(logits, label)
    print("MSELoss:")
    print(loss)

