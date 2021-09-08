# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/7/24 21:45
# @author  : Mo
# @function: Layer and Loss


from torch import nn
import torch

import numpy as np


__all__ = ["PriorMultiLabelSoftMarginLoss",
           "LabelSmoothingCrossEntropyV1",
           "LabelSmoothingCrossEntropy",
           "MultiLabelCircleLoss",
           "FocalLoss",
           "DiceLossV1",
           "DiceLoss",
           "SpanFCLayer",
           "FCLayer",
           "Mish",
           "CRF",
           "GridPointer",
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


class LabelSmoothingCrossEntropyV1(nn.Module):
    def __init__(self, eps=0.1, reduction="mean", ignore_index=-100):
        """【ERROR，直接smooth输入logits效果不好，原因未知】LabelSmoothingCrossEntropy, no-softmax-input
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
        >>> loss = LabelSmoothingCrossEntropyV1()(logits, label)
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
        logits_neg = logits - labels * self.inf         # <3, 4>
        logits_pos = logits - (1 - labels) * self.inf   # <3, 4>
        zeros = torch.zeros_like(logits[..., :1])       # <3, 1>
        logits_neg = torch.cat([logits_neg, zeros], dim=-1)  # <3, 5>
        logits_pos = torch.cat([logits_pos, zeros], dim=-1)  # <3, 5>
        neg_loss = torch.logsumexp(logits_neg, dim=-1)       # <3, >
        pos_loss = torch.logsumexp(logits_pos, dim=-1)       # <3, >
        loss = neg_loss + pos_loss
        if "mean" == self.reduction:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction="mean"):
        """FocalLoss
        聚焦损失, 不确定的情况下alpha==0.5效果可能会好一点
        Usage is same as nn.BCEWithLogits:
        >>> loss = criteria(logits, lbs)
        """
        super(FocalLoss, self).__init__()
        self.crit = nn.BCEWithLogitsLoss(reduction="none")
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
        intersect = predict * labels + self.epsilon
        unionset = predict + labels + self.epsilon
        loss = 1. - 2 * intersect.sum() / unionset.sum()
        return loss


class SpanFCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.1, is_active=True,
                 is_dropout=True, active_type="mish"):
        """SpanFCLayer
        Span-FC-Layer, mostly last output of span of model, 新增LayerNorm(条件层标准化)
        args:
            input_dim: input dimension, 输入维度, eg. 768
            output_dim: output dimension, 输出维度, eg. 32
            dropout_rate: dropout rate, 随机失活, eg. 0.1
            is_dropout: use dropout or not, 是否使用随机失活dropout, eg. True
            is_active: use activation or not, 是否使用激活函数如tanh, eg. True
            active_type: type of activate function, 激活函数类型, eg. "tanh", "relu", "mish"
        Returns:
            Tensor of batch.
        """
        super(SpanFCLayer, self).__init__()
        self.linear_0 = nn.Linear(input_dim, input_dim)
        self.linear_1 = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout_rate)  # probability of an element to be zeroed
        self.is_dropout = is_dropout
        self.active_type = active_type
        self.is_active = is_active
        self.softmax = nn.Softmax(1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)  # inplace是否覆盖, 为了节省内存
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()

    def forward(self, x):
        if self.is_dropout:
            x = self.dropout(x)
        x = self.linear_0(x)
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
        x = self.layer_norm(x)
        x = self.linear_1(x)
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
    def __init__(self):
        """ Mish函数可以看做是介于线性函数与ReLU函数之间的平滑函数.(tanh和Relu的拼凑)
        Script provides functional interface for Mish activation function.
        Applies the mish function element-wise:
            mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        See additional documentation for mish class.
        """
        super(Mish).__init__()

    def forword(self, x):
        x = x * torch.tanh(nn.functional.softplus(x))
        return x


class CRF(nn.Module):
    """Conditional random field.
    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.
    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.
    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.
    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.
    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError('invalid number of tags: {}'.format(num_tags))
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return '{}(num_tags={})'.format(self.__class__.__name__, self.num_tags)

    def forward(self, emissions: torch.Tensor, tags: torch.LongTensor, mask = None, reduction = 'mean'):
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.
        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError('invalid reduction: {}'.format(reduction))
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8, device=tags.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, tags=tags, mask=mask)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        return llh.sum() / mask.float().sum()

    def decode(self, emissions: torch.Tensor, mask = None, nbest = None, pad_tag = None):
        """Find the most likely tag sequence using Viterbi algorithm.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            nbest (`int`): Number of most probable paths for each sequence
            pad_tag (`int`): Tag at padded positions. Often input varies in length and
                the length will be padded to the maximum length in the batch. Tags at
                the padded positions will be assigned with a padding tag, i.e. `pad_tag`
        Returns:
            A PyTorch tensor of the best tag sequence for each batch of shape
            (nbest, batch_size, seq_length)
        """
        if nbest is None:
            nbest = 1
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.uint8,
                              device=emissions.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, mask=mask)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        if nbest == 1:
            return self._viterbi_decode(emissions, mask, pad_tag).unsqueeze(0)
        return self._viterbi_decode_nbest(emissions, mask, nbest, pad_tag)

    def _validate(self, emissions: torch.Tensor, tags = None, mask = None):
        if emissions.dim() != 3:
            raise ValueError('emissions must have dimension of 3, got {}'.format(emissions.dim()))
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                'expected last dimension of emissions is {}, '.format(self.num_tags) + 'got {}'.format(emissions.size(2)))

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    'got {} and {}'.format(tuple(emissions.shape[:2]), tuple(tags.shape)))

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    'got {} and {}'.format(tuple(emissions.shape[:2]), tuple(mask.shape)))
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(self, emissions: torch.Tensor, tags: torch.LongTensor, mask: torch.ByteTensor):
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, emissions: torch.Tensor, mask: torch.ByteTensor):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor, mask: torch.ByteTensor, pad_tag = None):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        # return: (batch_size, seq_length)
        if pad_tag is None:
            pad_tag = 0

        device = emissions.device
        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros((seq_length, batch_size, self.num_tags),
                                  dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags),
                              dtype=torch.long, device=device)
        oor_tag = torch.full((seq_length, batch_size), pad_tag,
                             dtype=torch.long, device=device)

        # - score is a tensor of size (batch_size, num_tags) where for every batch,
        #   value at column j stores the score of the best tag sequence so far that ends
        #   with tag j
        # - history_idx saves where the best tags candidate transitioned from; this is used
        #   when we trace back the best tag sequence
        # - oor_idx saves the best tags candidate transitioned from at the positions
        #   where mask is 0, i.e. out of range (oor)

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1), indices, oor_idx)
            history_idx[i - 1] = indices

        # End transition score
        # shape: (batch_size, num_tags)
        end_score = score + self.end_transitions
        _, end_tag = end_score.max(dim=1)

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1

        # insert the best tag at each sequence end (last position with mask == 1)
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(1, seq_ends.view(-1, 1, 1).expand(-1, 1, self.num_tags),
                             end_tag.view(-1, 1, 1).expand(-1, 1, self.num_tags))
        history_idx = history_idx.transpose(1, 0).contiguous()

        # The most probable path for each sequence
        best_tags_arr = torch.zeros((seq_length, batch_size),
                                    dtype=torch.long, device=device)
        best_tags = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx], 1, best_tags)
            best_tags_arr[idx] = best_tags.data.view(batch_size)

        return torch.where(mask, best_tags_arr, oor_tag).transpose(0, 1)

    def _viterbi_decode_nbest(self, emissions: torch.FloatTensor, mask: torch.ByteTensor, nbest: int, pad_tag = None):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        # return: (nbest, batch_size, seq_length)
        if pad_tag is None:
            pad_tag = 0

        device = emissions.device
        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros((seq_length, batch_size, self.num_tags, nbest),
                                  dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags, nbest),
                              dtype=torch.long, device=device)
        oor_tag = torch.full((seq_length, batch_size, nbest), pad_tag,
                             dtype=torch.long, device=device)

        # + score is a tensor of size (batch_size, num_tags) where for every batch,
        #   value at column j stores the score of the best tag sequence so far that ends
        #   with tag j
        # + history_idx saves where the best tags candidate transitioned from; this is used
        #   when we trace back the best tag sequence
        # - oor_idx saves the best tags candidate transitioned from at the positions
        #   where mask is 0, i.e. out of range (oor)

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            if i == 1:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1)
                # shape: (batch_size, num_tags, num_tags)
                next_score = broadcast_score + self.transitions + broadcast_emission
            else:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1).unsqueeze(2)
                # shape: (batch_size, num_tags, nbest, num_tags)
                next_score = broadcast_score + self.transitions.unsqueeze(1) + broadcast_emission

            # Find the top `nbest` maximum score over all possible current tag
            # shape: (batch_size, nbest, num_tags)
            next_score, indices = next_score.view(batch_size, -1, self.num_tags).topk(nbest, dim=1)

            if i == 1:
                score = score.unsqueeze(-1).expand(-1, -1, nbest)
                indices = indices * nbest

            # convert to shape: (batch_size, num_tags, nbest)
            next_score = next_score.transpose(2, 1)
            indices = indices.transpose(2, 1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags, nbest)
            score = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1), indices, oor_idx)
            history_idx[i - 1] = indices

        # End transition score shape: (batch_size, num_tags, nbest)
        end_score = score + self.end_transitions.unsqueeze(-1)
        _, end_tag = end_score.view(batch_size, -1).topk(nbest, dim=1)

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1

        # insert the best tag at each sequence end (last position with mask == 1)
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(1, seq_ends.view(-1, 1, 1, 1).expand(-1, 1, self.num_tags, nbest),
                             end_tag.view(-1, 1, 1, nbest).expand(-1, 1, self.num_tags, nbest))
        history_idx = history_idx.transpose(1, 0).contiguous()

        # The most probable path for each sequence
        best_tags_arr = torch.zeros((seq_length, batch_size, nbest),
                                    dtype=torch.long, device=device)
        best_tags = torch.arange(nbest, dtype=torch.long, device=device) \
                         .view(1, -1).expand(batch_size, -1)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx].view(batch_size, -1), 1, best_tags)
            best_tags_arr[idx] = best_tags.data.view(batch_size, -1) // nbest

        return torch.where(mask.unsqueeze(-1), best_tags_arr, oor_tag).permute(2, 1, 0)


class GridPointer(nn.Module):
    def __init__(self, head_nums, head_size, is_RoPE=True):
        """GridPointer, 分类-网格(全局)指针模块
        将序列的每个(start, end)作为整体来进行判断
        代码来源:
        网址url: [GlobalPointer：用统一的方式处理嵌套和非嵌套NER](https://kexue.fm/archives/8373)
        ptorch版gaohongkui: https://github.com/gaohongkui/GlobalPointer_pytorch
        """
        super(GridPointer, self).__init__()
        self.head_nums = head_nums
        self.head_size = head_size
        self.is_RoPE = is_RoPE

    def forward(self, x, attention_mask, token_type_ids):
        batch_size = x.size(0)
        max_len = x.size(1)

        outputs = torch.split(x, self.head_size * 2, dim=-1)  # <batch, len, label, head*2>
        outputs = torch.stack(outputs, dim=-2)
        qw, kw = outputs[..., :self.head_size], outputs[..., self.head_size:]  # <batch, len, label, head>
        if self.is_RoPE:
            def SinusoidalPositionEmbedding(output_size, batch_size, max_len, device):
                """embedding of Sinusoidal-Position
                """
                position_ids = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
                indices = torch.arange(0, output_size // 2, dtype=torch.float)
                indices = torch.pow(10000, -2 * indices / output_size)
                embeddings = position_ids * indices
                embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
                embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
                embeddings = torch.reshape(embeddings, (batch_size, max_len, output_size))
                embeddings = embeddings.to(device)
                return embeddings

            pos_emb = SinusoidalPositionEmbedding(self.head_size, batch_size, max_len, device=x.device)  # <batch, len, head>
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)   # <batch, len, 1, head>
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)   # <batch, len, 1, head>
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        logits = torch.einsum("bmhd, bnhd->bhmn", qw, kw)  # <batch, label, len, len>
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.head_nums, max_len, max_len)
        logits = logits*pad_mask - (1-pad_mask)*1e12
        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), diagonal=-1)
        logits = (logits - mask * 1e12)
        logits = logits / self.head_size**0.5  # scale
        return logits

