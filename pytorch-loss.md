# some nlp pytorch-loss
 - [通过互信息思想来缓解类别不平衡问题](https://spaces.ac.cn/archives/7615)
 - [将“softmax+交叉熵”推广到多标签分类问题](https://spaces.ac.cn/archives/7359)


#  1. Prior-BCE

```bash
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

```





#  2. CircleLoss of MultiLabel
```bash
from torch import nn
import numpy as np
import torch
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

```
希望对你又送帮助！

