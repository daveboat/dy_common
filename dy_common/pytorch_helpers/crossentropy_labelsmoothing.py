"""
CrossEntropyLoss with label smoothing, from https://github.com/eladhoffer/utils.pytorch/blob/master/cross_entropy.py
but with some changes for my personal use
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy(inputs, target, weight=None, ignore_index=-100, reduction='mean', smooth_eps=None, from_logits=True):
    """
    cross entropy loss, with support for label smoothing. See https://arxiv.org/abs/1512.00567

    if from_logits is True, input is expected to be a tensor of logits. if from_logits is False, input is expected to be
    a tensor of probabilities.

    target is expected to be a LongTensor of target indices, i.e. "sparse" in tensorflow terminology, i.e. not one-hot

    reduction = ['mean', 'sum', anything else=none]
    """
    smooth_eps = smooth_eps or 0

    # ordinary log-likelihood - use cross_entropy from nn
    if smooth_eps == 0:
        if from_logits:
            return F.cross_entropy(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)
        else:
            return F.nll_loss(torch.log(inputs), target, weight, ignore_index=ignore_index, reduction=reduction)

    # if from logits, take log softmax, else, we assume inputs are probabilities and just take the log
    if from_logits:
        lsm = F.log_softmax(inputs, dim=-1)
    else:
        lsm = torch.log(inputs)

    num_classes = inputs.size(-1)
    masked_indices = target.eq(ignore_index)

    if weight is not None:
        lsm = lsm * weight.unsqueeze(0)

    # this is the label smoothing calculation, and uses the bottom equation on page 7 of https://arxiv.org/abs/1512.00567
    # also see https://leimao.github.io/blog/Label-Smoothing/
    # this is different than what was originally in eladhoffer's repo, because I believe there was a
    eps_sum = smooth_eps / num_classes
    eps_nll = 1. - smooth_eps
    likelihood = lsm.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)  # this is sum_y p(y|x_i) log q(y|x_i)
    loss = -(eps_nll * likelihood + eps_sum * lsm.sum(-1))  # lsm.sum(-1) is sum_y log q(y|x_i)

    # handle masking
    if masked_indices is not None:
        loss.masked_fill_(masked_indices, 0)

    # handle reduction
    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        if masked_indices is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / float(loss.size(0) - masked_indices.sum())

    return loss


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """CrossEntropyLoss - with ability to recieve distrbution as targets, and optional label smoothing"""

    def __init__(self, weight=None, ignore_index=-100, reduction='mean', smooth_eps=None, from_logits=True):
        super(CrossEntropyLoss, self).__init__(weight=weight,
                                               ignore_index=ignore_index, reduction=reduction)
        self.smooth_eps = smooth_eps
        self.from_logits = from_logits

    def forward(self, input, target, smooth_dist=None):
        return cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index,
                             reduction=self.reduction, smooth_eps=self.smooth_eps, from_logits=self.from_logits)


if __name__ == '__main__':
    torch.manual_seed(42)

    input = torch.randn((5, 10))
    target = torch.randint(0, 9, (5,))

    print(cross_entropy(input, target, reduction='none'))
    print(cross_entropy(input, target, smooth_eps=0.5, reduction='none'))
