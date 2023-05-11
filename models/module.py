import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanAbsoluteError(torch.nn.Module):
    '''
    Ghosh, Aritra, Himanshu Kumar, and P. Shanti Sastry.
    "Robust loss functions under label noise for deep neural networks."
    Proceedings of the AAAI conference on artificial intelligence.

    Inputs:
        pred: unnormalized logits
        labels: target labels
    '''
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=-1)
        mae = 2 - 2*torch.gather(pred, -1, labels.view(-1, 1))
        mae = mae.squeeze(-1)
        # Note: Reduced MAE
        # Original: torch.abs(pred - label_one_hot).sum(dim=1)
        # $MAE = \sum_{k=1}^{K} |\bm{p}(k|\bm{x}) - \bm{q}(k|\bm{x})|$
        # $MAE = \sum_{k=1}^{K}\bm{p}(k|\bm{x}) - p(y|\bm{x}) + (1 - p(y|\bm{x}))$
        # $MAE = 2 - 2p(y|\bm{x})$
        #
        if self.reduction == "mean":
            return mae.mean()
        elif self.reduction == "sum":
            return mae.sum()
        elif self.reduction == "none":
            return mae
        else:
            raise NotImplementedError


class KLDivDistillationLoss(nn.KLDivLoss):
    """
    Modified nn.KLDivLoss which
    takes in logits instead of log-likelihoods
    """
    def __init__(self, temperature=1.0, reduction="none"):
        super().__init__(reduction=reduction)
        self.temperature = temperature

    def forward(self, pred_logits, target_logits):
        x = pred_logits.div(self.temperature).log_softmax(-1)
        y = target_logits.div(self.temperature).softmax(-1)
        if self.reduction == 'none':
            loss = super().forward(x, y).sum(-1)
        else:
            loss = super().forward(x, y)
        return loss * (self.temperature**2)


class L1DistillationLoss(nn.L1Loss):
    """
    Modified nn.L1Loss which
    takes in logits instead of probabilities
    """
    def __init__(self, temperature=1.0, reduction="none"):
        super().__init__(reduction=reduction)
        self.temperature = temperature

    def forward(self, pred_logits, target_logits):
        x = pred_logits.div(self.temperature).softmax(-1)
        y = target_logits.div(self.temperature).softmax(-1)
        if self.reduction == 'none':
            loss = super().forward(x, y).sum(-1)
        else:
            loss = super().forward(x, y)
        return loss * (self.temperature**2)


class SmoothL1DistillationLoss(nn.SmoothL1Loss):
    """
    Modified nn.SmoothL1Loss which
    takes in logits instead of probabilities
    """
    def __init__(self, temperature=1.0, reduction="none"):
        super().__init__(reduction=reduction)
        self.temperature = temperature

    def forward(self, pred_logits, target_logits):
        x = pred_logits.div(self.temperature).softmax(-1)
        y = target_logits.div(self.temperature).softmax(-1)
        if self.reduction == 'none':
            loss = super().forward(x, y).sum(-1)
        else:
            loss = super().forward(x, y)
        return loss * (self.temperature**2)


class MCDropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.dropout(input, self.p, True, self.inplace)


class GaussianDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        assert 0.0 <= p <= 1.0
        self.p = p
        self.std = (self.p / (1.0 - self.p))**0.5

    def forward(self, x):
        if self.training:
            epsilon = torch.rand_like(x).mul_(self.std)
            return x * epsilon
        else:
            return x


class GaussianMCDropout(GaussianDropout):
    def forward(self, x):
        epsilon = torch.rand_like(x).mul_(self.std)
        return x * epsilon


class LambdaLayer(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)