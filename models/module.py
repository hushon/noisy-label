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