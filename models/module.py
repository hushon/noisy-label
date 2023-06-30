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
    def __init__(self, scale=1.0, reduction="mean"):
        super().__init__()
        self.register_buffer('scale', torch.tensor(scale))
        self.reduction = reduction

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=-1)
        loss = 2 - 2*torch.gather(pred, -1, labels.view(-1, 1))
        loss = loss.squeeze(-1)
        # Note: Reduced MAE
        # Original: torch.abs(pred - label_one_hot).sum(dim=1)
        # $MAE = \sum_{k=1}^{K} |\bm{p}(k|\bm{x}) - \bm{q}(k|\bm{x})|$
        # $MAE = \sum_{k=1}^{K}\bm{p}(k|\bm{x}) - p(y|\bm{x}) + (1 - p(y|\bm{x}))$
        # $MAE = 2 - 2p(y|\bm{x})$
        #
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            loss = loss
        else:
            raise NotImplementedError
        return loss * self.scale


class ReverseCrossEntropyLoss(nn.Module):
    '''
    Wang, Yisen, et al. "Symmetric cross entropy for robust learning with noisy labels."
    Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.
    '''
    def __init__(self, num_classes, scale=1.0, reduction="mean"):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer('scale', torch.tensor(scale))
        self.reduction = reduction

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0) # 1e-4 indicates A = -4 in RCE
        loss = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            loss = loss
        else:
            raise NotImplementedError
        return loss * self.scale


class SymmetricCrossEntropyLoss(nn.Module):
    '''
    Wang, Yisen, et al. "Symmetric cross entropy for robust learning with noisy labels."
    Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.
    '''
    def __init__(self, alpha=0.1, beta=1.0, num_classes=10, reduction="mean"):
        super().__init__()
        self.register_buffer('alpha', torch.tensor(alpha))
        self.register_buffer('beta', torch.tensor(beta))
        self.num_classes = num_classes
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(reduction=self.reduction)
        self.reverse_cross_entropy = ReverseCrossEntropyLoss(
                                        self.num_classes,
                                        scale=1.0,
                                        reduction=self.reduction
                                        )

    def forward(self, pred, labels):
        ce = self.cross_entropy(pred, labels)
        rce = self.reverse_cross_entropy(pred, labels)
        loss = self.alpha * ce + self.beta * rce
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            loss = loss
        else:
            raise NotImplementedError
        return loss


class GeneralizedCrossEntropyLoss(nn.Module):
    '''
    Zhang, Zhilu, and Mert Sabuncu. "Generalized cross entropy loss for
    training deep neural networks with noisy labels."
    Advances in neural information processing systems 31 (2018).
    '''
    def __init__(self, num_classes, q=0.7, reduction="mean"):
        super().__init__()
        self.register_buffer('q', torch.tensor(q))
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float()
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            loss = loss
        else:
            raise NotImplementedError
        return loss


class KLDivDistillationLoss(nn.KLDivLoss):
    """
    Modified nn.KLDivLoss which
    takes in logits instead of log-likelihoods
    """
    def __init__(self, temperature=1.0, reduction="none"):
        super().__init__(reduction=reduction)
        self.register_buffer('temperature', torch.tensor(temperature))

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
        self.register_buffer('temperature', torch.tensor(temperature))

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
        self.register_buffer('temperature', torch.tensor(temperature))

    def forward(self, pred_logits, target_logits):
        x = pred_logits.div(self.temperature).softmax(-1)
        y = target_logits.div(self.temperature).softmax(-1)
        if self.reduction == 'none':
            loss = super().forward(x, y).sum(-1)
        else:
            loss = super().forward(x, y)
        return loss * (self.temperature**2)


class CrossEntropyDistillationLoss(nn.CrossEntropyLoss):
    """
    Modified nn.CrossEntropyLoss which
    takes in logits instead of probabilities
    """
    def __init__(self, temperature=1.0, reduction="none"):
        super().__init__(reduction=reduction)
        self.register_buffer('temperature', torch.tensor(temperature))

    def forward(self, pred_logits, target_logits):
        x = pred_logits.div(self.temperature)
        y = target_logits.div(self.temperature).softmax(-1)
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


class Normalize2D(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return x.sub_(self.mean).div_(self.std)