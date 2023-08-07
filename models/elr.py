import torch
import torch.nn as nn
import torch.nn.functional as F


class elr_loss(nn.Module):
    def __init__(self, num_examp, num_classes, lam=3.0, beta=0.7):
        r"""Early Learning Regularization.
        Parameters
        * `num_examp` Total number of training examples.
        * `num_classes` Number of classes in the classification problem.
        * `lambda` Regularization strength; must be a positive float, controling the strength of the ELR.
        * `beta` Temporal ensembling momentum for target estimation.
        """
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer('target', torch.zeros(num_examp, self.num_classes))
        self.register_buffer('lam', torch.tensor(lam))
        self.register_buffer('beta', torch.tensor(beta))

    def forward(self, index, output, label):
        r"""Early Learning Regularization.
         Args
         * `index` Training sample index, due to training set shuffling, index is used to track training examples in different iterations.
         * `output` Model's logits, same as PyTorch provided loss functions.
         * `label` Labels, same as PyTorch provided loss functions.
         """
        y_pred = F.softmax(output,dim=1).clamp(1e-4, 1.0-1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[index] = self.beta * self.target[index] + (1-self.beta) * ((y_pred_)/(y_pred_).sum(dim=1,keepdim=True))
        ce_loss = F.cross_entropy(output, label)
        elr_reg = ((1-(self.target[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss + self.lam*elr_reg
        return final_loss