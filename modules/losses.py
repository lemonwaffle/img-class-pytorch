"""Contains loss/metric modules.

All functions should contain the input signature (y_hat, y).
"""

from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class LabelSmoothingCrossEntropy(nn.Module):
    """Taken from: https://github.com/pytorch/pytorch/issues/7455
    """

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.0)
        weight.scatter_(-1, target.unsqueeze(-1), (1.0 - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()

        return loss
