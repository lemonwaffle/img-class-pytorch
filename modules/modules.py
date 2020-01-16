"""Contains common abstractions of layers/modules.
"""

import torch
from torch import nn


class AdaptiveConcatPool2d(nn.Module):
    """Pooling layer that concats both the avg and max operation.
    
    Typically placed as the final layer after the convolutional layers.
    Taken from fast.ai/course-v3/dl2/11a.ipynb.
    """

    def __init__(self, output_size: int = 1) -> None:
        """
        Parameters
        ----------
        output_size : int, optional
            Size of feature map to downsample to, by default 1
        """
        super().__init__()

        self.output_size = output_size
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size)
        self.max_pool = nn.AdaptiveMaxPool2d(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Outputs tensor of size: (B, 2*C, H, W)
        """
        # Concats the outputs from both poolings along the channels dim
        return torch.cat([
            self.max_pool(x),
            self.avg_pool(x)
        ], 1)
