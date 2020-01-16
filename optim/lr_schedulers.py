"""Contains learning rate schedulers.
"""

from torch.optim.lr_scheduler import (
    CyclicLR,  # step called after each batch
    OneCycleLR,  # step called after each batch
    CosineAnnealingWarmRestarts,  # step called after each batch
    CosineAnnealingLR,  # step called after each epoch
)
