"""Contains any preprocessing helper functions.
"""

from torchvision import transforms
from typing import Tuple


def image_transform(size: int = 224,
                    ) -> Tuple[transforms.Compose, transforms.Compose]:
    """Creates standard image transformation pipeline for train and val set.
    
    Parameters
    ----------
    size : int, optional
        Dimension of image to resize to, by default 224
    
    Returns
    -------
    Tuple[transforms.Compose, transforms.Compose]
        Image transformation pipeline for train and val set.
    """

    # With data augmentation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # ImageNet statistics
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # No data augmentation
    val_transforms = transforms.Compose([
        transforms.Resize(int(size*1.15)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        # ImageNet statistics
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transforms, val_transforms
