"""Contains data loading classes."""

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path
from typing import Union

from data_loader.preprocess import image_transform


class FolderDataLoader:
    """Generic dataloader class for images grouped by train/val and class label.

    Requires path to image directory, and performs a set of standard image
    transformations.

    Attributes
    ----------
    n_classes : int
        Number of classes for the labels.
    train_dataloader : DataLoader
    val_dataloader : DataLoader
        Train and val dataloaders.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        img_size: int = 224,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> None:
        """Initialize train/val dataloaders.

        Parameters
        ----------
        data_dir : Union[str, Path]
            Path to image directory. Should contain train/val -> class folder
            structure.
        img_size : int, optional
            Size of image to resize to, by default 224
        batch_size : int, optional
            by default 64
        shuffle : bool, optional
            Whether to shuffle dataset after each epoch, by default True
        num_workers : int, optional
            by default 0
        """
        # Initialize directories
        data_dir = Path(data_dir)
        train_dir = data_dir / "train"
        val_dir = data_dir / "val"

        # Initialize transforms
        train_transform, val_transform = image_transform(img_size)

        # Initialize datasets
        train_dataset = ImageFolder(train_dir, transform=train_transform)
        val_dataset = ImageFolder(val_dir, transform=val_transform)

        assert len(train_dataset.classes) == len(
            val_dataset.classes
        ), "Train and val dataset should have same number of classes."

        # Init number of classes
        self.n_classes = len(train_dataset.classes)

        init_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
        }

        # Initialize dataloaders
        self.train_dataloader = DataLoader(train_dataset, **init_kwargs)
        self.val_dataloader = DataLoader(val_dataset, **init_kwargs)
