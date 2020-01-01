from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from pathlib import Path
from typing import Union

from base import BaseDataLoader
from data_loader.preprocess import image_transform


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class FolderDataLoader(DataLoader):
    """Generic dataloader class for images grouped by train/val and class label.

    Requires path to image directory, and performs a set of standard image 
    transformations. 
    """

    def __init__(self, 
                 data_dir: Union[str, Path],
                 img_size: int = 224,
                 batch_size: int = 64,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 ) -> None:
        """Initializes the train dataloader.
        
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
        train_dir = data_dir/'train'
        val_dir = data_dir/'val'

        # Initialize transforms
        train_transform, val_transform = image_transform(img_size)

        # Initialize datasets
        self.train_dataset = ImageFolder(train_dir, transform=train_transform)
        self.val_dataset = ImageFolder(val_dir, transform=val_transform)

        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers
        }

        # Initialize train dataloader
        super().__init__(self.train_dataset, **self.init_kwargs)

    def split_validation(self) -> DataLoader:
        """Initializes the val dataloader.
        
        Returns
        -------
        DataLoader
            Dataloader for the validation dataset.
        """

        return DataLoader(self.val_dataset, **self.init_kwargs)