"""Generic convnet image classification model.

Uses pretrainedmodels from Cadene: https://github.com/Cadene/pretrained-models.pytorch
"""

import logging
from argparse import Namespace
from collections import OrderedDict

import pretrainedmodels
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from test_tube import HyperOptArgumentParser

from data_loader.data_loaders import FolderDataLoader
from modules import losses
from modules.modules import AdaptiveConcatPool2d
from optim.lr_schedulers import CosineAnnealingLR
from optim.optimizers import Adam


class ImgClassModule(pl.LightningModule):
    def __init__(self, hparams: Namespace):
        """
        Parameters
        ----------
        hparams : Namespace
        """
        super().__init__()
        self.hparams = hparams

        # Set dummy tensor to get a print out of sizes coming into
        # and out of every layer
        self.example_input_array = torch.rand(10, 3, 224, 224)

        # Initialize dataloader
        self.data_loader = FolderDataLoader(
            data_dir=hparams.data_dir,
            img_size=hparams.img_size,
            batch_size=hparams.batch_size,
        )
        # Initialize loss function
        self.loss = getattr(losses, hparams.loss)

        self.__build_model()

    def __build_model(self):
        """Initialize model."""
        # Initialize base architecture
        self.model = pretrainedmodels.__dict__[self.hparams.arch](
            pretrained="imagenet" if self.hparams.pretrained else None
        )

        # Freeze model (except for final classifier layer)
        if self.hparams.freeze:
            for params in self.model.parameters():
                params.requires_grad = False

        # Add custom pooling layer
        self.model.avgpool = AdaptiveConcatPool2d(1)

        # Replace classifier layer
        # TODO: Add option to stack more layers
        self.model.last_linear = nn.Linear(
            self.model.last_linear.in_features * 2, self.data_loader.num_classes
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch

        out = self.foward(x)
        loss = self.loss(out, y)

        tqdm_dict = {"train_loss": loss}

        output = OrderedDict(
            {"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )

        return output

    def validation_step(self, batch, batch_nb):
        x, y = batch

        out = self.foward(x)
        loss = self.loss(out, y)

        # Log example images
        sample_imgs = x[:6]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("example_images", grid, 0)

        # Calculate metrics
        labels_hat = torch.argmax(out, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss.device.index)

        output = OrderedDict({"val_loss": loss, "val_acc": val_acc,})

        return output

    def validation_end(self, outputs):
        val_loss_mean = 0
        val_acc_mean = 0

        for output in outputs:
            val_loss_mean += output["val_loss"]
            val_acc_mean += output["val_acc"]

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {"val_loss": val_loss_mean, "val_acc": val_acc_mean}

        # show val_loss and val_acc in progress bar but only log val_losses
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "val_loss": val_loss_mean,
        }

        return result

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        # scheduler = CosineAnnealingLR(optimizer, T_max=10)

        # return [optimizer], [scheduler]
        return optimizer

    @pl.data_loader
    def train_dataloader(self):
        logging.info("Training data loader called")
        return self.data_loader.get_train_dataloader()

    @pl.data_loader
    def val_dataloader(self):
        logging.info("Val data loader called")
        return self.data_loader.get_val_dataloader()

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        parser = HyperOptArgumentParser(
            strategy=parent_parser.strategy, parents=[parent_parser]
        )

        # Model
        parser.add_argument(
            "--arch", type=str, required=True, help="Name of architecture"
        )
        parser.add_argument(
            "--pretrained",
            dest="pretrained",
            action="store_true",
            help="Whether to use imagenet weights",
        )
        parser.add_argument(
            "--freeze",
            dest="freeze",
            action="store_true",
            help="Whether to freeze base arch (except final classifier layer)",
        )
        parser.add_argument(
            "--loss", type=str, default="cross_entropy", help="Name of loss function"
        )

        # Dataloading
        parser.add_argument(
            "--data_dir", type=str, required=True, help="Path to dataset folder"
        )
        parser.add_argument(
            "--img_size",
            type=int,
            default=224,
            help="Size of image to resize to, default is 224",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=32,
            help="batch size for training, default is 32",
        )

        # Training
        parser.add_argument(
            "--epochs",
            type=int,
            default=2,
            help="Number of training epochs, default is 2",
        )
        parser.add_argument(
            "--lr", type=float, default=1e-3, help="Learning rate, default is 1e-3"
        )

        return parser
