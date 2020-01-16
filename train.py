"""Training script."""

import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
from pytorch_lightning import Trainer

from modules.img_class_module import ImgClassModule


SEED = 2334
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True


def get_args():
    parent_parser = ArgumentParser(add_help=False)

    parent_parser.add_argument(
        "--use_cpu",
        dest="use_cpu",
        action="store_true",
        help="Whether to use cpu",
    )

    # GPU args
    parent_parser.add_argument(
        "--gpus", type=int, default=1, help="How many gpus, default is 1"
    )
    parent_parser.add_argument(
        "--use_16bit",
        dest="use_16bit",
        action="store_true",
        help="Whether to use 16 bit precision",
    )
    parent_parser.add_argument(
        "--distributed-backend",
        type=str,
        default="dp",
        choices=("dp", "ddp", "ddp2"),
        help="Supports three options None, dp, ddp, ddp2, default is dp",
    )

    parent_parser.add_argument(
        "--save-path",
        default="saved",
        type=str,
        help="Path to save output, default is saved",
    )
    parent_parser.add_argument(
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="Evaluate model on validation set",
    )

    # Add model specific args
    parser = ImgClassModule.add_model_specific_args(parent_parser, os.getcwd())

    return parser.parse_args()


def get_callbacks():
    pass


def main(hparams):
    model = ImgClassModule(hparams)

    if hparams.use_cpu:
        trainer = Trainer(
            default_save_path=hparams.save_path,
            max_nb_epochs=hparams.epochs,
        )
    else:
        trainer = Trainer(
            default_save_path=hparams.save_path,
            gpus=hparams.gpus,
            max_nb_epochs=hparams.epochs,
            distributed_backend=hparams.distributed_backend,
            use_amp=hparams.use_16bit,
        )

    if hparams.evaluate:
        trainer.run_evaluation()
    else:
        trainer.fit(model)


if __name__ == "__main__":
    main(get_args())
