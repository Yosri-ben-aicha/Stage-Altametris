
# script for training the model
import argparse
import os
import sys
import warnings
from pathlib import Path

import torch
from torchvision.transforms import v2

torch.backends.cudnn.enabled = False
warnings.filterwarnings("ignore")

ROOT = Path(os.path.abspath(__file__)).parents[2]
sys.path.append(str(ROOT))

from src.unet.dataset.dataset import SegrailsDataset
from src.unet.model.losses import Loss
from src.unet.model.optimizers import Optimizer, Scheduler
from src.unet.model.unet import Unet
from src.unet.trainer.trainer import Trainer
from src.unet.utils.parameters import train_parameters
from src.unet.utils.strings import dict2print, get_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--val_data", type=str, help="path to validation data")
    parser.add_argument("--log_dir", type=str, default="../../runs", help="path to log")
    parser.add_argument("--loss", type=str, help="loss function")
    parser.add_argument("--optimizer", type=str, help="optimizer function")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--weight_decay", type=str, help="weight decay")
    args = parser.parse_args()

    tuning_parameters = {
        "loss": str(args.loss),
        "optimizer": {
            "method": str(args.optimizer),
            "lr": float(args.lr),
            "momentum": 0.8,
            "weight_decay": float(args.weight_decay),
        },
    }

    config = train_parameters()
    model_params = config["model"]
    train_params = config["train"]
    eval_params = config["evaluate"]
    train_params.update(tuning_parameters)

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    log_filename = os.path.join(args.log_dir, "train.log")
    logger = get_logger(log_filename, "trainer", True)
    logger.info("\nNew Experiment\n")

    model = Unet(**model_params)
    logger.info("Successfully created the model with the following parameters:")
    logger.info(dict2print(model_params))
    logger.info(f"Training on {train_params['epochs']} epochs")

    checkpoint_file = os.path.join(args.log_dir, "best.pt")

    im_size = train_params["image_size"]
    logger.info(f"Images are resized to {str(im_size)}")
    transforms = v2.Compose([v2.RandomCrop(size=im_size)])
    train_dataset = SegrailsDataset(args.train_data, transforms=transforms)
    val_dataset = SegrailsDataset(args.val_data, transforms=transforms)
    logger.info(f"Loaded the train dataset: {len(train_dataset)} images")
    logger.info(f"Loaded the validation dataset: {len(val_dataset)} images")
    logger.info("\n")

    optimizer = Optimizer(**train_params["optimizer"]).attach(model)
    scheduler = Scheduler(**train_params["scheduler"]).attach(optimizer)
    loss_fn = Loss(train_params["loss"], True).func()

    trainer = Trainer(
        logger,
        train_dataset,
        val_dataset,
        model,
        loss_fn,
        optimizer,
        scheduler,
        train_params,
        eval_params,
        checkpoint_file,
    )

    trainer.train()
