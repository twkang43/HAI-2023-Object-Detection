import os
import argparse

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from get_data import CocoDataset
from models import DETR

HOME = os.getcwd()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    CocoDataLoader = CocoDataset.CocoDataLoader(args.batch_size)
    train_dataloader, val_dataloader, test_dataloader = CocoDataLoader.get_dataloader()

    model = DETR.DETR(
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader
    ).to(DEVICE)

    batch = next(iter(train_dataloader))
    outputs = model(pixel_values=batch["pixel_values"].to(DEVICE), pixel_mask=batch["pixel_mask"].to(DEVICE))
    print(outputs.logits.shape)

    logger = TensorBoardLogger(save_dir="logs/")

    trainer = Trainer(
        devices=1,
        accelerator="gpu",
        max_epochs=args.epochs,
        log_every_n_steps=5,
        logger=logger
    )

    trainer.fit(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2023 Fall HAI Project Team 2 - Object Detection")

    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", help="Backbone learning rate", type=float, default=1e-5)
    parser.add_argument("--epochs", help="Epochs", type=int, default=100)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=4)

    args = parser.parse_args()
    main(args)