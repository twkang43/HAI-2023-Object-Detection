import os
import argparse

import torch
from pytorch_lightning import Trainer

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

    trainer = Trainer(
        devices=1,
        accelerator="gpu",
        max_epochs=args.epochs,
        gradient_clip_val=0.1,
        accumulate_grad_batches=8,
        log_every_n_steps=5
    )

    trainer.fit(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2023 Fall HAI Project - Object Detection")

    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", help="Backbone learning rate", type=float, default=1e-5)
    parser.add_argument("--epochs", help="Epochs", type=int, default=100)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=4)

    args = parser.parse_args()
    main(args)