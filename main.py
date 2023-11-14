import os
import argparse

from pytorch_lightning import Trainer
import torch

from models import DETR

HOME = os.getcwd()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):
    model = DETR.DETR(lr=args.lr, lr_backbone=args.lr_backbone).to(DEVICE)
    trainer = Trainer(devices=1, accelerator=DEVICE, max_epochs=args.epochs)
    trainer.fit(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2023 Fall HAI Project - Object Detection")

    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", help="Backbone learning rate", type=float, default=1e-5)
    parser.add_argument("--epochs", help="Epochs", type=int, default=100)

    args = parser.parse_args()
    main(args)