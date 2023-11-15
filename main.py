import os
import argparse

import torch
from torch.optim import AdamW
from transformers import Trainer
from transformers import TrainingArguments

from get_data import CocoDataset
from models import DETR

HOME = os.getcwd()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    CocoDataLoader = CocoDataset.CocoDataLoader(args.batch_size)
    train_dataloader, val_dataloader, test_dataloader = CocoDataLoader.get_dataloader()

    model = DETR.DETR(
        lr=args.lr,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader
    ).to(DEVICE)

    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr
    )

    trainer = Trainer(
        model=model,
        optimizers=(AdamW(model.parameters()), None),
        args=training_args,
        train_dataset=train_dataloader,
        eval_dataset=val_dataloader
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2023 Fall HAI Project Team 2 - Object Detection")

    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-4)
    parser.add_argument("--epochs", help="Epochs", type=int, default=100)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=4)

    args = parser.parse_args()
    main(args)