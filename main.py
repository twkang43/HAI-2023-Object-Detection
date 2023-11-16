import os
import argparse
import random

import torch
from torch.optim import AdamW
from transformers import Trainer, TrainingArguments, DetrForObjectDetection

from get_data import CocoDataset
from utils import visualize

HOME = os.getcwd()
OUTPUTS = os.path.join(os.getcwd(), "outputs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    dataset = CocoDataset.CocoDataset(args.batch_size)
    train_dataset, val_dataset, test_dataset = dataset.get_dataset()

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        revision="no_timm"
    )

    training_args = TrainingArguments(
        output_dir=OUTPUTS,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        save_steps=2*args.epochs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        optimizers=(AdamW(model.parameters()), None),
        data_collator=dataset.collate_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    test_result = trainer.predict(test_dataset)

    print(test_result["loss_ce"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2023 Fall HAI Project Team 2 - Object Detection")

    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-4)
    parser.add_argument("--epochs", help="Epochs", type=int, default=30)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=4)

    args = parser.parse_args()
    
    # Print Arguments
    print("------------ Arguments -------------")
    for key, value in vars(args).items():
        print(f"{key} : {value}")
    print("------------------------------------")
    
    main(args)