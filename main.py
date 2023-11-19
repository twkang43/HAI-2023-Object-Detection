import os
import argparse

import torch
from pytorch_lightning import Trainer
from transformers import DetrForObjectDetection, DetrImageProcessor, DetrConfig

from get_data import CocoDataset
from utils import draw_image
from models import DETR

HOME = os.getcwd()
OUTPUTS = os.path.join(os.getcwd(), "outputs")
SAVE_MODEL = os.path.join(os.getcwd(), "save_model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    print(f"device : {DEVICE}")

    if args.model == "detr":
        checkpoint = "facebook/detr-resnet-50"
    elif args.model == "saved":
        checkpoint = os.path.join(SAVE_MODEL, "model")
    else:
        return
    
    print(f"checkpoint : {checkpoint}")
    model = DetrForObjectDetection.from_pretrained(checkpoint, revision="no_timm")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    dataset = CocoDataset.CocoDataset(args.batch_size, processor)
    train_dataloader, val_dataloader, test_dataloader = dataset.get_dataloader()
    id2label = dataset.get_id2label()

    if args.exec_mode == "train":
        model = DETR.DETR(
            lr=args.lr,
            lr_backbone=args.lr_backbone,
            weight_decay=args.weight_decay,
            checkpoint=checkpoint,
            id2label=id2label,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader
        )

        batch = next(iter(train_dataloader))
        print(f"batch.keys() : {batch.keys()}")
        outputs = model(pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"])
        print(outputs.logits.shape)

        trainer = Trainer(devices=1, accelerator="gpu", max_steps=args.epochs, gradient_clip_val=0.1)
        trainer.fit(model)

    if args.exec_mode == "eval":
        eval_trainer = Trainer(
            model=model,
            data_collator=dataset.collate_fn,
            eval_dataset=test_dataset,
        )

        with torch.no_grad():
            eval_result = eval_trainer.evaluate()
            print(f"Evaluation Result: {eval_result}")

        # test_dataset 내 이미지 랜덤으로 그리기
        draw_result = draw_image.DrawImage(model, processor, test_dataset)
        draw_result.draw_image()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2023 Fall HAI Project Team 2 - Object Detection")

    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", help="Backbone lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", help="Weight Decay", type=float, default=1e-4)
    parser.add_argument("--epochs", help="Epochs", type=int, default=30)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=10)
    parser.add_argument("--exec_mode", help="Execution mode", type=str, default="eval", choices=["train", "eval"])
    parser.add_argument("--model", help="Vanilla DETR or Saved Model", type=str, default="detr", choices=["detr", "saved"])

    args = parser.parse_args()
    
    # Print Arguments
    print("------------ Arguments -------------")
    for key, value in vars(args).items():
        print(f"{key} : {value}")
    print("------------------------------------")
    
    main(args)