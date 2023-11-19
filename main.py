import os
import argparse

import torch
from torch.optim import AdamW
from transformers import Trainer, TrainingArguments, DetrForObjectDetection, DetrImageProcessor

from get_data import CocoDataset
from utils import draw_image

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
    train_dataset, val_dataset, test_dataset = dataset.get_dataset()

    if args.exec_mode == "train":
        training_args = TrainingArguments(
            output_dir=OUTPUTS,
            logging_dir="./logs",
            logging_steps=10*args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            save_steps=50*args.epochs,
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
        
        if not os.path.exists(SAVE_MODEL):
            os.mkdir(SAVE_MODEL)
        model.save_pretrained(os.path.join(SAVE_MODEL, "model"))

        test_result = trainer.predict(test_dataset)
        print(f"Cross Entropy Loss : {test_result.predictions[0]['loss_ce']}")

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