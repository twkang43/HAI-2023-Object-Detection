import os
import argparse

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import DetrImageProcessor, DetrForObjectDetection
from coco_eval import CocoEvaluator
from tqdm import tqdm

from get_data import CocoDataset
from utils import draw_image, evaluate, predict_with_input
from models import DETR

HOME = os.getcwd()
SAVE_MODEL = os.path.join(os.getcwd(), "save_model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    print(f"device : {DEVICE}")

    if args.model == "detr" or args.model == "checkpoint":
        checkpoint = "facebook/detr-resnet-50"
    elif args.model == "saved":
        checkpoint = os.path.join(SAVE_MODEL, "model")
    else:
        return
    
    print(f"checkpoint : {checkpoint}")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    dataset = CocoDataset.CocoDataset(args.batch_size, processor)
    train_dataloader, val_dataloader, test_dataloader = dataset.get_dataloader()
    train_dataset, val_dataset, test_dataset = dataset.get_dataset()
    id2label = dataset.get_id2label()
    label2id = dataset.get_label2id()

    if not os.path.exists("input_image"):
        os.mkdir("input_image")
    if not os.path.exists("outputs"):
        os.mkdir("outputs")

    if args.exec_mode == "train" or args.exec_mode == "eval":
        model = DETR.DETR(
            lr=args.lr,
            lr_backbone=args.lr_backbone,
            weight_decay=args.weight_decay,
            checkpoint=checkpoint,
            id2label=id2label,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader
        ).to(DEVICE)

    elif args.exec_mode == "draw" or args.exec_mode == "use_model":
        model = DetrForObjectDetection.from_pretrained(checkpoint, revision="no_timm").to(DEVICE)

    if args.exec_mode == "train":
        print("Training...")

        model.model.config.id2label = id2label
        model.model.config.label2id = label2id

        checkpoint_callback = ModelCheckpoint(
            monitor="validation_loss",
            dirpath=os.path.join("save_model", "tmp_model"),
            filename="model-{epochs:02d}-{val_loss:.2f}",
            save_top_k=1,
            mode="min"
        )

        trainer = Trainer(
            devices=1, 
            accelerator="gpu",
            max_epochs=args.epochs,
            gradient_clip_val=0.1,
            accumulate_grad_batches=8,
            log_every_n_steps=5,
            callbacks=[checkpoint_callback]
        )

        if args.model == "checkpoint":
            trainer.fit(model, ckpt_path=os.path.join(SAVE_MODEL, "tmp_model", "model-epochs=00-val_loss=0.00.ckpt"))
        else:
            trainer.fit(model)

        # Train 후 모델 저장
        if not os.path.exists(SAVE_MODEL):
            os.mkdir(SAVE_MODEL)
        model.model.save_pretrained(os.path.join(SAVE_MODEL, "model"))

    elif args.exec_mode == "eval":
        print("Evaluation...")

        with torch.no_grad():
            model.eval()
            evaluator = CocoEvaluator(coco_gt=test_dataset.coco, iou_types=["bbox"])

            for _, batch in enumerate(tqdm(test_dataloader)):
                pixel_values = batch["pixel_values"].to(DEVICE)
                pixel_mask = batch["pixel_mask"].to(DEVICE)
                labels = [{k: v.to(DEVICE) for k,v in t.items()} for t in batch["labels"]]

                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

                original_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
                results = processor.post_process_object_detection(outputs, target_sizes=original_target_sizes, threshold=0)

                predictions = {target["image_id"].item(): output for target,output in zip(labels, results)}
                predictions = evaluate.prepare_for_coco_detection(predictions)
                evaluator.update(predictions)

            evaluator.synchronize_between_processes()
            evaluator.accumulate()
            
            print(evaluator.summarize())

    elif args.exec_mode == "draw":
        print("Drawing...")

        # test_dataset 내 이미지 랜덤으로 그리기
        draw_result = draw_image.DrawImage(model, processor, test_dataset)
        draw_result.draw_image()

    elif args.exec_mode == "use_model":
        print("Prediction with input images")
        id2label = {k: v["name"] for k,v in train_dataset.coco.cats.items()}

        predict = predict_with_input.InputPrediction(
            model=model, 
            processor=processor, 
            id2label=id2label,
            device=DEVICE
        )

        # input image 가져오기
        try:
            input_image, image_name = predict.get_input()
        except Exception as e:
            print(f"ERROR! : {e}")
            return

        # input image에 대한 prediction
        prediction = predict.predict(input_image, threshold=0.2)

        # prediction 시각화 & 식재료 set 저장
        predict.draw_bbox(input_image, image_name, prediction)
        predict.save_ingredients_set(prediction, image_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2023 Fall HAI Project Team 2 - Object Detection")

    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", help="Backbone lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", help="Weight Decay", type=float, default=1e-4)
    parser.add_argument("--epochs", help="Epochs", type=int, default=30)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=10)
    parser.add_argument("--exec_mode", help="Execution mode", type=str, default="eval", choices=["train", "eval", "draw", "use_model"])
    parser.add_argument("--model", help="Vanilla DETR or Saved Model", type=str, default="detr", choices=["detr", "saved", "checkpoint"])

    args = parser.parse_args()
    
    # Print Arguments
    print("------------ Arguments -------------")
    for key, value in vars(args).items():
        print(f"{key} : {value}")
    print("------------------------------------")
    
    main(args)