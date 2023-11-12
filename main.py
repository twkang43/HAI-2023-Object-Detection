import os
import random
import numpy as np
import supervision as sv

from transformers import DetrImageProcessor
from transformers import Trainer, TrainingArguments
import torch
from PIL import Image

from get_data import CocoDataset

HOME = os.getcwd()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DATASET_PATH = os.path.join("dataset", "test")
VALID_DATASET_PATH = os.path.join("dataset", "valid")
TEST_DATASET_PATH = os.path.join("dataset", "test")

def main():
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    train_dataset = CocoDataset.CocoDetection(
        image_dir=TRAIN_DATASET_PATH,
        processor=processor,
        train=True
    )

    val_dataset = CocoDataset.CocoDetection(
        image_dir=VALID_DATASET_PATH,
        processor=processor,
        train=False
    )

    test_dataset = CocoDataset.CocoDetection(
        image_dir=TEST_DATASET_PATH,
        processor=processor,
        train=False
    )

    # Select random image
    images_id = random.choice(train_dataset.coco.getImgIds())
    print(f"Image #{images_id}")



if __name__ == "__main__":
    main()