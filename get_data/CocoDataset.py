import os
import torchvision
from transformers import DetrImageProcessor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.io as io
import numpy as np

ANNOTATION_FILE_NAME = "_annotations.coco.json"

TRAIN_DATASET_PATH = os.path.join("dataset", "train")
VALID_DATASET_PATH = os.path.join("dataset", "valid")
TEST_DATASET_PATH = os.path.join("dataset", "test")

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, image_dir:str, processor, train=True):
        annotation_file = self.get_annotation_file_path(image_dir)
        super(CocoDetection, self).__init__(image_dir, annotation_file)
        self.processor = processor

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return {"pixel_values": pixel_values, "labels": target}
    
    def get_annotation_file_path(self, image_dir):
        return os.path.join(image_dir, ANNOTATION_FILE_NAME)
    
class CocoDataset():
    def __init__(self, batch_size):
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.batch_size = batch_size

        self.train_dataset = CocoDetection(
            image_dir=TRAIN_DATASET_PATH,
            processor=self.processor,
            train=True
        )

        self.val_dataset = CocoDetection(
            image_dir=VALID_DATASET_PATH,
            processor=self.processor,
            train=False
        )

        self.test_dataset = CocoDetection(
            image_dir=TEST_DATASET_PATH,
            processor=self.processor,
            train=False
        )
    
    def get_dataset(self):
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def collate_fn(self, batch):
        pixel_values = [item["pixel_values"] for item in batch]
        encoding = self.processor.pad(pixel_values, return_tensors="pt")
        labels = [item["labels"] for item in batch]

        batch = {}
        batch["pixel_values"] = encoding["pixel_values"]
        batch["pixel_mask"] = encoding["pixel_mask"]
        batch["labels"] = labels

        return batch