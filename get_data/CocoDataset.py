import os
import torchvision
from transformers import DetrImageProcessor
from torch.utils.data import DataLoader

ANNOTATION_FILE_NAME = "_annotations.coco.json"

TRAIN_DATASET_PATH = os.path.join("dataset", "test")
VALID_DATASET_PATH = os.path.join("dataset", "valid")
TEST_DATASET_PATH = os.path.join("dataset", "test")

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, image_dir:str, processor, train=True):
        annotation_file = self.get_annotation_file_path(image_dir)
        super(CocoDetection, self).__init__(image_dir, annotation_file)
        self.processor = processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        dic_annotations = {"image_id" : image_id, "annotations" : annotations}

        encoding = self.processor(images=images, annotations=dic_annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target
    
    def get_annotation_file_path(self, image_dir):
        return os.path.join(image_dir, ANNOTATION_FILE_NAME)
    
class CocoDataLoader():
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

    def get_dataloader(self):
        train_dataloader = DataLoader(dataset=self.train_dataset, collate_fn=self.collate_fn, batch_size=4, shuffle=True)
        val_dataloader = DataLoader(dataset=self.val_dataset, collate_fn=self.collate_fn, batch_size=self.batch_size)
        test_dataloader = DataLoader(dataset=self.test_dataset, collate_fn=self.collate_fn, batch_size=self.batch_size)

        return train_dataloader, val_dataloader, test_dataloader

    def collate_fn(self, batch):
        pixel_values = [item[0] for item in batch]
        encoding = self.processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        
        return {"pixel_values" : encoding["pixel_values"], "pixel_mask" : encoding["pixel_mask"], "labels" : labels}