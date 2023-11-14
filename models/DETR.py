import os
import torch
import torch.optim as optim
import pytorch_lightning as pl
from transformers import DetrForObjectDetection, DetrImageProcessor

from get_data import CocoDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DATASET_PATH = os.path.join("dataset", "test")
VALID_DATASET_PATH = os.path.join("dataset", "valid")
TEST_DATASET_PATH = os.path.join("dataset", "test")

class DETR(pl.LightningModule):
    def __init__(self, lr, lr_backbone):
        super(DETR, self).__init__()
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad], "lr": self.lr_backbone},
        ]
        return optim.AdamW(param_dicts, lr=self.lr)

    def training_step(self, batch):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        self.log("training_loss", outputs.loss)
        for k, v in outputs.loss_dicts.items():
            self.log("train_" + k, v.item())

        return outputs.loss
    
    def validation_step(self, batch):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch ["pixel_mask"]
        labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        self.log("validation_loss", outputs.loss)
        for k, v in outputs.loss_dicts.items():
            self.log("validation_" + k, v.item())

        return outputs.loss
    
    def train_dataloader(self):
        train_dataset = CocoDataset.CocoDetection(
            image_dir=TRAIN_DATASET_PATH,
            processor=self.processor,
            train=True
        )

        return train_dataset
    
    def val_dataloader(self):
        val_dataset = CocoDataset.CocoDetection(
            image_dir=TRAIN_DATASET_PATH,
            processor=self.processor,
            train=True
        )
        return val_dataset