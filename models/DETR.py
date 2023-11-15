import torch
import torch.optim as optim
import pytorch_lightning as pl
from transformers import DetrForObjectDetection

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DETR(pl.LightningModule):
    def __init__(self, lr, lr_backbone, train_dataloader, val_dataloader, test_dataloader):
        super(DETR, self).__init__()
        self.lr = lr
        self.lr_backbone = lr_backbone

        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            revision="no_timm"
        )

        self.train_data = train_dataloader
        self.val_data = val_dataloader
        self.test_data = test_dataloader

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
        for k, v in outputs.loss_dict.items():
            self.log("train_" + k, v.item())

        return outputs.loss
    
    def validation_step(self, batch):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch ["pixel_mask"]
        labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        self.log("validation_loss", outputs.loss)
        for k, v in outputs.loss_dict.items():
            self.log("validation_" + k, v.item())

        return outputs.loss
    
    def train_dataloader(self):
        return self.train_data
    
    def val_dataloader(self):
        return self.val_data
    
    def test_dataloader(self):
        return self.test_data