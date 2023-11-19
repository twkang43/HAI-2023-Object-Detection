import torch
import torch.optim as optim
import pytorch_lightning as pl
from transformers import DetrForObjectDetection

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DETR(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay, checkpoint, id2label, train_dataloader, val_dataloader):
        super().__init__()
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.train_data = train_dataloader
        self.val_data = val_dataloader
        self.model = DetrForObjectDetection.from_pretrained(checkpoint, num_labels=len(id2label), ignore_mismatched_sizes=True)

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    
    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad], "lr": self.lr_backbone},
        ]
        return optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(DEVICE) for k,v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        return outputs.loss, outputs.loss_dict
    
    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)

        self.log("training_loss", loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)

        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def train_dataloader(self):
        return self.train_data
    
    def val_dataloader(self):
        return self.val_data