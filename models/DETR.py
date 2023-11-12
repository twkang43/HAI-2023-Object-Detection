from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch.optim as optim
import pytorch_lightning as pl
from transformers import DetrForObjectDetection

class DETR(pl.LightningModule):
    def __init__(self, lr, lr_backbone):
        super(DETR, self).__init__()
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    def forward(self, x):
        pass

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        pass
        