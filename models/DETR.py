import torch
import torch.nn as nn
from transformers import DetrForObjectDetection

class DETR(nn.Module):
    def __init__(self, lr):
        super(DETR, self).__init__()
        self.lr = lr
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)