import torch
import torch.nn as nn
from .encoder import ImageEncoder
from .decoder import DetectionDecoder

class DiffusionDet(nn.Module):
    def __init__(self, num_classes=3):
        super(DiffusionDet, self).__init__()
        self.encoder = ImageEncoder(pretrained=True)
        self.decoder = DetectionDecoder(num_classes=num_classes)
    
    def forward(self, images, noisy_boxes):
        """
        Args:
            images: (B, 3, H, W)
            noisy_boxes: (B, N, 4)
        """
        feats = self.encoder(images)
        pred_boxes, pred_logits = self.decoder(noisy_boxes, feats)
        return pred_boxes, pred_logits
