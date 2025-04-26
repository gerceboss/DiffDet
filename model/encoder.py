import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ImageEncoder, self).__init__()
        # Load ResNet-18 backbone
        resnet = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
        self.out_channels = 512  # Output feature size
        
    def forward(self, x):
        return self.backbone(x)
