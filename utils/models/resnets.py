import torch
import torch.nn as nn
from torchvision import models

class SingleResNet34(nn.Module):
    def __init__(self, pretrained=False):
        super(SingleResNet34, self).__init__()
        self.count = 1

        self.model = models.resnet34(pretrained=pretrained)

    def forward(self, x):
        output = self.model(x)

        return output
