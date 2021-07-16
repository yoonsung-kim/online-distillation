import torch
import torch.nn as nn
from torchvision import models

class SingleAlexNet(nn.Module):
    def __init__(self, pretrained=False):
        super(SingleAlexNet, self).__init__()
        self.count = 1

        self.model = models.alexnet(pretrained=pretrained)

    def forward(self, x):
        output = self.model(x)

        return output
