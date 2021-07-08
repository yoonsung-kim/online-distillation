import torch
import torch.nn as nn
from torchvision import models


class SingleMobileNetV2(nn.Module):
    def __init__(self):
        super(SingleMobileNetV2, self).__init__()
        self.count = 1

        self.model = models.mobilenet_v2()

    def forward(self, x):
        output = self.model(x)

        return output


class MultiMobileNetV2_2(nn.Module):
    def __init__(self):
        super(MultiMobileNetV2_2, self).__init__()

        self.count = 2
        self.model_0 = SingleMobileNetV2()
        self.model_1 = SingleMobileNetV2()

    def forward(self, x):
        outputs = [self.model_0(x), self.model_1(x)]

        return torch.stack(outputs, dim=0)
