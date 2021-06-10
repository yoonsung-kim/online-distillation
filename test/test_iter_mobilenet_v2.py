import time

import torch
import torch.autograd.profiler as profiler

import numpy as np

from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import models, datasets, transforms


class OnlineDistillationLoss(torch.nn.Module):
    def __init__(self):
        super(OnlineDistillationLoss, self).__init__()

    def forward(self, outputs, target):
        cee = torch.nn.CrossEntropyLoss()
        kld = torch.nn.KLDivLoss()

        cnt_outputs = outputs.shape[0]

        if cnt_outputs < 2:
            raise ValueError(f'online distillation loss must have at least 2 outputs')

        cpy_outputs = outputs.detach().clone()
        losses = []

        for i in range(cnt_outputs):
            other_outputs = torch.cat([cpy_outputs[0:i], cpy_outputs[i+1:]])
            loss = cee(outputs[i], target) + kld(torch.mean(other_outputs, dim=0), outputs[i])
            losses.append(loss)

        loss = losses[0] + losses[1]

        for i in range(2, cnt_outputs):
            loss += losses[i]

        return loss


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


class MultiMobileNetV2_4(nn.Module):
    def __init__(self):
        super(MultiMobileNetV2_4, self).__init__()

        self.count = 4
        self.model_0 = SingleMobileNetV2()
        self.model_1 = SingleMobileNetV2()
        self.model_2 = SingleMobileNetV2()
        self.model_3 = SingleMobileNetV2()

    def forward(self, x):
        outputs = [self.model_0(x),
                   self.model_1(x),
                   self.model_2(x),
                   self.model_3(x)]

        return torch.stack(outputs, dim=0)


if __name__ == '__main__':
    batch_size = 32
    input_shape = (batch_size, 3, 224, 224)
    model = MultiMobileNetV2_2()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    model.train()
    dummy_input = torch.randn(input_shape)
    output = model(dummy_input)
