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


class SingleModel(nn.Module):
    def __init__(self):
        super(SingleModel, self).__init__()
        self.count = 1
        # conv 0
        self.conv2d_0 = nn.Conv2d(in_channels=3,
                                  out_channels=32,
                                  kernel_size=5,
                                  stride=1,
                                  padding=2)
        self.relu_0 = nn.ReLU()
        self.max_pool2d_0 = nn.MaxPool2d(kernel_size=2,
                                         stride=2)
        self.batch_norm_0 = nn.BatchNorm2d(32)

        # conv 1
        self.conv2d_1 = nn.Conv2d(in_channels=32,
                                  out_channels=64,
                                  kernel_size=5,
                                  stride=1,
                                  padding=2)
        self.relu_1 = nn.ReLU()
        self.max_pool2d_1 = nn.MaxPool2d(kernel_size=2,
                                         stride=2)
        self.batch_norm_1 = nn.BatchNorm2d(64)

        self.flatten = nn.Flatten()

        self.fc_0 = nn.Linear(8 * 8 * 64, 500)
        self.relu_2 = nn.ReLU()

        self.fc_1 = nn.Linear(500, 10)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        output = self.conv2d_0(x)
        output = self.relu_0(output)
        output = self.max_pool2d_0(output)
        output = self.batch_norm_0(output)

        output = self.conv2d_1(output)
        output = self.relu_1(output)
        output = self.max_pool2d_1(output)
        output = self.batch_norm_1(output)

        output = self.flatten(output)

        output = self.fc_0(output)
        output = self.relu_2(output)
        output = self.fc_1(output)

        output = self.softmax(output)

        return output


class MultiModel_2(nn.Module):
    def __init__(self):
        super(MultiModel_2, self).__init__()

        self.count = 2
        self.model_0 = SingleModel()
        self.model_1 = SingleModel()

    def forward(self, x):
        outputs = [self.model_0(x),
                   self.model_1(x)]

        #return outputs
        return torch.stack(outputs, dim=0)


class SingleResNet18(nn.Module):
    def __init__(self):
        super(SingleResNet18, self).__init__()
        self.count = 1

        resnet = models.resnet18()
        resnet.fc = nn.Linear(512, 10)

        self.model = nn.Sequential(resnet, nn.Softmax(dim=1))

    def forward(self, x):
        output = self.model(x)

        return output


class MultiResNet18_2(nn.Module):
    def __init__(self):
        super(MultiResNet18_2, self).__init__()

        self.count = 2
        self.model_0 = SingleResNet18()
        self.model_1 = SingleResNet18()

    def forward(self, x):
        outputs = [self.model_0(x), self.model_1(x)]

        return torch.stack(outputs, dim=0)

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
    batch_size = 1
    #input_shape = (batch_size, 3, 224, 224)
    input_shape = (batch_size, 3, 224, 224)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MultiMobileNetV2_2().to(device)
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    if True:
        dummy_input = torch.randn(input_shape).cuda()
        model = torch.cuda.Nimble(model)
        model.prepare(dummy_input, training=True)

    for i in range(1):
        dummy_input = torch.randn(input_shape).to(device)
        output = model(dummy_input)
