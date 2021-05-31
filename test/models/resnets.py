import torch
import torch.nn as nn
from torchvision import models


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


class MultiResNet18_4(nn.Module):
    def __init__(self):
        super(MultiResNet18_4, self).__init__()

        self.count = 4
        self.model_0 = SingleResNet18()
        self.model_1 = SingleResNet18()
        self.model_2 = SingleResNet18()
        self.model_3 = SingleResNet18()

    def forward(self, x):
        outputs = [self.model_0(x),
                   self.model_1(x),
                   self.model_2(x),
                   self.model_3(x)]

        return torch.stack(outputs, dim=0)


class MultiResNet18_8(nn.Module):
    def __init__(self):
        super(MultiResNet18_8, self).__init__()

        self.count = 8
        self.model_0 = SingleResNet18()
        self.model_1 = SingleResNet18()
        self.model_2 = SingleResNet18()
        self.model_3 = SingleResNet18()
        self.model_4 = SingleResNet18()
        self.model_5 = SingleResNet18()
        self.model_6 = SingleResNet18()
        self.model_7 = SingleResNet18()

    def forward(self, x):
        outputs = [self.model_0(x),
                   self.model_1(x),
                   self.model_2(x),
                   self.model_3(x),
                   self.model_4(x),
                   self.model_5(x),
                   self.model_6(x),
                   self.model_7(x)]

        return torch.stack(outputs, dim=0)
