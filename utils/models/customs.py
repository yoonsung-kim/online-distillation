import torch
import torch.nn as nn


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

        return torch.stack(outputs, dim=0)


class MultiModel_4(nn.Module):
    def __init__(self):
        super(MultiModel_4, self).__init__()

        self.count = 4
        self.model_0 = SingleModel()
        self.model_1 = SingleModel()
        self.model_2 = SingleModel()
        self.model_3 = SingleModel()

    def forward(self, x):
        outputs = [self.model_0(x),
                   self.model_1(x),
                   self.model_2(x),
                   self.model_3(x)]

        return torch.stack(outputs, dim=0)


class MultiModel_8(nn.Module):
    def __init__(self):
        super(MultiModel_8, self).__init__()

        self.count = 8
        self.model_0 = SingleModel()
        self.model_1 = SingleModel()
        self.model_2 = SingleModel()
        self.model_3 = SingleModel()
        self.model_4 = SingleModel()
        self.model_5 = SingleModel()
        self.model_6 = SingleModel()
        self.model_7 = SingleModel()

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
