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


class SingleManualAlexNet(nn.Module):
    def __init__(self, big_gpu, small_gpu):
        super(SingleManualAlexNet, self).__init__()
        self.count = 1

        self.big_gpu = big_gpu
        self.small_gpu = small_gpu

        self.conv_0 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2).to(self.big_gpu)
        self.relu_0 = nn.ReLU(inplace=False).to(self.small_gpu)
        self.max_pool_0 = nn.MaxPool2d(kernel_size=3, stride=2).to(self.small_gpu)

        self.conv_1 = nn.Conv2d(64, 192, kernel_size=5, padding=2).to(self.big_gpu)
        self.relu_1 = nn.ReLU(inplace=False).to(self.small_gpu)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=3, stride=2).to(self.small_gpu)

        self.conv_2 = nn.Conv2d(192, 384, kernel_size=3, padding=1).to(self.big_gpu)
        self.relu_2 = nn.ReLU(inplace=False).to(self.small_gpu)

        self.conv_3 = nn.Conv2d(384, 256, kernel_size=3, padding=1).to(self.big_gpu)
        self.relu_3 = nn.ReLU(inplace=False).to(self.small_gpu)

        self.conv_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1).to(self.big_gpu)
        self.relu_4 = nn.ReLU(inplace=False).to(self.small_gpu)
        self.max_pool_4 = nn.MaxPool2d(kernel_size=3, stride=2).to(self.small_gpu)

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6)).to(self.small_gpu)

        self.flatten_0 = nn.Flatten().to(self.small_gpu)
        self.dropout_0 = nn.Dropout().to(self.small_gpu)
        self.linear_0 = nn.Linear(256 * 6 * 6, 4096).to(self.big_gpu)
        self.relu_5 = nn.ReLU(inplace=False).to(self.small_gpu)

        self.dropout_1 = nn.Dropout().to(self.small_gpu)
        self.linear_1 = nn.Linear(4096, 4096).to(self.big_gpu)
        self.relu_6 = nn.ReLU(inplace=False).to(self.small_gpu)

        self.linear_2 = nn.Linear(4096, 1000).to(self.big_gpu)

    def forward(self, x):
        out = self.conv_0(x)
        out = out.to(self.small_gpu)
        out = self.relu_0(out)
        out = self.max_pool_0(out)

        out = out.to(self.big_gpu)
        out = self.conv_1(out)
        out = out.to(self.small_gpu)
        out = self.relu_1(out)
        out = self.max_pool_1(out)

        out = out.to(self.big_gpu)
        out = self.conv_2(out)
        out = out.to(self.small_gpu)
        out = self.relu_2(out)

        out = out.to(self.big_gpu)
        out = self.conv_3(out)
        out = out.to(self.small_gpu)
        out = self.relu_3(out)

        out = out.to(self.big_gpu)
        out = self.conv_4(out)
        out = out.to(self.small_gpu)
        out = self.relu_4(out)
        out = self.max_pool_4(out)

        out = self.avg_pool(out)

        out = self.flatten_0(out)
        out = self.dropout_0(out)
        out = out.to(self.big_gpu)
        out = self.linear_0(out)
        out = out.to(self.small_gpu)
        out = self.relu_5(out)

        out = self.dropout_1(out)
        out = out.to(self.big_gpu)
        out = self.linear_1(out)
        out = out.to(self.small_gpu)
        out = self.relu_6(out)

        out = out.to(self.big_gpu)
        out = self.linear_2(out)

        return out
