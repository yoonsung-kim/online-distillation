#%%

import torchvision.models

import utils
from utils import models
from utils import losses

import torch
from torch import optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

#%%

# configurations
batch_size = 16
input_channels = 3
input_width = 224
input_height = input_width

lr = 0.01
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%

data_root = '../data/cifar-10'

data_transforms = transforms.Compose([
    transforms.Resize((input_width, input_height)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010))
])

train_data = datasets.CIFAR10(root=data_root,
                              train=True,
                              transform=data_transforms,
                              download=True)
train_data_loader = DataLoader(train_data,
                               batch_size=batch_size,
                               shuffle=False)

#%%

import torch.nn as nn

class OfflineKD(nn.Module):
    def __init__(self, teacher, student):
        super(OfflineKD, self).__init__()

        self.count = 2

        self.teacher = teacher

        # freeze teacher module
        for child in self.teacher.children():
            for param in child.parameters():
                param.requires_grad = False

        self.student = student

    def forward(self, x):
        teacher_logits = self.teacher(x)
        student_logits = self.student(x)

        #soft_labels = self.teacher_softmax_t(teacher_output / self.temperature)
        #soft_predictions = self.student_softmax_t(student_output / self.temperature)
        #hard_predictions = self.student_softmax(student_output)

        outputs = [teacher_logits,
                   student_logits]

        return torch.stack(outputs, dim=0)

class CrossEntropyLossWithLogits(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLossWithLogits, self).__init__()

    def forward(self, outputs, target):
        loss = torch.mean(-1.0 * torch.sum(target * torch.log(outputs), dim=1))

        return loss

class OfflineKDLoss(torch.nn.Module):
    def __init__(self, temperature, distillation_loss_ratio=0.5):
        super(OfflineKDLoss, self).__init__()

        self.temperature = temperature
        self.distillation_loss_ratio = distillation_loss_ratio

        self.teacher_softmax_t = nn.Softmax(dim=1)
        self.student_softmax_t = nn.Softmax(dim=1)
        self.student_softmax = nn.Softmax(dim=1)

    def forward(self, outputs, target):
        cee = torch.nn.CrossEntropyLoss()
        dist_cee = CrossEntropyLossWithLogits()

        teacher_logits = outputs[0]
        student_logits = outputs[1]

        cnt_outputs = outputs.shape[0]

        if cnt_outputs != 2:
            raise ValueError(f'offline distillation must have 2 logits')

        soft_labels = self.teacher_softmax_t(teacher_logits / self.temperature)
        soft_predictions = self.student_softmax_t(student_logits / self.temperature)

        distillation_loss = dist_cee(soft_predictions, soft_labels)
        student_loss = cee(student_logits, target)

        loss = self.distillation_loss_ratio * distillation_loss + \
               (1.0 - self.distillation_loss_ratio) * student_loss

        return loss

#%%

# training configuration

model = OfflineKD(teacher=torchvision.models.resnet152(pretrained=True),
                  student=torchvision.models.mobilenet_v2(pretrained=False)).to(device)

optimizer = optim.SGD(model.parameters(), lr=lr)
loss_function = OfflineKDLoss(temperature=10, distillation_loss_ratio=0.5)

train_config = {
    'device': device,
    'model': model,
    'optimizer': optimizer,
    'loss_function': loss_function,
    'epochs': 100,
    'output_file_path': 'test_multi_model.json',
    'use_nimble': True,
    'input_shape': (batch_size, 3, input_height, input_width),
    'train_data_loader': train_data_loader,
}

utils.train(train_config, test_one_iter=True)

#%%
