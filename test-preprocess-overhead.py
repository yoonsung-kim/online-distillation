import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from utils.train import test_preprocess_overhead
import numpy as np

# configurations
batch_size = 64
input_channels = 3
input_width = 224
input_height = input_width
iterations = 256

lr = 0.01
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_root = './data/cifar-10'

data_transforms = transforms.Compose([
    transforms.Resize((input_width, input_height)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
    #                     std=(0.2023, 0.1994, 0.2010))
])

train_data = datasets.CIFAR10(root=data_root,
                              train=True,
                              transform=data_transforms,
                              download=True)
train_data_loader = DataLoader(train_data,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=16)

model = models.alexnet().to(device)

optimizer = optim.SGD(model.parameters(), lr=lr)
loss_function = torch.nn.CrossEntropyLoss()

model_name = "alexnet"

config = {
    'device': device,
    'model': model,
    'loss_function': loss_function,
    'input_shape': (batch_size, 3, input_height, input_width),
    "model_name": model_name,
    "batch_size": batch_size,
    "iterations": iterations,
    "optimizer": optimizer,
    "train_data_loader": train_data_loader,
    "output_file_path": f"preproc-analysis-{model_name}-batch-{batch_size}"
}

test_preprocess_overhead(config)
