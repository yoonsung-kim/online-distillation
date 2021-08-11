import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from utils.train import test_preprocess_overhead
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms

# configurations
input_channels = 3
input_height = 224
input_width = input_height
image_size = (input_height, input_width)
lr = 0.01

iterations = 64
batch_size = 8
num_workers = 32

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation([-180, 180]),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

dataset = ImageFolder(root="/data/imagenet/train", transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=num_workers)

model = models.vgg16().to(device)

optimizer = optim.SGD(model.parameters(), lr=lr)
loss_function = torch.nn.CrossEntropyLoss()

model_name = "vgg16"

config = {
    'device': device,
    'model': model,
    'loss_function': loss_function,
    'input_shape': (batch_size, 3, input_height, input_width),
    "model_name": model_name,
    "batch_size": batch_size,
    "iterations": iterations,
    "optimizer": optimizer,
    "train_data_loader": dataloader,
    "output_file_path": f"preproc-analysis-{model_name}-batch-{batch_size}.json"
}

test_preprocess_overhead(config)
print(f"saved json output: {config['output_file_path']}")
