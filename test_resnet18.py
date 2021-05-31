import time
import torch
from test.models import *
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import OnlineDistillationLoss
import numpy as np


def train(model,
          loss_function,
          train_data_loader,
          valid_data_loader,
          target_valid_accuracy,
          epochs,
          learning_rate,
          input_shape,
          use_nimble,
          output_file_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device {device}')

    cnt_model = model.count
    print(f'model count: {cnt_model}')
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    #
    if use_nimble:
        dummy_input = torch.randn(input_shape).cuda()
        model = torch.cuda.Nimble(model)
        model.prepare(dummy_input, training=True)
    #

    train_losses = 0.0
    cnt_train_data = len(train_data_loader)
    cnt_valid_data = len(valid_data_loader)

    chances = 2
    remain_chances = chances

    prev_valid_loss = float("inf")

    start = time.time_ns()
    for epoch in range(epochs):
        print(f'epoch: {epoch}')
        for batch_idx, (input_data, target) in enumerate(train_data_loader, 0):
            optimizer.zero_grad()

            input_data = input_data.to(device)
            target = target.to(device)

            outputs = model(input_data)
            loss = loss_function(outputs, target)
            train_losses += (loss.item() / cnt_model)

            loss.backward()
            optimizer.step()

            if batch_idx % 10000 == 0:
                print(f'avg loss among models: {loss / cnt_model}')

        train_loss = train_losses / cnt_train_data

        corrects = [0.0] * cnt_model
        valid_losses = 0.0
        for batch_idx, (input_data, target) in enumerate(valid_data_loader, 0):
            input_data = input_data.to(device)
            target = target.to(device)

            output = model(input_data)

            loss = loss_function(output, target)
            valid_losses += (loss.item() / cnt_model)

            for i in range(cnt_model):
                _, predicted = torch.max(output[i], 1) if cnt_model > 1 else torch.max(output, 1)

                if predicted == target:
                    corrects[i] += 1.0

        valid_loss = valid_losses / cnt_valid_data

        should_finish = False

        valid_accuracies = (np.array(corrects) / cnt_valid_data)

        max_valid_index = np.argmax(valid_accuracies)
        max_valid_acc = np.max(valid_accuracies)

        print(f'max valid accuracy from model #{max_valid_index}: {max_valid_acc * 100.0}')

        if max_valid_acc > target_valid_accuracy:
            should_finish = True

        print(f'train loss: {train_loss}, valid loss: {valid_loss}')
        if valid_loss > prev_valid_loss:
            if remain_chances == 0:
                should_finish = True
            else:
                remain_chances -= 1
        else:
            remain_chances = chances

        prev_valid_loss = valid_loss

        if should_finish:
            end = time.time_ns()
            print(f'stop training')
            best_accuracy = max_valid_acc * 100.0
            print(f'achieved best valid accuracy: {best_accuracy}%')
            print(f'executed epochs: {epoch}')
            elapsed_time = (end - start) / 1000_000_000.0
            print(f'ett (elapsed training time)')
            print(f'total ett: {elapsed_time} seconds')
            print(f'avg ett: {elapsed_time / float(epochs + 1)} seconds')

            with open(f'{output_file_path}.txt', 'w') as f:
                f.write(f'{epoch + 1} {elapsed_time} {elapsed_time / float(epoch + 1)} {target_valid_accuracy} {best_accuracy}')
            break


if __name__ == '__main__':
    data_root = './data/cifar-10'
    batch_size = 1

    input_width = 224
    input_height = input_width

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

    test_data = datasets.CIFAR10(root=data_root,
                                 train=False,
                                 transform=data_transforms,
                                 download=True)
    test_data_loader = DataLoader(test_data,
                                  batch_size=batch_size,
                                  shuffle=False)

    lr = 0.01
    epochs = 1000
    target_valid_accuracy = 0.4

    train(model=MultiResNet18_2(),
          loss_function=OnlineDistillationLoss(),
          train_data_loader=train_data_loader,
          valid_data_loader=test_data_loader,
          target_valid_accuracy=target_valid_accuracy,
          epochs=epochs,
          learning_rate=lr,
          input_shape=[batch_size, 3, input_width, input_height],
          use_nimble=True,
          output_file_path='multi_reset18_2_nimble.txt')

    train(model=MultiResNet18_2(),
          loss_function=OnlineDistillationLoss(),
          train_data_loader=train_data_loader,
          valid_data_loader=test_data_loader,
          target_valid_accuracy=target_valid_accuracy,
          epochs=epochs,
          learning_rate=lr,
          input_shape=[batch_size, 3, input_width, input_height],
          use_nimble=False,
          output_file_path='multi_reset18_2.txt')

    '''
    train(model=MultiResNet18_4(),
          loss_function=OnlineDistillationLoss(),
          train_data_loader=train_data_loader,
          valid_data_loader=test_data_loader,
          target_valid_accuracy=target_valid_accuracy,
          epochs=epochs,
          learning_rate=lr,input_shape=[batch_size, 3, input_width, input_height],
          use_nimble=True,
          output_file_path='multi_reset18_4.txt')

    train(model=MultiResNet18_8(),
          loss_function=OnlineDistillationLoss(),
          train_data_loader=train_data_loader,
          valid_data_loader=test_data_loader,
          target_valid_accuracy=target_valid_accuracy,
          epochs=epochs,
          learning_rate=lr,
          input_shape=[batch_size, 3, input_width, input_height],
          use_nimble=True,
          output_file_path='multi_reset18_8.txt')

    train(model=SingleResNet18(),
          loss_function=nn.CrossEntropyLoss(),
          train_data_loader=train_data_loader,
          valid_data_loader=test_data_loader,
          target_valid_accuracy=target_valid_accuracy,
          epochs=epochs,
          learning_rate=lr,
          input_shape=[batch_size, 3, input_width, input_height],
          use_nimble=True,
          output_file_path='single_reset18.txt')
    '''
