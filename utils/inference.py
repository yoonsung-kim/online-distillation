import time
import json

import torch
from torch.profiler import profile, record_function, ProfilerActivity

import numpy as np

from utils.losses import *


def test_inference(config):
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = config['device']
    print(f'device {device}')

    model = config['model']
    cnt_model = model.count
    print(f'model count: {cnt_model}')
    #model = model.to(device)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # check Nimble availability and optimize a model
    if config['use_nimble']:
        dummy_input = torch.zeros(config['input_shape']).to(device)
        model = torch.cuda.Nimble(model)
        model.prepare(dummy_input, training=True, use_multi_stream=config['use_multi_stream'])

    if 'test_data_loader' in config:
        input_data, target = next(iter(config['test_data_loader']))
    else:
        input_data = torch.randn(config['input_shape'])
        target = torch.randint(high=1, size=(config['input_shape'][0],), dtype=torch.int64)

    loss_function = config['loss_function']

    print(f'test one iteration...')

    million = 1000_000.0

    dict = {
        'elapsed_time': {
            'unit': 'millisecond',
            'inference': 0.0,
            'loss_calculation': 0.0,
        }
    }

    elapsed_time = dict['elapsed_time']

    model.eval()
    input_data = input_data.to(device)
    target = target.to(device)

    start = time.time_ns()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function('infernce'):
            for i in range(1):
                outputs = model(input_data)
    end = time.time_ns()
    elapsed_time['inference'] = (end - start) / million

    prof.export_chrome_trace(config['output_trace_path'])

    start = time.time_ns()
    loss = loss_function(outputs, target)
    end = time.time_ns()
    elapsed_time['loss_calculation'] = (end - start) / million

    print(f'stop inference')

    with open(f'{config["output_file_path"]}', 'w', encoding='utf-8') as f:
        json.dump(dict, f, ensure_ascii=False)


def usual_inference(config):
    #TODO: Implement when dataset is given for the inference
    raise ValueError('Not implemented')

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = config['device']
    print(f'device {device}')

    model = config['model']
    cnt_model = model.count
    print(f'model count: {cnt_model}')
    #model = model.to(device)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # check Nimble availability and optimize a model
    if config['use_nimble']:
        dummy_input = torch.zeros(config['input_shape']).to(device)
        model = torch.cuda.Nimble(model)
        model.prepare(dummy_input, training=True)

    train_losses = 0.0

    train_data_loader = config['train_data_loader']
    cnt_train_data = len(train_data_loader)

    valid_data_loader = config['valid_data_loader']
    cnt_valid_data = len(valid_data_loader)

    chances = 2
    remain_chances = chances

    prev_valid_loss = float("inf")

    # instances for backpropagation & updating weights
    optimizer = config['optimizer']
    loss_function = config['loss_function']

    start = time.time_ns()
    for epoch in range(config['epochs']):
        print(f'epoch: {epoch}')
        for batch_idx, (input_data, target) in enumerate(train_data_loader, 0):
            optimizer.zero_grad()

            input_data = input_data.to(device)
            target = target.to(device)

            outputs = model(input_data)
            loss = loss_function(outputs, target)
            train_losses += loss.item()

            loss.backward()
            optimizer.step()

            if batch_idx % 10000 == 0:
                print(f'avg loss among models: {loss}')

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

            #with open(f'{config["output_file_path"]}', 'w') as f:
            #    f.write(f'{epoch + 1} {elapsed_time} {elapsed_time / float(epoch + 1)} {best_accuracy}')
            break


def inference(config, test_one_iter):
    if test_one_iter:
        test_inference(config)
    else:
        usual_inference(config)
