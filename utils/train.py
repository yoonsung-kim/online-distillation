import time
import json

import torch

import numpy as np

from tqdm import tqdm

from utils.losses import *

def test_preprocess_overhead(config):
    device = config['device']
    print(f'device {device}')

    model = config['model']

    train_data_loader = config['train_data_loader']

    # instances for backpropagation & updating weights
    optimizer = config['optimizer']
    loss_function = config['loss_function']

    train_data_iter = iter(train_data_loader)

    iterations = config["iterations"]

    million = 1000_000.0

    results = {
        "model-name": config["model_name"],
        "iterations": config["iterations"],
        "batch-size": config["batch_size"],
        "ett": {
            "unit": "millisecond",
            "batch-load": [],
            "data-copy-to-gpu": [],
            "train": [],
            "total": [],
        }
    }
    
    
    for _ in range(iterations):
    #for _ in tqdm(range(iterations)):
        batch_load_start = time.time_ns()
        inputs, targets = next(train_data_iter)
        batch_load_end = time.time_ns()
        batch_load_ett = (batch_load_end - batch_load_start) / million

        data_cpy_to_gpu_start = time.time_ns()
        inputs = inputs.to(device)
        targets = targets.to(device)
        data_cpy_to_gpu_end = time.time_ns()
        data_cpy_to_gpu_ett = (data_cpy_to_gpu_end - data_cpy_to_gpu_start) / million

        train_start = time.time_ns()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_end = time.time_ns()
        train_ett = (train_end - train_start) / million

        results["ett"]["batch-load"].append(batch_load_ett)
        results["ett"]["data-copy-to-gpu"].append(data_cpy_to_gpu_ett)
        results["ett"]["train"].append(train_ett)
        results["ett"]["total"].append(batch_load_ett + data_cpy_to_gpu_ett + train_ett)

    with open(f'{config["output_file_path"]}', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)


def test_batch_size_train(config):
    device = config['device']
    print(f'device {device}')

    model = config['model']

    # instances for backpropagation & updating weights
    optimizer = config['optimizer']
    loss_function = config['loss_function']

    print(f'test one iteration...')

    million = 1000_000.0

    dict = {
        'elapsed_time': {
            'unit': 'millisecond',
            'batch_sizes': [],
            'forward': [],
            'loss_calculation': [],
            'backward': [],
            'set_gradients_zero': [],
            'update_weights': [],
            'iteration_sum': []
        }
    }

    elapsed_time = dict['elapsed_time']

    batchs = config['batch-sizes']
    iters = config['iterations']
    input_shape_chw = config['input-shape-chw']

    for batch in batchs:
        elapsed_time['batch_sizes'].append(batch)

        forward = 0.0
        loss_calculation = 0.0
        set_gradients_zero = 0.0
        backward = 0.0
        update_weights = 0.0

        for i in range(iters):
            input_data = torch.randn((batch,) + input_shape_chw).to(device)
            target = torch.randint(high=1, size=(batch, 1)).to(device)

            start = time.time_ns()
            outputs = model(input_data)
            end = time.time_ns()
            forward += (end - start)

            start = time.time_ns()
            loss = loss_function(outputs, target)
            end = time.time_ns()
            loss_calculation += (end - start)

            start = time.time_ns()
            optimizer.zero_grad()
            end = time.time_ns()
            set_gradients_zero += (end - start)

            start = time.time_ns()
            loss.backward()
            end = time.time_ns()
            backward += (end - start)

            start = time.time_ns()
            optimizer.step()
            end = time.time_ns()
            update_weights += (end - start)

        iteration_sum = forward + loss_calculation + set_gradients_zero + backward + update_weights

        forward /= (iters * million)
        loss_calculation /= (iters * million)
        set_gradients_zero /= (iters * million)
        backward /= (iters * million)
        update_weights /= (iters * million)
        iteration_sum /= (iters * million)

        elapsed_time['forward'].append(forward)
        elapsed_time['loss_calculation'].append(loss_calculation)
        elapsed_time['set_gradients_zero'].append(set_gradients_zero)
        elapsed_time['backward'].append(backward)
        elapsed_time['update_weights'].append(update_weights)
        elapsed_time['iteration_sum'].append(iteration_sum)

    print(f'stop training')

    with open(f'{config["output_file_path"]}', 'w', encoding='utf-8') as f:
        json.dump(dict, f, ensure_ascii=False)


def test_train(config):
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

    train_data_loader = config['train_data_loader']

    # instances for backpropagation & updating weights
    optimizer = config['optimizer']
    loss_function = config['loss_function']

    print(f'test one iteration...')

    million = 1000_000.0

    dict = {
        'elapsed_time': {
            'unit': 'millisecond',
            'forward': 0.0,
            'loss_calculation': 0.0,
            'backward': 0.0,
            'set_gradients_zero': 0.0,
            'update_weights': 0.0
        }
    }

    elapsed_time = dict['elapsed_time']

    for batch_idx, (input_data, target) in enumerate(train_data_loader, 0):

        input_data = input_data.to(device)
        target = target.to(device)

        start = time.time_ns()
        outputs = model(input_data)
        end = time.time_ns()
        elapsed_time['forward'] = (end - start) / million

        start = time.time_ns()
        loss = loss_function(outputs, target)
        end = time.time_ns()
        elapsed_time['loss_calculation'] = (end - start) / million

        start = time.time_ns()
        optimizer.zero_grad()
        end = time.time_ns()
        elapsed_time['set_gradients_zero'] = (end - start) / million

        start = time.time_ns()
        loss.backward()
        end = time.time_ns()
        elapsed_time['backward'] = (end - start) / million

        start = time.time_ns()
        optimizer.step()
        end = time.time_ns()
        elapsed_time['update_weights'] = (end - start) / million

        break

    print(f'stop training')

    with open(f'{config["output_file_path"]}', 'w', encoding='utf-8') as f:
        json.dump(dict, f, ensure_ascii=False)


def usual_train(config):
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


def train(config,
          test_one_iter=False):
    if test_one_iter:
        test_train(config)
    else:
        usual_train(config)
