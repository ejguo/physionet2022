import os
import sys

import numpy as np
import torch
import configparser

from torch import nn

from evaluation_model import print_evaluation, print_patient_evaluation
from models import Basic_model, Dac_model
from preprocess import load_data
from train_helper import train, test


def train_basic(name, config, dataloader, device, verbose=1):
    print(f"training basic model: {name}, verbose={verbose}")
    config = config[name]
    # create model
    model = Basic_model(config).to(device)
    print(model)
    # create loss_fn
    loss_weights = np.array(config.get('loss_weights').split(','), dtype=float)
    print(f"loss_weights={loss_weights} CrossEntropyLoss Adam")
    loss_fn = nn.CrossEntropyLoss(torch.tensor(loss_weights)).to(device)

    # create optimiser
    lr = config.getfloat('lr')
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Adam optimiser, lr={lr}")

    # get epochs
    epochs = config.getint('epochs')
    print(f"{epochs} epochs")

    # create confusion matrix
    _, y = next(iter(dataloader))
    y_size = y.shape[1]
    confusion = np.zeros((y_size, y_size), dtype=int)

    # train
    model.train()
    for i in range(epochs):
        losses = list()
        n_total = 0
        n_correct = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y1 = model(x)
            index1 = torch.argmax(y1, dim=1)
            index = torch.argmax(y, dim=1)
            for idx1, idx in torch.stack((index1, index), dim=1):
                confusion[idx1, idx] += 1
            n_total += y.size(0)
            n_correct += torch.sum(index1 == index)
            loss = loss_fn(y1, y)
            losses.append(loss.item())

            optimiser.zero_grad()
            loss.backward()
            # print(f"min {torch.min(model.conv1.weight.grad)} {torch.min(model.conv2.weight.grad)} {torch.min(model.conv3.weight.grad)} {torch.min(model.conv4.weight.grad)}")
            # print(f"max {torch.max(model.conv1.weight.grad)} {torch.max(model.conv2.weight.grad)} {torch.max(model.conv3.weight.grad)} {torch.max(model.conv4.weight.grad)}")
            optimiser.step()

        # print(f"len(dataloader) = {len(dataloader)} =?= n_total = {n_total}")
        print(f"loss = {np.mean(losses)} accuracy = {n_correct / n_total}")
        print(confusion)

    return model, loss_fn

def test(model, dataset, loss_fn, device, verbose):
    num_samples = len(dataset)
    y_size = dataset[0][1].shape[0]
    y_array = torch.zeros((num_samples, y_size), dtype=torch.float32)
    y1_array = torch.zeros((num_samples, y_size), dtype=torch.float32)
    losses = np.zeros(num_samples, dtype=float)
    confusion = np.zeros((y_size, y_size), dtype=int)
    with torch.set_grad_enabled(False):
        model.eval()
        n_correct = 0
        for i in range(len(dataset)):
            x, y = dataset[i]
            x, y = x.to(device), y.to(device)
            y1 = model(x.unsqueeze(0))[0]
            index1 = torch.argmax(y1)
            index = torch.argmax(y)
            confusion[index1, index] += 1
            if index1 == index:
                n_correct += 1
            elif verbose > 1:
                print(f"{y.cpu().numpy()} <> {y1.cpu().numpy()}")
            y_array[i] = y
            y1_array[i] = y1
            losses[i] = loss_fn(y1, y).item()

        print(f"Test loss: {np.mean(losses)} accuracy: {n_correct / len(dataset)}")
        print(f"Confusion matrix:")
        print(confusion)

        return y_array.numpy(), y1_array.numpy()


def test_basic(name, config, train_dataloader, test_dataset, device, verbose):
    model, loss_fn = train_basic(name, config, train_dataloader, device, verbose)
    torch.save(model.state_dict(), os.path.join(output_dir, f"{name}_model"))
    return test(model, test_dataset, loss_fn, device, verbose)
    
# This will be the final version of main.py
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = configparser.ConfigParser()
    config_path = sys.argv[1]
    config.read(config_path)
    output_dir = os.path.dirname(config_path)

    verbose = config['DEFAULT'].getint('verbose')
    if verbose > 0:
        print(f"Using {device}")

    train_murmur_dataloader, train_outcome_dataloader, test_murmur_dataset, test_outcome_dataset, test_ids = load_data(config['preprocess'])

    murmur_labels, murmur_probs = test_basic('murmur_basic', config, train_murmur_dataloader, test_murmur_dataset, device, verbose)
    outcome_labels, outcome_probs = test_basic('outcome_basic', config, train_outcome_dataloader, test_outcome_dataset, device, verbose)

    print_evaluation(murmur_labels, murmur_probs, outcome_labels, outcome_probs)

    data_dir = config['DEFAULT'].get('data_dir')
    print_patient_evaluation(data_dir, test_ids, murmur_probs, outcome_probs, verbose)
