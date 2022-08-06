import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y

    def __len__(self):
        return len(self.x)


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    losses = list()
    model.train()
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        y1 = model(x)
        loss = loss_fn(y1, y)
        losses.append(loss.item())

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"average loss: {np.mean(losses)}")


def print_parameters(model):
    for p in model.parameters():
        print(p)


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
        # print_parameters(model)
    print("Finished training")


def build_dataset(x, y, indices):
    dataset = SimpleDataset(x, y)
    return torch.utils.data.Subset(dataset, indices)


def build_dataloader(x, y, indices, batch_size):
    dataset = build_dataset(x, y, indices)
    return DataLoader(dataset, batch_size=batch_size)


def test(model, dataset, loss_fn, device, verbose):
    num_samples = len(dataset)
    y_size = dataset[0][1].shape[0]
    y_array = torch.zeros((num_samples, y_size), dtype=torch.float32)
    y1_array = torch.zeros((num_samples, y_size), dtype=torch.float32)
    losses = np.zeros(num_samples, dtype=float)
    with torch.set_grad_enabled(False):
        model.eval()
        n_correct = 0
        for i in range(len(dataset)):
            x, y = dataset[i]
            x, y = x.to(device), y.to(device)
            y1 = model(x.unsqueeze(0))[0]
            if torch.argmax(y1) == torch.argmax(y):
                n_correct += 1
            if verbose > 1:
                print(f"y: {y} {y1}")
            y_array[i] = y
            y1_array[i] = y1
            losses[i] = loss_fn(y1, y).item()

        print(f"Test average loss: {np.mean(losses)} accuracy: {n_correct / len(dataset)}")
        return y_array.numpy(), y1_array.numpy()