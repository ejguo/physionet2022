import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



def train_single_epoch(model, dataloader, loss_fn, optimiser, device):
    _, y = next(iter(dataloader))
    y_size = y.shape[1]
    confusion = np.zeros((y_size, y_size), dtype=int)
    losses = list()
    model.train()
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
        optimiser.step()
        # print(f"min {torch.min(model.conv1.weight.grad)} {torch.min(model.conv2.weight.grad)} {torch.min(model.conv3.weight.grad)} {torch.min(model.conv4.weight.grad)}")
        # print(f"max {torch.max(model.conv1.weight.grad)} {torch.max(model.conv2.weight.grad)} {torch.max(model.conv3.weight.grad)} {torch.max(model.conv4.weight.grad)}")

    print(f"loss = {np.mean(losses)} accuracy = {n_correct / n_total}")
    print(confusion)


def print_parameters(model):
    for p in model.parameters():
        print(p)


def train(model, dataloader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}", end=' ')
        train_single_epoch(model, dataloader, loss_fn, optimiser, device)
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
    raise

def gen_wave_ids(ids):
    """convert ids[i,j] into a list o.f ids, each for a wave"""
    n = -1
    id_list = []
    for patient_id in ids:
        for n in range(n + 1, patient_id[1]):
            id_list.append(patient_id[0])
    return id_list


def check_dataloader(dataloader, ids):
    id_list = gen_wave_ids(ids)

    for _, batch_y in dataloader:
        for y in batch_y:
            print(f"{id_list.pop(0)} {y}")

    assert not id_list, "ids contains more waves than dataloader"

