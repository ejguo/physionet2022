import os
import sys

import numpy as np
import torch
import configparser

from evaluation_model import print_evaluation, print_patient_evaluation
from models import build_model
from preprocess import load_data
from train_helper import train, test

# only use list of 2 elements
class LinearLoss(torch.nn.Module):
    def __init__(self, functions, weights):
        super(LinearLoss, self).__init__()
        self.functions = functions
        self.weights = weights
        self.loss_0 = None
        self.loss_1 = None

    def forward(self, x, y):
        self.loss_0 = self.functions[0](x[0], y[0])
        self.loss_1 = self.functions[1](x[1], y[1])
        # print(f"loss_0 = {self.loss_0.item()}  loss_1 = {self.loss_1.item()}")
        return self.loss_0 * self.weights[0] + self.loss_1 * self.weights[1]

class Dac:
    """
        Denoising Autoencoder Classification
    """

    def __init__(self, name, model, data_loader, class_loss_fn, config, device):
        print(f"Running denoising autoencoder classification for {name}:")
        self.name = name
        self.model = model
        self.data_loader = data_loader
        self.class_loss = class_loss_fn
        self.config = config
        self.output_dir = config.get('output_dir')
        self.noise_factor = config.getfloat('noise_factor')
        noise_type = config.get('noise_type')
        if noise_type == 'rand':
            self.add_noise = self.rand_noise
        elif noise_type == 'blackout':
            self.add_noise = self.blackout_noise
        else:
            raise(f"noise_type {noise_type} is not recognized")
        self.device = device
        print(f"noise_type={noise_type}  noise_factor={self.noise_factor}")

        self.dae_loss = torch.nn.MSELoss()

    def rand_noise(self, x):
        x_noise = x + self.noise_factor * torch.randn_like(x) * x
        z = x_noise - x
        return x_noise

    def blackout_noise(self, x):
        y = x * (torch.rand(x.shape) > self.noise_factor)
        return y

    def train_denoising_autoencoder(self):
        """ train x_train_noisy --> [ y_zeros, x_train ] """
        print("train_denoising_autoencoder:")
        path = os.path.join(self.output_dir, f"{self.name}_dae_model")
        if self.config.getboolean('resume_training') and os.path.exists(path):
            print("resume training")
            state = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state)

        lr = self.config.getfloat(self.name + '_dae_lr')
        optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)
        print(f"Adam optimiser, lr={lr}")
        epochs = self.config.getint(self.name + '_dae_epochs')
        print(f"{epochs} epochs")
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for x, _ in self.data_loader:
                x_noise = self.add_noise(x)
                x = x.to(self.device)
                x_noise = x_noise.to(self.device)
                optimiser.zero_grad()
                x_decode = self.model(x_noise)[1]
                loss = self.dae_loss(x, x_decode)
                loss.backward()
                optimiser.step()
                running_loss += loss.item()

            running_loss /= len(self.data_loader)
            print('Epoch: {}/{} \tLoss: {}'.format(epoch + 1, epochs, running_loss))
        torch.save(self.model.state_dict(), path)

    def train_regularized_model(self):
        """ assume self.model has been trained by train_denoising_autoencoder() """
        print("train_regularized_model:")
        lr = self.config.getfloat(self.name + '_reg_lr')
        optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)
        print(f"Adam optimiser, lr = {lr}")
        epochs = self.config.getint(self.name + '_reg_epochs')
        print(f"{epochs} epochs")
        loss_weights = np.array(self.config.get(self.name + '_dac_loss_weights').split(','), dtype=float)
        print(f"{self.name}_dac_loss_weights = {loss_weights}")
        loss_fn = LinearLoss((self.class_loss, self.dae_loss), loss_weights).to(self.device)
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            class_loss = 0.0
            for x, y in self.data_loader:
                x_noise = self.add_noise(x)
                x = x.to(self.device)
                y = y.to(self.device)
                x_noise = x_noise.to(self.device)
                optimiser.zero_grad()
                pred = self.model(x_noise)
                loss = loss_fn((y, x), pred)
                loss.backward()
                optimiser.step()
                running_loss += loss.item()
                class_loss += loss_fn.loss_0.item()

            running_loss /= len(self.data_loader)
            class_loss /= len(self.data_loader)
            print('Epoch: {}/{} \tLoss: {}\tClass Loss: {}'.format(epoch + 1, epochs, running_loss, class_loss))


    def test(self, dataset):
        print("test:")
        verbose = self.config.getint('verbose')
        num_samples = len(dataset)
        y_size = dataset[0][1].shape[0]
        y_array = torch.zeros((num_samples, y_size), dtype=torch.float32)
        y1_array = torch.zeros((num_samples, y_size), dtype=torch.float32)
        losses = np.zeros(num_samples, dtype=float)
        confusion = np.zeros((y_size, y_size), dtype=int)
        with torch.set_grad_enabled(False):
            self.model.eval()
            n_correct = 0
            for i in range(len(dataset)):
                x, y = dataset[i]
                x, y = x.to(self.device), y.to(self.device)
                y1 = self.model(x.unsqueeze(0))[0][0]
                index1 = torch.argmax(y1)
                index = torch.argmax(y)
                confusion[index1, index] += 1
                if index1 == index:
                    n_correct += 1
                if verbose > 1:
                    if index1 == index:
                        print(f"{y.cpu().numpy()} == {y1.cpu().numpy()}")
                    else:
                        print(f"{y.cpu().numpy()} <> {y1.cpu().numpy()}")
                y_array[i] = y
                y1_array[i] = y1
                losses[i] = self.class_loss(y1, y).item()

            print(f"Test loss: {np.mean(losses)} accuracy: {n_correct / len(dataset)}")
            print(f"Confusion matrix:")
            print(confusion)

            return y_array.numpy(), y1_array.numpy()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    train_config = config['train']
    verbose = train_config.getint('verbose')
    if verbose > 0:
        print(f"Using {device}")

    train_murmur_dataloader, train_outcome_dataloader, test_murmur_dataset, test_outcome_dataset, test_ids = load_data(config['preprocess'])
    murmur_model, outcome_model, murmur_loss_fn, outcome_loss_fn = build_model(train_config, device)

    output_dir = train_config.get('output_dir')

    murmur_dac = Dac('murmur', murmur_model, train_murmur_dataloader, murmur_loss_fn, train_config, device)
    murmur_dac.train_denoising_autoencoder()
    murmur_dac.train_regularized_model()
    murmur_labels, murmur_probs = murmur_dac.test(test_murmur_dataset)

    outcome_dac = Dac('outcome', outcome_model, train_outcome_dataloader, outcome_loss_fn, train_config, device)
    outcome_dac.train_denoising_autoencoder()
    outcome_dac.train_regularized_model()
    outcome_labels, outcome_probs = outcome_dac.test(test_outcome_dataset)

    print_evaluation(murmur_labels, murmur_probs, outcome_labels, outcome_probs)

    data_dir = train_config.get('data_dir')
    print_patient_evaluation(data_dir, test_ids, murmur_probs, outcome_probs, verbose)
