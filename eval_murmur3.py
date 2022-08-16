"""

    eval murmur: very similar to train_murmur.py

"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from train_helper import SimpleDataset
from train_murmur3 import MurmurModel1
from preprocess import get_mfcc3

if __name__ == "__main__":
    data_dir = 'D:\\git\\challenge2022\\the-circor-digiscope-phonocardiogram-dataset-1.0.3\\training_data'
    verbose = 1
    device = "cpu"

    # load back the model
    model = MurmurModel1().to(device)
    filename = "murmur3.model"
    state_dict = torch.load(filename)
    model.load_state_dict(state_dict)

    n_fft = 1024
    hop_length = 512   # max_len = 152080, try to get 300 time banks, hop_length = 507 or 508
    n_mels = 3
    x, y = get_mfcc3(data_dir, n_fft, hop_length, n_mels, verbose)

    dataset = SimpleDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    count = torch.count_nonzero(dataset.y, dim=0)
    print(f"count murmurs {count}")

    # get a sample from the dataset for inference
    test_indices = np.random.permutation(len(dataset))
    n = 100
    correct = 0
    for i in test_indices[:n]:
        x, y = dataset[i]  # [batch size, num_channels, fr, time]
        model.eval()
        with torch.no_grad():
            y1 = model(x.unsqueeze_(0))
            murmur = np.argmax(y1[0])
            murmur_true = np.argmax(y)
            message = f"{y1[0]} \t {y} \t "
            if murmur == murmur_true:
                message += f"{murmur}"
                correct += 1
            else:
                message += f"{murmur} <> {murmur_true}"
            print(message)

    print(f"Accuracy = {correct/n*100} %")
