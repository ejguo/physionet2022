import torch
from torch import nn
from torch.utils.data import DataLoader
from train_helper import SimpleDataset, train
import torch.nn.functional as F
from preprocess import get_mfcc3


"""

    Make a cnn
    X -> (3141, 3, 500), mel spectrogram with 3 frequency bins and 300 time banks
    Y-> (3141, 2), 1402 samples either murmur present (1, 0) or murmur absent (0, 1)


"""


class MurmurModel1(nn.Module):
    def __init__(self):
        super(MurmurModel1, self).__init__()
        self.conv1 = nn.Conv1d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool1d(20, stride=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4, stride=3, padding=1)

        self.linear1 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.linear2 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, 3)
        self.bn4 = nn.BatchNorm1d(3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print(x.shape)
        x = F.leaky_relu(self.conv1(x))
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = F.leaky_relu(self.conv2(x))
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.bn3(self.linear2(x)))
        x = F.leaky_relu(self.linear3(x))
        x = self.softmax(x)
        return x


class MurmurModel2(nn.Module):
    def __init__(self):
        super(MurmurModel2, self).__init__()
        self.conv1 = nn.Conv1d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool1d(20, stride=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4, stride=3, padding=1)

        self.linear1 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.linear2 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, 2)
        self.bn4 = nn.BatchNorm1d(2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print(x.shape)
        x = F.leaky_relu(self.conv1(x))
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = F.leaky_relu(self.conv2(x))
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.bn3(self.linear2(x)))
        x = F.leaky_relu(self.linear3(x))
        x = self.softmax(x)
        return x

class MurmurModel3(nn.Module):
    def __init__(self):
        super(MurmurModel3, self).__init__()
        self.conv1 = nn.Conv1d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool1d(20, stride=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4, stride=3, padding=1)

        self.linear1 = nn.Linear(2048, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.linear3 = nn.Linear(32, 3)
        self.bn4 = nn.BatchNorm1d(3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print(x.shape)
        x = F.leaky_relu(self.conv1(x))
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = F.leaky_relu(self.conv2(x))
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.bn3(self.linear2(x)))
        x = F.leaky_relu(self.linear3(x))
        x = self.softmax(x)
        return x


class MurmurModel(nn.Module):
    def __init__(self):
        super(MurmurModel, self).__init__()
        self.conv1 = nn.Conv1d(3, 32, 19, padding=9)
        self.pool1 = nn.MaxPool1d(20, stride=5)
        self.conv2 = nn.Conv1d(32, 64, 9, padding=4)
        self.pool2 = nn.MaxPool1d(4, stride=2)

        self.linear1 = nn.Linear(3008, 1024)
        # self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(1024, 512)
        # self.bn2 = nn.BatchNorm1d(2)
        self.linear3 = nn.Linear(512, 2)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return x


if __name__ == "__main__":
    data_dir = 'D:\\git\\challenge2022\\the-circor-digiscope-phonocardiogram-dataset-1.0.3\\training_data'
    verbose = 1
    batch_size = 32
    EPOCHS = 100
    LEARNING_RATE = 0.00001

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    n_fft = 1024
    hop_length = 512   # max_len = 152080, try to get 300 time banks, hop_length = 507 or 508
    n_mels = 3
    x, y = get_mfcc3(data_dir, n_fft, hop_length, n_mels, verbose)
    # for i in range(x.size(dim=0)):
    #     plot_spectrogram(x[i], 4000)

    model = MurmurModel1().to(device)

    train_data = SimpleDataset(x, y)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # weights = [0.7, 0.1, 0.3, 0.5, 0.5]
    weights = [0.5, 0.8, 0.1]
    weights = torch.tensor(weights, dtype=torch.float)
    loss_fn = nn.CrossEntropyLoss(weights).to(device)
    # loss_fn = nn.CrossEntropyLoss().to(device)
    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(model, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    filename = "murmur3.model"
    torch.save(model.state_dict(), filename)
    print(f"Trained feed forward net saved at {filename}")