import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
from preprocess import murmur_classes, outcome_classes

# This model is unfinished
class DacModel(nn.Module):
    def __init__(self, out_features, dropout_rate):
        super(DacModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 2),   nn.BatchNorm2d(16),  F.relu, nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 2),  nn.BatchNorm2d(32),  F.relu, nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 2),  nn.BatchNorm2d(64),  F.relu, nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 2), nn.BatchNorm2d(128), F.relu, nn.MaxPool2d(2),
            nn.Dropout(dropout_rate),
            nn.AvgPool2d((1, 11)),
            nn.Flatten
        )

        self.classify = nn.Sequential(nn.Linear(128, out_features), nn.Softmax(dim=1))

        self.decoder = nn.Sequential()

    def forward(self, x):
        x = self.encoder(x)
        output = self.classify(x)
        decode = self.decoder(x)
        return output, decode

# essentially copy from Eugenia Anello's code
class EncDecClass(nn.Module):
    def __init__(self, out_features, encoded_space_dim):
        super(EncDecClass, self).__init__()

        self.encoder_cnn = nn.Sequential(
            ### Convolutional section
            nn.Conv2d(1, 8, 3, stride=2, padding=(1,0)),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
        )

        self.encoder_lin = nn.Sequential(
            ### Flatten layer
            nn.Flatten(start_dim=1),
            ### Linear section
            nn.Linear(32 * 5 * 25, 128),    # 3 * 3 need change
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )

        self.classify = nn.Sequential(
            nn.Linear(encoded_space_dim, out_features),
            nn.Softmax(dim=1)
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 32 * 5 * 25),    # 3 * 3 need change
            nn.ReLU(True),
            nn.Unflatten(dim=1, unflattened_size=(32, 5, 25)),   # change 32, 3, 3
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=(1,0), output_padding=(1,0)),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(x.shape)  # torch.Size([64, 1, 40, 201])
        x = self.encoder_cnn(x)
        # print(x.shape)  # torch.Size([64, 32, 5, 26])
        x = self.encoder_lin(x)
        output = self.classify(x)
        decode = self.decoder(x)
        return output, decode


class C4F1(nn.Module):
    def __init__(self, out_features, dropout_rate):
        super(C4F1, self).__init__()
        self.conv0 = nn.Conv2d(1, 4, (1, 1))
        self.bn0 = nn.BatchNorm2d(4)
        self.conv1 = nn.Conv2d(4, 4, (1, 3))
        self.bn1 = nn.BatchNorm2d(4)
        self.pool1 = nn.MaxPool2d((1, 2))
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(4, 8, (1, 3))
        self.bn2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d((1, 2))
        self.conv3 = nn.Conv2d(8, 16, (1, 3))
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d((1, 2))
        self.conv4 = nn.Conv2d(16, 32, (1, 3))
        self.bn4 = nn.BatchNorm2d(32)
        self.pool4 = nn.MaxPool2d((1, 2))
        self.pool = nn.AvgPool2d((1, 17))  # (1, 24) for 4 secs x 40 n_mfcc
        self.fc = nn.Linear(1792, out_features)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = self.dropout(self.pool1(F.relu(self.conv1(x))))
        x = self.dropout(self.pool2(F.relu(self.conv2(x))))
        x = self.dropout(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        x = self.dropout(self.pool4(F.relu(self.bn4(self.conv4(x)))))
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        x = self.softmax(x)
        return x

class C4F1_1(nn.Module):
    def __init__(self, out_features, dropout_rate):
        super(C4F1_1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(16, 32, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 2)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(64, 128, 2)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2)
        # self.pool = nn.AvgPool2d(1, 17)   # seems error, 20221013
        self.pool = nn.AvgPool2d((1, 11))   # for 3 seconds wave, i.e., 12000 points, mfcc 201, dim (64, 128, 1, 11) now
        self.fc = nn.Linear(128, out_features)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dropout(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.dropout(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        x = self.dropout(self.pool4(F.relu(self.bn4(self.conv4(x)))))
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        x = self.softmax(x)
        return x

# no dropout before last one
class C4F1_2(nn.Module):
    def __init__(self, out_features, dropout_rate):
        super(C4F1_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(16, 32, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 2)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(64, 128, 2)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2)
        # self.pool = nn.AvgPool2d(1, 17)   # seems error, 20221013
        self.pool = nn.AvgPool2d((1, 11))   # for 3 seconds wave, i.e., 12000 points, mfcc 201, dim (64, 128, 1, 11) now
        self.fc = nn.Linear(128, out_features)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(self.pool4(F.relu(self.bn4(self.conv4(x)))))
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class C4F1_raw(nn.Module):
    def __init__(self, out_features):
        super(C4F1_raw, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 2)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(16, 32, 2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(32, 64, 2)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)
        self.conv4 = nn.Conv1d(64, 128, 2)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(2)
        self.pool = nn.AvgPool1d(1, 17)
        self.fc = nn.Linear(128, out_features)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dropout(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.dropout(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        x = self.dropout(self.pool4(F.relu(self.bn4(self.conv4(x)))))
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class C2F2(nn.Module):
    def __init__(self, n_mfcc, out_features):
        super(C2F2, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (2, 20), padding='same')
        self.pool1 = nn.MaxPool2d((1, 20), stride=(1, 5), padding=(0, 8))
        self.conv2 = nn.Conv2d(64, 64, (2, 10), padding='same')
        self.pool2 = nn.MaxPool2d((1, 4), stride=(1, 2), padding=(0, 1))
        self.fc1 = nn.Linear(64 * n_mfcc * 30, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, out_features)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print(x.shape)
        x = F.leaky_relu((self.conv1(x)))
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = F.relu(self.fc2(x))
        # print(x.shape)
        x = F.relu(self.fc3(x))
        # print(x.shape)
        x = self.softmax(x)
        return x


class FcNN(nn.Module):
    def __init__(self, in_length, out_features):
        super(FcNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 4, 4, stride=2)
        self.pool1 = nn.MaxPool1d(4, stride=4)
        out = in_length // 10
        self.fc1 = nn.Linear(5996, out)
        in_features = out
        out = in_features // 10
        self.fc2 = nn.Linear(in_features, out)
        self.fc3 = nn.Linear(out, out_features)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = F.relu(self.fc2(x))
        # print(x.shape)
        x = F.relu(self.fc3(x))
        # print(x.shape)
        x = self.softmax(x)
        return x


# CNN_cc_lll:
#   (B, in_features)   conv -> leaky_relu -> pool(20) ->
#   (B, 32, in/20)   conv -> leaky_relu -> pool(4) ->
#   (B, 64, in/80 )   flatten ->
#   (B, 2048)    linear -> leaky_relu ->
#   (B, 1024)    linear -> bn -> leaky_relu ->
#   (B, 512)     linear -> leaky_relu
#   (B, out_features)
class C2F3(nn.Module):
    def __init__(self, in_features, out_features):
        super(C2F3, self).__init__()
        self.conv1 = nn.Conv1d(in_features, 32, 3, padding=1)  # L = num_samples
        self.pool1 = nn.MaxPool1d(20, stride=5, padding=2)  # L = L // 20
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4, stride=3, padding=1)  # L = L // 4

        self.linear1 = nn.Linear(2048, 1024)  # L * 64
        self.bn2 = nn.BatchNorm1d(1024)
        self.linear2 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, out_features)
        self.bn4 = nn.BatchNorm1d(out_features)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        print(x.shape)
        x = F.leaky_relu(self.conv1(x))
        print(x.shape)
        x = self.pool1(x)
        print(x.shape)
        x = F.leaky_relu(self.conv2(x))
        print(x.shape)
        x = self.pool2(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = F.leaky_relu(self.linear1(x))
        print(x.shape)
        x = F.leaky_relu(self.bn3(self.linear2(x)))
        print(x.shape)
        x = F.leaky_relu(self.linear3(x))
        print(x.shape)
        x = self.softmax(x)
        return x


class WaveCNN(nn.Module):
    def __init__(self, in_features, out_features):
        super(WaveCNN, self).__init__()
        self.conv0 = nn.Conv1d(in_features, 4, 1)
        self.conv1 = nn.Conv1d(4, 32, 3, padding=1)  # L = num_samples
        self.pool1 = nn.MaxPool1d(20, stride=5, padding=2)  # L = L // 20
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4, stride=3, padding=1)  # L = L // 4

        self.linear1 = nn.Linear(68224, 1024)  # L * 64
        self.bn2 = nn.BatchNorm1d(1024)
        self.linear2 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, out_features)
        self.bn4 = nn.BatchNorm1d(out_features)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        print(x.shape)
        x = F.leaky_relu(self.conv0(x))
        print(x.shape)
        x = F.leaky_relu(self.conv1(x))
        print(x.shape)
        x = self.pool1(x)
        print(x.shape)
        x = F.leaky_relu(self.conv2(x))
        print(x.shape)
        x = self.pool2(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = F.leaky_relu(self.linear1(x))
        print(x.shape)
        x = F.leaky_relu(self.bn3(self.linear2(x)))
        print(x.shape)
        x = F.leaky_relu(self.linear3(x))
        print(x.shape)
        x = self.softmax(x)
        return x


class MurmurModel1(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate):
        super(MurmurModel1, self).__init__()
        self.conv1 = nn.Conv1d(in_features, 32, 3, padding=1)
        self.pool1 = nn.MaxPool1d(20, stride=5, padding=2)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4, stride=3, padding=1)

        self.linear1 = nn.Linear(1536, 1024)  # 2048
        self.bn2 = nn.BatchNorm1d(1024)
        self.linear2 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, out_features)
        self.bn4 = nn.BatchNorm1d(out_features)
        self.softmax = nn.Softmax(dim=1)
        self.linear4 = nn.Linear(1536, out_features)

    def forward(self, x):
        # print(x.shape)
        x = F.leaky_relu(self.conv1(x))
        # print(x.shape)
        x = self.pool1(x)
        x = self.dropout(x)
        # print(x.shape)
        x = F.leaky_relu(self.bn1(self.conv2(x)))
        # print(x.shape)
        x = self.pool2(x)
        x = self.dropout(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        # x = F.leaky_relu(self.linear1(x))
        # x = self.dropout(x)
        # x = F.leaky_relu(self.bn3(self.linear2(x)))
        # x = F.leaky_relu(self.linear3(x))
        x = F.leaky_relu(self.linear4(x))
        x = self.softmax(x)
        return x


class FCLayer(nn.Module):
    def __init__(self):
        super(FCLayer, self).__init__()
        self.linear1 = nn.Linear(64 * 321, 1000)
        self.bn1 = nn.BatchNorm1d(1000)
        # self.pool = nn.MaxPool1d(3, stride=2)
        self.linear2 = nn.Linear(1000, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.linear3 = nn.Linear(100, 10)
        self.bn3 = nn.BatchNorm1d(10)
        self.linear4 = nn.Linear(10, 5)
        self.bn4 = nn.BatchNorm1d(5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print(x.shape)
        x = x.contiguous()
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)

        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = F.relu(self.bn3(self.linear3(x)))
        x = F.relu(self.bn4(self.linear4(x)))
        x = torch.cat((self.softmax(x[:, :3]), self.softmax(x[:, 3:])), 1)
        return x


class BasicCNNLayer(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, stride=1):
        super(BasicCNNLayer, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, stride)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        layer_output = F.relu(self.bn(self.conv(x)))
        return layer_output


class BasicCNN(nn.Module):
    def __init__(self, num_channels, num_classes, dropout_rate, widen_factor=2,
                 kernel_size=2, verbose=0):
        super(BasicCNN, self).__init__()
        self.verbose = verbose
        in_features = num_channels
        out_features = in_features * widen_factor
        self.layer1 = BasicCNNLayer(in_features, out_features, kernel_size=5)

        self.pool = nn.MaxPool2d(kernel_size, stride=2)

        in_features = out_features
        out_features = in_features * widen_factor
        self.layer2 = BasicCNNLayer(in_features, out_features, kernel_size=3, stride=2)

        in_features = out_features
        out_features = in_features * widen_factor
        self.layer3 = BasicCNNLayer(in_features, out_features, kernel_size=3, stride=2)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(48, num_classes)

        self.softmax = nn.Softmax(dim=1)

    def print_shape(self, x):
        if self.verbose > 2:
            print(x.shape)

    def forward(self, x):
        self.print_shape(x)
        x = self.layer1(x)
        self.print_shape(x)
        x = self.pool(x)
        self.print_shape(x)
        x = self.layer2(x)
        self.print_shape(x)
        x = self.layer3(x)
        self.print_shape(x)
        x = F.avg_pool2d(x, x.size(2))
        self.print_shape(x)
        x = self.dropout(x)
        self.print_shape(x)
        x = x.view(x.size(0), -1)
        self.print_shape(x)
        x = self.fc(x)
        self.print_shape(x)
        x = torch.cat((self.softmax(x[:, :3]), self.softmax(x[:, 3:])), 1)
        self.print_shape(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self, in_features, out_features, kernel_size):
        super(BasicLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size, padding=3)
        self.bn = nn.BatchNorm2d(out_features)
        self.conv2 = nn.Conv2d(out_features, out_features, kernel_size, padding=3)

    def forward(self, x):
        x = self.bn(F.relu(self.conv1(x)))
        x = self.bn(F.relu(self.conv2(x)))
        return x


class BasicLayerPool(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, pool_size):
        super(BasicLayerPool, self).__init__()
        self.layer = BasicLayer(in_features, out_features, kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)

    def forward(self, x):
        x = self.layer(x)
        return self.pool(x)


class DeepSleep(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate, verbose):
        super(DeepSleep, self).__init__()
        self.verbose = verbose
        features = [in_features, 10, 15, 20, 25, 30, 60, 120, 240, 480]
        # pool_sizes = [2, 2, 2, 2, 2, 2, 2, 2]
        self.layers = nn.ModuleList()
        # n = len(features)
        # for i in range(n - 2):
        #     self.layers.append(BasicLayerPool(features[i], features[i + 1], 7, pool_sizes[i]))
        self.layers.append(BasicLayer(features[-2], features[-1], 7))

        self.dropout = nn.Dropout(p=dropout_rate)
        self.out_features = out_features
        self.fc = nn.Linear(17018860, out_features)

        self.softmax = nn.Softmax(dim=1)

    def print(self, x):
        if self.verbose > 2:
            print(x)

    def forward(self, x):
        self.print(x.shape)
        z = [x]
        for i in range(len(self.layers)):
            z.append(self.layers[i](z[i]))
            self.print(z[-1].shape)
        x = z.pop()
        z.pop(0)
        x = self.dropout(x)
        self.print(x.shape)
        x = x.view(x.size(0), -1)
        self.print(x.shape)
        x = self.fc(x)
        self.print(x.shape)
        x = torch.cat((self.softmax(x[:, :3]), self.softmax(x[:, 3:])), 1)
        self.print(x.shape)
        return x


def build_model(config, device):
    num_murmurs = len(murmur_classes)
    num_outcomes = len(outcome_classes)
    epochs = config.getint('epochs')
    learning_rate = config.getfloat('learning_rate')
    dropout_rate = config.getfloat('dropout_rate')
    model_type = config.get('model_type')
    print(f"epochs={epochs}  learning_rate={learning_rate}")
    murmur_model = None
    outcome_model = None
    if model_type == 'MurmurModel1':
        n_mels = config.getint('n_mels')
        murmur_model = MurmurModel1(n_mels, num_murmurs, dropout_rate)
        outcome_model = MurmurModel1(n_mels, num_outcomes, dropout_rate)
    elif model_type == 'C2F3':
        n_mels = config.getint('n_mels')
        murmur_model = C2F3(n_mels, num_murmurs)
        outcome_model = C2F3(n_mels, num_outcomes)
    elif model_type == 'WaveCNN':
        murmur_model = WaveCNN(1, num_murmurs)
        outcome_model = WaveCNN(1, num_outcomes)
    elif model_type == 'C2F2':
        n_mfcc = config.getint('n_mfcc')
        murmur_model = C2F2(n_mfcc, num_murmurs)
        outcome_model = C2F2(n_mfcc, num_outcomes)
    elif model_type == 'FcNN':
        wave_len = 4000 * config.getint('wave_seconds')
        murmur_model = FcNN(wave_len, num_murmurs)
        outcome_model = FcNN(wave_len, num_outcomes)
    elif model_type == 'C4F1':
        murmur_model = C4F1(num_murmurs, dropout_rate)
        outcome_model = C4F1(num_outcomes, dropout_rate)
    elif model_type == 'C4F1_raw':
        murmur_model = C4F1_raw(num_murmurs)
        outcome_model = C4F1_raw(num_outcomes)
    elif model_type == 'C4F1_1':
        murmur_model = C4F1_1(num_murmurs, dropout_rate)
        outcome_model = C4F1_1(num_outcomes, dropout_rate)
    elif model_type == 'C4F1_2':
        murmur_model = C4F1_2(num_murmurs, dropout_rate)
        outcome_model = C4F1_2(num_outcomes, dropout_rate)
    elif model_type == 'EncDecClass':
        encoded_space_dim = config.getint('encoded_space_dim')
        murmur_model = EncDecClass(num_murmurs, encoded_space_dim)
        outcome_model = EncDecClass(num_outcomes, encoded_space_dim)

    murmur_model, outcome_model = murmur_model.to(device), outcome_model.to(device)
    print(f"Murmur model: {murmur_model}")
    print(f"Outcome model: {outcome_model}")

    loss_murmur_weights = np.array(config.get('murmur_loss_weights').split(','), dtype=float)
    print(f"loss_murmur_weights={loss_murmur_weights} CrossEntropyLoss Adam")
    murmur_loss_fn = nn.CrossEntropyLoss(torch.tensor(loss_murmur_weights)).to(device)
    murmur_optimiser = torch.optim.Adam(murmur_model.parameters(), lr=learning_rate)

    loss_outcome_weights = np.array(config.get('outcome_loss_weights').split(','), dtype=float)
    print(f"loss_outcome_weights={loss_outcome_weights}  CrossEntropyLoss Adam")
    # outcome_loss_fn = nn.BCELoss(torch.tensor(loss_outcome_weights)).to(device)
    outcome_loss_fn = nn.CrossEntropyLoss(torch.tensor(loss_outcome_weights)).to(device)
    outcome_optimiser = torch.optim.Adam(outcome_model.parameters(), lr=learning_rate)

    return murmur_model, outcome_model, murmur_loss_fn, outcome_loss_fn, murmur_optimiser, outcome_optimiser

if __name__ == "__main__":
    cnn = BasicCNN(1, 5, 0.2)
    summary(cnn.cuda(), (1, 64, 161))
