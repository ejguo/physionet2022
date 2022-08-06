import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary


# CNN_cc_lll:
#   (B, in_features)   conv -> leaky_relu -> pool(20) ->
#   (B, 32, in/20)   conv -> leaky_relu -> pool(4) ->
#   (B, 64, in/80 )   flatten ->
#   (B, 2048)    linear -> leaky_relu ->
#   (B, 1024)    linear -> bn -> leaky_relu ->
#   (B, 512)     linear -> leaky_relu
#   (B, out_features)


class SimpleCNN(nn.Module):
    def __init__(self, in_features, out_features):
        super(SimpleCNN, self).__init__()
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
        # print(x.shape)
        x = F.leaky_relu(self.bn3(self.linear2(x)))
        # print(x.shape)
        x = F.leaky_relu(self.linear3(x))
        # print(x.shape)
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


if __name__ == "__main__":
    cnn = BasicCNN(1, 5, 0.2)
    summary(cnn.cuda(), (1, 64, 161))
