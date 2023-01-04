import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
from preprocess import murmur_classes, outcome_classes

def output_dim_padding(input_dim, kernel, stride, padding):
    """
    :param input_dim:
    :param kernel:  kernel_size
    :param stride:
    :param padding:
    :return: output_dim, output_padding
    """
    x = input_dim - kernel + 2 * padding
    return x // stride + 1, x % stride


def get_array_from_config(config, name, dtype, size=0):
    x = config.get(name)
    a = np.array(x.split(','), dtype=dtype)
    if size != 0 and a.size != size:
        raise Exception(f"{name} {x} should have {size} elements")
    return a


def get_conv2d_parameters(config):
    """ get 3 conv2d parameters from config file """
    num_feature = get_array_from_config(config, 'num_feature', int, 4)
    kernel_size = get_array_from_config(config, 'kernel_size', int, 6)
    stride = get_array_from_config(config, 'stride', int, 6)
    padding = get_array_from_config(config, 'padding', int, 6)
    kernel_size = np.reshape(kernel_size, (3, 2))
    stride = np.reshape(stride, (3, 2))
    padding = np.reshape(padding, (3, 2))
    return num_feature, kernel_size, stride, padding

def compute_output_dim_padding_2d(input_dim, kernel_size, stride, padding):
    """ config defines 3 conv2d layers and input dim (width, height)
        return (output_dim, output_padding) for all layers """
    n_layers = kernel_size.shape[0]
    n_dim = input_dim.size   # 2d
    output_dim = np.empty((n_layers+1, n_dim), dtype=int)
    output_dim[0] = input_dim
    output_padding = np.empty((n_layers, n_dim), dtype=int)
    for i in range(n_layers):
        output_dim[i+1], output_padding[i] = output_dim_padding(output_dim[i], kernel_size[i], stride[i], padding[i])
    return output_dim[1:], output_padding

class Classify(nn.Module):
    def __init__(self, in_features, mid_features, out_features):
        super(Classify, self).__init__()
        self.classify = nn.Sequential(
            nn.Linear(in_features, mid_features),
            nn.ReLU(True),
            nn.Linear(mid_features, out_features),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.classify(x)

class Encode(nn.Module):
    def __init__(self, config):
        super(Encode, self).__init__()
        num_feature, kernel_size, stride, padding = get_conv2d_parameters(config)
        self.encode = nn.Sequential(
            nn.Conv2d(num_feature[0], num_feature[1], kernel_size[0], stride[0], padding[0]),
            nn.BatchNorm2d(num_feature[1]),
            nn.ReLU(True),
            nn.Conv2d(num_feature[1], num_feature[2], kernel_size[1], stride[1], padding[1]),
            nn.BatchNorm2d(num_feature[2]),
            nn.ReLU(True),
            nn.Conv2d(num_feature[2], num_feature[3], kernel_size[2], stride[2], padding[2]),
            nn.BatchNorm2d(num_feature[3]),
            nn.ReLU(True),
            nn.Flatten(start_dim=1)
        )

    def forward(self, x):
        return self.encode(x)

class Decode(nn.Module):
    def __init__(self, config):
        super(Decode, self).__init__()
        num_feature, kernel_size, stride, padding = get_conv2d_parameters(config)
        input_dim = get_array_from_config(config, 'input_dim', int, 2)
        output_dim, output_padding = compute_output_dim_padding_2d(input_dim, kernel_size, stride, padding)
        self.decoder = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(num_feature[3], output_dim[0][2], output_dim[1][2])),
            nn.ConvTranspose2d(num_feature[3], num_feature[2], kernel_size[2], stride[2], padding[2], output_padding[2]),
            nn.BatchNorm2d(num_feature[2]),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_feature[2], num_feature[1], kernel_size[1], stride[1], padding[1], output_padding[1]),
            nn.BatchNorm2d(num_feature[1]),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_feature[1], num_feature[0], kernel_size[0], stride[0], padding[0], output_padding[0]),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)


class Basic_model(nn.Module):
    def __init__(self, config):
        super(Basic_model, self).__init__()
        num_feature, kernel_size, stride, padding = get_conv2d_parameters(config)
        input_dim = get_array_from_config(config, 'input_dim', int, 2)
        output_dim, _ = compute_output_dim_padding_2d(input_dim, kernel_size, stride, padding)
        latent_dim = num_feature[3] * output_dim[2][0] * output_dim[2][1]
        mid_dim = config.getint('classify_mid_dim')
        self.encoder = Encode(config)
        self.classify = Classify(latent_dim, mid_dim, config.getint('num_classes'))

    def forward(self, x):
        encode = self.encoder(x)
        output = self.classify(encode)
        return output

class Dac_model(Basic_model):
    def __init__(self, config):
        super(Dac_model, self).__init__(config)
        self.decoder = Decode(config)

    def forward(self, x):
        encode = self.encoder(x)
        output = self.classify(encode)
        decode = self.decoder(encode)
        return output, decode

class Encoder_3C0P1B(nn.Module):
    def __init__(self):
        super(Encoder_3C0P1B, self).__init__()
        self.encoder = nn.Sequential(
            ### Convolutional section
            nn.Conv2d(1, 8, 3, stride=2, padding=(1, 0)),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            ### Flatten layer
            nn.Flatten(start_dim=1)  # 32 * 5 * 25
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder_3C0P2B(nn.Module):
    def __init__(self):
        super(Decoder_3C0P2B, self).__init__()

        self.decoder = nn.Sequential(
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
        return self.decoder(x)

class Dac_3C0P(nn.Module):
    def __init__(self, out_features, latent_features):
        super(Dac_3C0P, self).__init__()

        self.encoder = Encoder_3C0P1B()
        self.classify = Classify(latent_features, 128, out_features)
        self.decoder = Decoder_3C0P2B()

    def forward(self, x):
        encode = self.encoder(x)
        output = self.classify(encode)
        decode = self.decoder(encode)
        return output, decode


class Encoder_3Conv2d(nn.Module):
    def __init__(self, latent_features):
        super(Encoder_3Conv2d, self).__init__()
        self.encoder = nn.Sequential(
            ### Convolutional section
            nn.Conv2d(1, 8, 3, stride=2, padding=(1, 0)),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            ### Flatten layer
            nn.Flatten(start_dim=1),
            ### Linear section
            nn.Linear(32 * 5 * 25, 128),  # 3 * 3 need change
            nn.ReLU(True),
            nn.Linear(128, latent_features)
        )

    def forward(self, x):
        return self.encoder(x)



class Decoder_3Conv2d(nn.Module):
    def __init__(self, latent_features):
        super(Decoder_3Conv2d, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_features, 128),
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
        return self.decoder(x)

class Dac_3Conv2d(nn.Module):
    def __init__(self, out_features, latent_features):
        super(Dac_3Conv2d, self).__init__()

        self.encoder = Encoder_3Conv2d(latent_features)
        self.classify = Classify(latent_features, 16, out_features)
        self.decoder = Decoder_3Conv2d(latent_features)

    def forward(self, x):
        encode = self.encoder(x)
        output = self.classify(encode)
        decode = self.decoder(encode)
        return output, decode


class EncDecClass(nn.Module):
    def __init__(self, out_features, latent_features):
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
            nn.Linear(128, latent_features)
        )

        self.classify = nn.Sequential(
            nn.Linear(latent_features, out_features),
            nn.Softmax(dim=1)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_features, 128),
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
    dropout_rate = config.getfloat('dropout_rate')
    model_type = config.get('model_type')
    print(f"epochs={epochs}")
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
        latent_features = config.getint('latent_features')
        murmur_model = EncDecClass(num_murmurs, latent_features)
        outcome_model = EncDecClass(num_outcomes, latent_features)
    elif model_type == 'Dac_3Conv2d':
        latent_features = config.getint('latent_features')
        murmur_model = Dac_3Conv2d(num_murmurs, latent_features)
        outcome_model = Dac_3Conv2d(num_outcomes, latent_features)
    elif model_type == 'Dac_3C0P':
        latent_features = 32 * 5 * 25
        murmur_model = Dac_3C0P(num_murmurs, latent_features)
        outcome_model = Dac_3C0P(num_outcomes, latent_features)

    murmur_model, outcome_model = murmur_model.to(device), outcome_model.to(device)
    print(f"Murmur model: {murmur_model}")
    print(f"Outcome model: {outcome_model}")

    loss_murmur_weights = np.array(config.get('murmur_loss_weights').split(','), dtype=float)
    print(f"loss_murmur_weights={loss_murmur_weights} CrossEntropyLoss Adam")
    murmur_loss_fn = nn.CrossEntropyLoss(torch.tensor(loss_murmur_weights)).to(device)

    loss_outcome_weights = np.array(config.get('outcome_loss_weights').split(','), dtype=float)
    print(f"loss_outcome_weights={loss_outcome_weights}  CrossEntropyLoss Adam")
    # outcome_loss_fn = nn.BCELoss(torch.tensor(loss_outcome_weights)).to(device)
    outcome_loss_fn = nn.CrossEntropyLoss(torch.tensor(loss_outcome_weights)).to(device)

    return murmur_model, outcome_model, murmur_loss_fn, outcome_loss_fn

if __name__ == "__main__":
    cnn = BasicCNN(1, 5, 0.2)
    summary(cnn.cuda(), (1, 64, 161))

