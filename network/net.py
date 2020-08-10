import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint


class Net(nn.Module):
    def __init__(self, n_channel=3, n_classes=10):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(n_channel, 32, 3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool2_1 = nn.MaxPool2d(2, 2)
        self.bn2_1 = nn.BatchNorm2d(128)

        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2_2 = nn.MaxPool2d(2, 2)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(4 * 4 * 128, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x1_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x1_2 = self.pool1(F.relu(self.bn1_2(self.conv1_2(x1_1))))

        x2_1 = self.pool2_1(F.relu(self.bn2_1(self.conv2_1(x1_2))))
        x2_2 = self.pool2_2(F.relu(self.bn2_2(self.conv2_2(x2_1))))
        x = x2_2.view(-1, 4 * 4 * 128)

        x2 = F.relu(self.fc1(x))

        x = self.fc2(x2)

        return x, x2


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.constant_(m.bias, 0)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 128)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        latent = self.fc1(x)

        x = self.fc2(latent)
        x = x.view(x.size(0), 8, 4, 4)
        x = self.decoder(x)
        return x, latent


def call_bn(bn, x):
    return bn(x)


class CNN9LAYER(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25):
        self.dropout_rate = dropout_rate
        super(CNN9LAYER, self).__init__()
        self.c1 = nn.Conv2d(input_channel, 128, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.c8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.c9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.l_c1 = nn.Linear(128, n_outputs)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm2d(128)

    def forward(self, x, ):

        inter_out = {}

        h = x
        h = self.c1(h)
        inter_out['act_fc1'] = F.relu(call_bn(self.bn1, h))
        h = F.leaky_relu(call_bn(self.bn1, h), negative_slope=0.01)
        h = self.c2(h)
        inter_out['act_fc2'] = F.relu(call_bn(self.bn2, h))
        h = F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
        h = self.c3(h)
        inter_out['act_fc3'] = F.relu(call_bn(self.bn3, h))
        h = F.leaky_relu(call_bn(self.bn3, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c4(h)
        inter_out['act_fc4'] = F.relu(call_bn(self.bn4, h))
        h = F.leaky_relu(call_bn(self.bn4, h), negative_slope=0.01)
        h = self.c5(h)
        inter_out['act_fc5'] = F.relu(call_bn(self.bn5, h))
        h = F.leaky_relu(call_bn(self.bn5, h), negative_slope=0.01)
        h = self.c6(h)
        inter_out['act_fc6'] = F.relu(call_bn(self.bn6, h))
        h = F.leaky_relu(call_bn(self.bn6, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c7(h)
        inter_out['act_fc7'] = F.relu(call_bn(self.bn7, h))
        h = F.leaky_relu(call_bn(self.bn7, h), negative_slope=0.01)
        h = self.c8(h)
        inter_out['act_fc8'] = F.relu(call_bn(self.bn8, h))
        h = F.leaky_relu(call_bn(self.bn8, h), negative_slope=0.01)
        h = self.c9(h)
        inter_out['act_fc9'] = F.relu(call_bn(self.bn9, h))
        h = F.leaky_relu(call_bn(self.bn9, h), negative_slope=0.01)
        h = F.avg_pool2d(h, kernel_size=h.data.shape[2])

        h = h.view(h.size(0), h.size(1))
        logit = self.l_c1(h)

        return logit, h, inter_out


class LSTMTiny(nn.Module):
    def __init__(self, num_class):
        super(LSTMTiny, self).__init__()
        self.num_class = num_class
        self.num_words = 5000
        self.embed_dim = 128
        self.lstm_hidden_dim = 512

        self.embed = nn.Embedding(self.num_words, self.embed_dim)
        self.lstm = nn.LSTM(self.embed_dim, self.lstm_hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(self.lstm_hidden_dim, self.lstm_hidden_dim // 2)
        self.fc2 = nn.Linear(self.lstm_hidden_dim // 2, self.num_class)

    def forward(self, x):
        embed_x = self.embed(x)
        rnn_out, (hn, cn) = self.lstm(embed_x)

        hn = torch.transpose(hn, 0, 1).contiguous()
        hn = hn.view(hn.size(0), -1)

        feat = self.fc1(hn)
        # x = F.dropout(x)

        x = self.fc2(feat)

        return x, feat




