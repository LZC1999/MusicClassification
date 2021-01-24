# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):
    """配置参数"""

    def __init__(self):
        self.class_list = ["blues", "classical", "country", "disco", "hiphop",
                           "jazz", "metal", "pop", "reggae", "rock"]
        self.dropout = 1.0  # 随机失活
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 10  # epoch数                                          # mini-batch大小
        self.learning_rate = 1e-3  # 学习率
        self.width = 128
        self.hidden_size = 256  # lstm隐藏层
        self.num_layers = 1  # lstm层数
        self.batch_size = 100
        self.proportion = [800, 100, 100]


'''Recurrent Convolutional Neural Networks for Text Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(config.width, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxPool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.width, config.num_classes)

    def forward(self, x):
        x, _ = x
        out, _ = self.lstm(x)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxPool(out).squeeze()
        out = self.fc(out)
        return out
