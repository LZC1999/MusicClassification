# (spec): MelspectrogramStretch(num_bands=128, fft_len=2048, norm=spec_whiten, stretch_param=[0.4, 0.4])
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.audio import MelSpectrogramStretch
from src.model.BaseModel import BaseModel


class Config(object):
    """配置参数"""

    def __init__(self):
        self.dropout = 0.5  # 随机失活
        self.num_classes = 10  # 类别数
        self.num_epochs = 100  # epoch数
        self.learning_rate = 0.01  # 学习率
        self.width = 128
        self.filter_sizes = (128, 64, 32, 16)  # 卷积核尺寸
        # self.filter_sizes = (2, 4, 6)  # 卷积核尺寸
        self.num_filters = 256  # (channels数)
        self.batch_size = 100
        self.proportion = [8000, 1000, 1000]
        # self.class_list = ["blues", "classical", "country", "disco", "hiphop",
        #                   "jazz", "metal", "pop", "reggae", "rock"]

# 如果是立体声，应改为双声道
class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__()

        self.class_list = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
                           "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

        self.spec = MelSpectrogramStretch(hop_length=None,
                                          num_mels=128,
                                          fft_length=2048,
                                          norm='whiten',
                                          stretch_param=[0.4, 0.4])

        self.conv2d_0 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=[0, 0])
        self.batchNorm2d_0 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.elu_0 = nn.ELU(alpha=1.0)
        self.maxPool2d_0 = nn.MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
        self.dropout_0 = nn.Dropout(p=0.1)

        self.conv2d_1 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=[0, 0])
        self.batchNorm2d_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.elu_1 = nn.ELU(alpha=1.0)
        self.maxPool2d_1 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
        self.dropout_1 = nn.Dropout(p=0.1)

        self.conv2d_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=[0, 0])
        self.batchNorm2d_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.elu_2 = nn.ELU(alpha=1.0)
        self.maxPool2d_2 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
        self.dropout_2 = nn.Dropout(p=0.1)

        self.recur = nn.LSTM(128, 64, num_layers=2)
        self.dropout_3 = nn.Dropout(p=0.3)
        self.batchNorm1d_0 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.linear_0 = nn.Linear(in_features=64, out_features=10, bias=True)

    def forward(self, batch):
        # x-> (batch, time, channel)
        x = batch  # unpacking seqs, lengths and srs

        # x-> (batch, channel, time)
        xt = x.float().transpose(1, 2)
        # xt -> (batch, channel, freq, time)
        xt, lengths = self.spec(xt, lengths)

        # (batch, channel, freq, time)
        xt = self.net['convs'](xt)
        lengths = self.modify_lengths(lengths)

        # xt -> (batch, time, freq, channel)
        x = xt.transpose(1, -1)

        # xt -> (batch, time, channel*freq)
        batch, time = x.size()[:2]
        x = x.reshape(batch, time, -1)
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)

        # x -> (batch, time, lstm_out)
        x_pack, hidden = self.net['recur'](x_pack)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x_pack, batch_first=True)

        # (batch, lstm_out)
        x = self._many_to_one(x, lengths)
        # (batch, classes)
        x = self.net['dense'](x)

        x = F.log_softmax(x, dim=1)
        return x
