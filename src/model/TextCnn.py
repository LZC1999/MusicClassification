import torch
import torch.nn as nn
import torch.nn.functional as F


class Config(object):
    """配置参数"""

    def __init__(self):
        self.dropout = 0.5  # 随机失活
        self.num_classes = 10  # 类别数
        self.num_epochs = 10  # epoch数
        self.learning_rate = 0.01  # 学习率
        self.width = 128
        self.filter_sizes = (128, 64, 32, 16)  # 卷积核尺寸
        # self.filter_sizes = (2, 4, 6)  # 卷积核尺寸
        self.num_filters = 256  # (channels数)
        self.batch_size = 100
        self.proportion = [800, 100, 100]
        self.class_list = ["blues", "classical", "country", "disco", "hiphop",
                           "jazz", "metal", "pop", "reggae", "rock"]


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=1, eps=1e-5, momentum=0.1, affine=True)
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, groups=1, out_channels=config.num_filters, kernel_size=(height, config.width)) for
             height in
             config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = x.type(torch.FloatTensor).cuda()
        out = conv(x)
        out = F.relu(out)
        out = out.squeeze(3)
        out = F.max_pool1d(out, out.size(2))
        out = out.squeeze(2)
        return out

    def forward(self, x):
        out = x.unsqueeze(1)
        out = self.bn(out)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)  # 横着拼，按列拼接
        out = self.dropout(out)
        out = self.fc(out)
        return out
