from torch.utils.data import dataset, dataloader
import torch
from time import time

from torchvision.transforms import Compose

FeatureDataPath = {
    "cutting": 'C:/Users/*****/Desktop/DataMaker/music/cutting/features.pt',
    "blocking": 'C:/Users/*****/Desktop/DataMaker/music/blocking/features.pt',
}
LabelDataPath = {
    "cutting": 'C:/Users/*****/Desktop/DataMaker/music/cutting/labels.pt',
    "blocking": 'C:/Users/*****/Desktop/DataMaker/music/blocking/labels.pt',
}


class MyDataset(dataset.Dataset):
    def __init__(self, xPath, yPath):
        self.features = torch.load(xPath)
        self.labels = torch.load(yPath)
        self.len = self.labels.shape[0]

    def __getitem__(self, index):  # 返回的是tensor
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.len


def sets(FeaturePath, LabelPath, batch_size, proportion):
    raw_data = MyDataset(FeaturePath, LabelPath)
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(raw_data, proportion)
    train_loader = dataloader.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = dataloader.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = dataloader.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    t1 = time()
    batch = 1000
    train_data, valid_data, test_data = sets(FeatureDataPath["cutting"], LabelDataPath["cutting"], batch,[700,200,100])
    print(len(train_data))
    print(len(valid_data))
    print(len(test_data))
    t2 = time()



