import os
from librosa.core import load
from librosa.feature import melspectrogram
import torch
import numpy as np
import threading
import time


class DataMaker(threading.Thread):
    def __init__(self, name, featurePath, labelDict, xCatData, yCatData, Type):
        super().__init__()
        self.x_cat_data = xCatData
        self.y_cat_data = yCatData
        self.name = name
        self.fType = Type
        self.features = None
        self.labels = None
        # 原始数据的路径
        self.featurePath = featurePath
        self.labelDict = labelDict

    def run(self):
        print("Starting " + self.name)
        try:
            if self.fType == "cutting":
                self.make_cutting_data()
            elif self.fType == "blocking":
                self.make_blocking_data()
        except Exception as e:
            print(e, end='\n')
        print("Exiting " + self.name)

    def make_cutting_data(self):
        xData, yData = list(), list()
        path = self.featurePath + self.name + '/'
        for j, filename in enumerate(os.listdir(path)):
            print(f"{self.name} {filename} ({j + 1})")
            WavPath = path + filename
            y, sr = load(WavPath, mono=True)
            S = melspectrogram(y, sr).T
            S = S[:-1 * (S.shape[0] % 128)]
            xData.append(S)
            yData.append(self.labelDict[self.name])

        self.features = torch.tensor(data=xData, device=device)
        self.labels = torch.tensor(data=yData, device=device)
        print(self.features.shape)
        print(self.labels.shape)
        self.x_cat_data.append(self.features)
        self.y_cat_data.append(self.labels)
        return

    def make_blocking_data(self):
        xData, yData = list(), list()
        path = self.featurePath + self.name + '/'
        for j, filename in enumerate(os.listdir(path)):
            print(f"{self.name} {filename} ({j + 1})")
            WavPath = path + filename
            y, sr = load(WavPath, mono=True)
            S = melspectrogram(y, sr).T
            S = S[:-1 * (S.shape[0] % 128)]
            num_chunk = S.shape[0] / 128
            data_chunks = np.split(S, num_chunk)
            xChunks, yChunks = list(), list()
            for unit in data_chunks:
                xChunks.append(unit)
                yChunks.append(self.labelDict[self.name])
            xData.append(xChunks)
            yData.append(yChunks)
        xData = [unit for record in xData for unit in record]
        yData = [unit for record in yData for unit in record]

        self.features = torch.tensor(data=xData, device=device)
        self.labels = torch.tensor(data=yData, device=device)
        print(self.features.shape)
        print(self.labels.shape)
        self.x_cat_data.append(self.features)
        self.y_cat_data.append(self.labels)
        return


def save(featureSavePath, labelSavePath):
    features = torch.cat(tensors=x_cat_data, dim=0)
    labels = torch.cat(tensors=y_cat_data, dim=0)
    print(features.shape)
    print(labels.shape)
    torch.save(features, featureSavePath)
    torch.save(labels, labelSavePath)
    print("数据已保存")
    return


def loads(featureSavePath, labelSavePath):
    features = torch.load(featureSavePath)
    labels = torch.load(labelSavePath)
    print(features.shape)
    print(labels.shape)
    print("数据已加载")


if __name__ == '__main__':
    device = torch.device('cpu')

    OriginalMusicDataPath = 'C:/Users/90430/Desktop/DataMaker/original_data/genres/'

    MusicLabels = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4,
                   'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9, }

    SavaPath = {
        "cutting": {"features": "cutting/features.pt",
                    "labels": "cutting/labels.pt"},

        "blocking": {"features": "blocking/features.pt",
                     "labels": "blocking/labels.pt"},
    }

    threadList = ["blues", "classical", "country", "disco", "hiphop",
                  "jazz", "metal", "pop", "reggae", "rock"]
    # 线程池
    threads = []

    # 拼接张量的列表
    x_cat_data = []
    y_cat_data = []
    fType = "cutting"

    start = time.time()
    # 创建新线程
    for tName in threadList:
        thread = DataMaker(tName, OriginalMusicDataPath, MusicLabels, x_cat_data, y_cat_data, fType)
        thread.start()
        threads.append(thread)

    # 等待所有线程完成
    for t in threads:
        t.join()

    end = time.time()
    print(f"多线程分批执行时间{end - start}")
    if len(x_cat_data) and len(y_cat_data):
        save(SavaPath[fType]["features"], SavaPath[fType]["labels"])
        loads(SavaPath[fType]["features"], SavaPath[fType]["labels"])
    else:
        print("张量为空")
