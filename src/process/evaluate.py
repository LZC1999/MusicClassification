import numpy as np
import torch
from sklearn import metrics
from torch.autograd import Variable


def evaluate(model, valid_data, criterion, scheduler=None):
    model.eval()
    val_loss = []
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    # 不记录梯度信息，因为评估模型时不改权重和偏置
    with torch.no_grad():
        for i, (x, y) in enumerate(valid_data):
            x_valid = Variable(x).cuda()
            y_valid = Variable(y).cuda()
            outputs = model(x_valid)
            loss = criterion(outputs, y_valid)
            predict = torch.max(outputs.data, 1)[1]  # 函数有两个返回值:每行的最大值和下标
            val_loss.append(loss.item())
            predict_all = np.append(predict_all, predict.cpu())
            labels_all = np.append(labels_all, y_valid.cpu())
    val_acc = metrics.accuracy_score(labels_all, predict_all)
    val_loss = np.mean(val_loss)
    if scheduler is not None:
        scheduler.step()  # 学习率衰减
    return val_acc, val_loss
