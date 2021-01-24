import numpy as np
import torch
from sklearn import metrics
from torch.autograd import Variable


def test(config, model, test_data, criterion):
    model.eval()
    test_loss = []
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    # 不记录梯度信息，因为评估模型时不改权重和偏置
    with torch.no_grad():
        for i, (x, y) in enumerate(test_data):
            x_valid = Variable(x).cuda()
            y_valid = Variable(y).cuda()
            outputs = model(x_valid)
            loss = criterion(outputs, y_valid)
            predict = torch.max(outputs.data, 1)[1]  # 函数有两个返回值:每行的最大值和下标

            test_loss.append(loss.item())
            predict_all = np.append(predict_all, predict.cpu())
            labels_all = np.append(labels_all, y_valid.cpu())

    test_acc = metrics.accuracy_score(labels_all, predict_all)
    test_loss = np.mean(test_loss)

    report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)
    print("----------------------------------------------------------------------------------------------------")
    print(f"Test Loss: {test_loss * 100}%, Test Acc: {test_acc}")
    print("Precision, Recall and F1-Score")
    print(report)
    print("Confusion Matrix")
    print(confusion)
