from torch.backends.cudnn import deterministic
import numpy as np
import torch
from time import time
from src.utils.earlystopping import EarlyStopping
from src.utils.data import sets, FeatureDataPath, LabelDataPath
from src.process.train import train
from src.process.evaluate import evaluate
from src.process.test import test
from src.utils.view import draw
from src.model.TextCnn import Model, Config


np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

if __name__ == '__main__':
    config = Config()
    train_data, valid_data, test_data = sets(FeatureDataPath["cutting"], LabelDataPath["cutting"],
                                             config.batch_size, config.proportion)

    model = Model(config)
    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    early_stopping = EarlyStopping(patience=20, verbose=True)
    best_loss = float('inf')
    use_time = 0
    Loss, Acc = [], []

    for epoch in range(config.num_epochs):
        start = time()
        train(model, train_data, criterion, optimizer)
        val_acc, val_loss = evaluate(model, valid_data, criterion, scheduler)
        end = time()

        Loss.append(val_loss * 100)
        Acc.append(val_acc)
        use_time += (end - start)
        print(f'Epoch [{epoch + 1}/{config.num_epochs}]')
        print(f"Time: {round(end - start, 2)}, Val Loss: {round(val_loss, 5)}, Val Acc: {val_acc * 100}%")
        '''
        if best_loss > val_loss:
            best_loss = val_loss
            torch.save(models.state_dict(), "models.pt")
        '''
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    draw(Loss, Acc)
    print(f"Train Time: {round(use_time / 60, 2)} min")
    test(config, model, test_data, criterion)
