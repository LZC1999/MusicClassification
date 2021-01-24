from torch.autograd import Variable


def train(model, train_data, criterion, optimizer):
    model.train()
    for i, (x, y) in enumerate(train_data):
        x_train = Variable(x).cuda()
        y_train = Variable(y).cuda()
        outputs = model(x_train)
        model.zero_grad()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

