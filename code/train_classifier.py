import torch
from architecture import PointNetClassification
from utils import prepare_data


# test dimension
if __name__ == '__main__':
    # prepare data
    data_train, data_test, labels_train, labels_test = prepare_data(dataset='../dataset/ModelNet40', batch_size=32)

    batch_num = len(data_train)
    if torch.cuda.is_available():
        X_train = torch.autograd.Variable(torch.FloatTensor(data_train[batch_num/5:]).transpose(-1,-2).type(torch.cuda.FloatTensor))
        X_valid = torch.autograd.Variable(torch.FloatTensor(data_train[:batch_num/5]).transpose(-1,-2).type(torch.cuda.FloatTensor))
        t_train = torch.autograd.Variable(torch.LongTensor(labels_train[batch_num/5:]).type(torch.cuda.LongTensor))
        t_valid = torch.autograd.Variable(torch.LongTensor(labels_train[:batch_num/5]).type(torch.cuda.LongTensor))

        X_test = torch.autograd.Variable(torch.FloatTensor(data_test).type(torch.cuda.FloatTensor).transpose(-1,-2))
        t_test = torch.autograd.Variable(torch.LongTensor(labels_test).type(torch.cuda.LongTensor))

    else:
        X_train = torch.autograd.Variable(torch.FloatTensor(data_train[batch_num / 5:]).transpose(-1,-2))
        X_valid = torch.autograd.Variable(torch.FloatTensor(data_train[:batch_num / 5]).transpose(-1,-2))
        t_train = torch.autograd.Variable(torch.LongTensor(labels_train[batch_num / 5:]))
        t_valid = torch.autograd.Variable(torch.LongTensor(labels_train[:batch_num / 5]))

        X_test = torch.autograd.Variable(torch.FloatTensor(data_test).transpose(-1,-2))
        t_test = torch.autograd.Variable(torch.LongTensor(labels_test))


    N = 2048
    pn_classify = PointNetClassification(N, 40)
    # out = pn_classify(X_train[0])
    # print 'Classifier:', out.size()


    #
    # learning_rate = 0.001
    # momentum = 0.9
    # batch_size = 32
    # annealing = 2.0
    optimizer = torch.optim.SGD(params=pn_classify.parameters() , lr=0.001, momentum=0.9, weight_decay=0.001, nesterov=True)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in xrange(500):
        scheduler.step()
        pn_classify.train(True)
        print epoch
        for batch in xrange(batch_num):
            optimizer.zero_grad()
            # forward
            y_train = pn_classify(X_train[batch])  # N * 256
            # _, preds = torch.max(y.data, 1)  # N
            loss = criterion(y_train, t_train[batch])

            loss.backward()
            optimizer.step()

            print loss.data[0]




