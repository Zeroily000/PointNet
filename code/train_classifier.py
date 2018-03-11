import torch
from architecture import PointNetClassification
from utils import prepare_data
import torch.optim.lr_scheduler

# test dimension
if __name__ == '__main__':
    print 'Loading data...'
    # prepare data
    # data: num_images x num_points x 3
    # labels: num_images
    data_train, labels_train = prepare_data(dataset='../dataset/ModelNet40', temp=320, dataset_type='train')
    data_test, labels_test = prepare_data(dataset='../dataset/ModelNet40', temp=80, dataset_type='test')

    num_images = data_train.shape[0]
    X_train = data_train[num_images/5:, :, :]
    t_train = labels_train[num_images/5:]

    X_valid = data_train[:num_images/5, :, :]
    t_valid = labels_train[num_images/5:]

    X_test = data_test
    t_test = labels_test
    print 'Done'





    # batch_num = len(data_train)
    # if torch.cuda.is_available():
    #     X_train = torch.autograd.Variable(torch.FloatTensor(data_train[batch_num/5:]).transpose(-1,-2).type(torch.cuda.FloatTensor))
    #     X_valid = torch.autograd.Variable(torch.FloatTensor(data_train[:batch_num/5]).transpose(-1,-2).type(torch.cuda.FloatTensor))
    #     t_train = torch.autograd.Variable(torch.LongTensor(labels_train[batch_num/5:]).type(torch.cuda.LongTensor))
    #     t_valid = torch.autograd.Variable(torch.LongTensor(labels_train[:batch_num/5]).type(torch.cuda.LongTensor))
    #
    #     X_test = torch.autograd.Variable(torch.FloatTensor(data_test).type(torch.cuda.FloatTensor).transpose(-1,-2))
    #     t_test = torch.autograd.Variable(torch.LongTensor(labels_test).type(torch.cuda.LongTensor))
    #
    # else:
    #     X_train = torch.autograd.Variable(torch.FloatTensor(data_train[batch_num / 5:]).transpose(-1,-2))
    #     X_valid = torch.autograd.Variable(torch.FloatTensor(data_train[:batch_num / 5]).transpose(-1,-2))
    #     t_train = torch.autograd.Variable(torch.LongTensor(labels_train[batch_num / 5:]))
    #     t_valid = torch.autograd.Variable(torch.LongTensor(labels_train[:batch_num / 5]))
    #
    #     X_test = torch.autograd.Variable(torch.FloatTensor(data_test).transpose(-1,-2))
    #     t_test = torch.autograd.Variable(torch.LongTensor(labels_test))


    N = 2048
    pn_classify = PointNetClassification(N, 40)
    if torch.cuda.is_available():
        pn_classify = pn_classify.cuda()


    #
    # learning_rate = 0.001
    # momentum = 0.9
    # batch_size = 32
    # annealing = 2.0
    optimizer = torch.optim.SGD(params=pn_classify.parameters() , lr=0.001, momentum=0.9, weight_decay=0.001, nesterov=True)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    batch_size = 32
    batch_num = X_train.shape[0] / batch_size

    for epoch in xrange(500):
        scheduler.step()
        pn_classify.train(True)
        loss_epoch = 0
        for bn in xrange(batch_num):
            X_batch = torch.autograd.Variable(X_train[bn*batch_size: bn*batch_size+batch_size, :, :])
            t_batch = torch.autograd.Variable(t_train[bn*batch_size: bn*batch_size+batch_size])
            if torch.cuda.is_available():
                X_batch = X_batch.cuda()
                t_batch = t_batch.cuda()

            optimizer.zero_grad()
            y_batch = pn_classify(X_batch)
            loss = criterion(y_batch, t_batch)

            loss.backward()
            optimizer.step()

            loss_epoch += loss.data[0] / batch_num
        print epoch, ':', loss_epoch




