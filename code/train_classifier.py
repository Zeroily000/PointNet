import torch
from architecture import PointNetClassification
import torch.optim.lr_scheduler
import os
import math


def rot_mat(num_images):
    # roll = torch.Tensor(num_images, 1, 1).uniform_(0, math.pi)
    # Rx = torch.cat(
    #     (torch.cat((torch.ones(num_images, 1, 1), torch.zeros(num_images, 1, 1), torch.zeros(num_images, 1, 1)), 2),
    #      torch.cat((torch.zeros(num_images, 1, 1), torch.cos(roll), -torch.sin(roll)), 2),
    #      torch.cat((torch.zeros(num_images, 1, 1), torch.sin(roll), torch.cos(roll)), 2)), 1)
    #
    # pitch = torch.Tensor(num_images, 1, 1).uniform_(0, math.pi)
    # Ry = torch.cat((torch.cat((torch.cos(pitch), torch.zeros(num_images, 1, 1), torch.sin(pitch)), 2),
    #                 torch.cat(
    #                     (torch.zeros(num_images, 1, 1), torch.ones(num_images, 1, 1), torch.zeros(num_images, 1, 1)),
    #                     2),
    #                 torch.cat((-torch.sin(pitch), torch.zeros(num_images, 1, 1), torch.cos(pitch)), 2)), 1)

    yaw = torch.Tensor(num_images, 1, 1).uniform_(0, math.pi)
    Rz = torch.cat((torch.cat((torch.cos(yaw), -torch.sin(yaw), torch.zeros(num_images, 1, 1)), 2),
                    torch.cat((torch.sin(yaw), torch.cos(yaw), torch.zeros(num_images, 1, 1)), 2),
                    torch.cat((torch.zeros(num_images, 1, 1), torch.zeros(num_images, 1, 1), torch.ones(num_images, 1, 1)),2)), 1)

    # return torch.bmm(Rx, torch.bmm(Ry, Rz))
    return Rz


def eval_acc(X, t, classifier):
    batch_size = 32
    batch_num = X.size(0) / batch_size
    # train Accuracy
    correct = 0
    for bn in range(batch_num):
        X_batch = torch.autograd.Variable(X[bn * batch_size: bn * batch_size + batch_size, :, :])
        t_batch = t[bn * batch_size: bn * batch_size + batch_size]
        if torch.cuda.is_available():
            X_batch = X_batch.cuda()
        y_batch = classifier(X_batch).data.max(1)[1]
        # outputs = classifier(X_batch)
        # _, y_batch = torch.max(outputs.data, 1)
        correct += (y_batch.cpu() == t_batch).sum()

    return correct * 1.0 / X.size(0)


def train_classifier(learning_rate=0.001, regularization=0.001, reshuffle=True,
                     batch_size=32, step_size = 20, annealing=0.5):
    data_dir = '../dataset/ModelNet40'
    # catset = set([os.path.join(f) for f in os.listdir(dataset)])
    # cat2lab = {c: i for c, i in zip(catset, xrange(len(catset)))}

    print 'Loading data...'
    # load data
    # data: num_images x 3 x num_points
    # labels: num_images
    data_train = torch.load(os.path.join(data_dir, 'data_train.pth'))
    data_test = torch.load(os.path.join(data_dir, 'data_test.pth'))
    labels_train = torch.load(os.path.join(data_dir, 'labels_train.pth'))
    labels_test = torch.load(os.path.join(data_dir, 'labels_test.pth'))

    # augmentation
    augment = 4
    data_train = torch.cat([data_train] + [torch.bmm(R, data_train) for R in [rot_mat(data_train.size(0)) for i in xrange(augment)]], 0)
    data_train += torch.normal(means=torch.zeros(data_train.size()), std=0.02*torch.ones(data_train.size()))
    # data_train += torch.normal(means=0, std=0.02 * torch.ones(data_train.size()))
    labels_train = labels_train.repeat(augment + 1)
    print 'Done'

    num_images = data_train.size(0)
    X_train = data_train[num_images / 5:, :, :]
    t_train = labels_train[num_images / 5:]

    X_valid = data_train[:num_images / 5, :, :]
    t_valid = labels_train[num_images / 5:]

    X_test = data_test
    t_test = labels_test

    pn_classify = PointNetClassification(num_classes=40)
    if torch.cuda.is_available():
        pn_classify = pn_classify.cuda()


    optimizer = torch.optim.Adam(params=pn_classify.parameters(), lr=learning_rate, weight_decay=regularization)
    # optimizer = torch.optim.SGD(params=pn_classify.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001,
    #                             nesterov=True)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=annealing)

    num_images = X_train.size(0)
    batch_num = num_images / batch_size
    idx = torch.LongTensor(range(num_images))
    for epoch in xrange(500):
        scheduler.step()
        pn_classify.train(True)
        loss_epoch = 0

        if reshuffle:
            idx = torch.randperm(num_images)
        for bn in xrange(batch_num):
            X_batch = torch.autograd.Variable(X_train[idx[bn * batch_size: bn * batch_size + batch_size], :, :])
            t_batch = torch.autograd.Variable(t_train[idx[bn * batch_size: bn * batch_size + batch_size]])
            if torch.cuda.is_available():
                X_batch = X_batch.cuda()
                t_batch = t_batch.cuda()

            optimizer.zero_grad()
            y_batch = pn_classify(X_batch)
            loss = criterion(y_batch, t_batch)

            loss.backward()
            optimizer.step()

            loss_epoch += loss.data[0] / batch_num

        pn_classify.train(False)
        print epoch, ':', loss_epoch
        print 'train acc:', eval_acc(X_train, t_train, pn_classify)
        print 'validation acc:', eval_acc(X_valid, t_valid, pn_classify)
        print 'test acc:', eval_acc(X_test, t_test, pn_classify)



# test dimension
if __name__ == '__main__':
    torch.manual_seed(19270817)
    torch.cuda.manual_seed_all(19270817)
    train_classifier(learning_rate=0.001, regularization=0.001, reshuffle=True, batch_size=32, step_size=20, annealing=0.5)

