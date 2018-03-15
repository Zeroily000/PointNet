import torch
import numpy as np
from architecture import PointNetClassification
import torch.optim.lr_scheduler
import os
import copy
import time

def rot_mat(num_images):
    # roll = torch.Tensor(num_images, 1, 1).uniform_(0, math.pi)
    # Rx = torch.cat(
    #     (torch.cat((torch.ones(num_images, 1, 1), torch.zeros(num_images, 1, 1), torch.zeros(num_images, 1, 1)), 2),
    #      torch.cat((torch.zeros(num_images, 1, 1), torch.cos(roll), -torch.sin(roll)), 2),
    #      torch.cat((torch.zeros(num_images, 1, 1), torch.sin(roll), torch.cos(roll)), 2)), 1)
    #
    pitch = torch.Tensor(num_images, 1, 1).uniform_(-np.pi, np.pi)
    Ry = torch.cat((torch.cat((torch.cos(pitch), torch.zeros(num_images, 1, 1), torch.sin(pitch)), 2),
                    torch.cat(
                        (torch.zeros(num_images, 1, 1), torch.ones(num_images, 1, 1), torch.zeros(num_images, 1, 1)),
                        2),
                    torch.cat((-torch.sin(pitch), torch.zeros(num_images, 1, 1), torch.cos(pitch)), 2)), 1)

    # yaw = torch.Tensor(num_images, 1, 1).uniform_(-np.pi, np.pi)
    # Rz = torch.cat((torch.cat((torch.cos(yaw), -torch.sin(yaw), torch.zeros(num_images, 1, 1)), 2),
    #                 torch.cat((torch.sin(yaw), torch.cos(yaw), torch.zeros(num_images, 1, 1)), 2),
    #                 torch.cat((torch.zeros(num_images, 1, 1), torch.zeros(num_images, 1, 1), torch.ones(num_images, 1, 1)),2)), 1)

    # return torch.bmm(Rx, torch.bmm(Ry, Rz))
    return Ry


def eval_acc(X, t, classifier):
    classifier.train(False)
    batch_size = 32
    num_batches = X.size(0) / batch_size
    correct = 0
    for bn in range(num_batches):
        X_batch = torch.autograd.Variable(X[bn * batch_size: bn * batch_size + batch_size, :, :])
        t_batch = t[bn * batch_size: bn * batch_size + batch_size]
        if torch.cuda.is_available():
            X_batch = X_batch.cuda()
            t_batch = t_batch.cuda()
        y_batch = classifier(X_batch)[0].data.max(1)[1]
        correct += (y_batch == t_batch).sum()
    return correct * 1.0 / X.size(0)

def eval_loss(X, t, classifier, criterion):
    classifier.train(False)
    batch_size = 32
    num_batches = X.size(0) / batch_size
    loss = 0.0
    for bn in xrange(num_batches):
        X_batch = torch.autograd.Variable(X[bn * batch_size: bn * batch_size + batch_size, :, :])
        t_batch = torch.autograd.Variable(t[bn * batch_size: bn * batch_size + batch_size])
        if torch.cuda.is_available():
            X_batch = X_batch.cuda()
            t_batch = t_batch.cuda()
        y_batch = classifier(X_batch)[0]
        loss += criterion(y_batch, t_batch).data[0]
    return loss / num_batches


def train_classifier(learning_rate=0.001, regularization=0.001, reshuffle=True,
                     batch_size=32, step_size = 20, annealing=0.5,
                     num_epochs = 500, early_stop = 3):

    print 'Loading data...',
    # load data
    # data: num_images x 3 x num_points
    # labels: num_images
    data_dir = '../dataset/ModelNet40'
    f = np.load(os.path.join(data_dir, 'data_train.npz'))
    data = torch.FloatTensor(f['data'])
    labels = torch.LongTensor(f['labels'])
    # data = torch.load(os.path.join(data_dir, 'data_train.pth'))
    # labels = torch.load(os.path.join(data_dir, 'labels_train.pth'))

    # for cpu
    if not torch.cuda.is_available():
        data = data[:160, :, :]
        labels = labels[:160]

    # data_test = torch.load(os.path.join(data_dir, 'data_test.pth'))
    # labels_test = torch.load(os.path.join(data_dir, 'labels_test.pth'))

    # augment = 4
    # data = torch.cat([data] + [torch.bmm(R, data)+torch.normal(means=torch.zeros(data.size()), std=0.02*torch.ones(data.size()))
    #                            for R in [rot_mat(data.size(0)) for i in xrange(augment)]], 0)
    # data += torch.normal(means=torch.zeros(data.size()), std=0.02*torch.ones(data.size()))
    # labels = labels.repeat(augment + 1)

    print 'Done\n'

    num_images = data.size(0)
    idx = torch.randperm(num_images)
    data = data[idx, :, :]
    labels = labels[idx]
    X_train = data[num_images / 5:, :, :]
    t_train = labels[num_images / 5:]

    X_valid = data[:num_images / 5, :, :]
    t_valid = labels[:num_images / 5]


    pn_classify = PointNetClassification(num_classes=40)
    if torch.cuda.is_available():
        pn_classify = pn_classify.cuda()


    # optimizer = torch.optim.Adam(params=pn_classify.parameters(), lr=learning_rate, weight_decay=regularization)
    optimizer = torch.optim.Adam(params=pn_classify.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=annealing)

    num_images = X_train.size(0)
    num_batches = num_images / batch_size
    idx = torch.randperm(num_images)
    cnt = 0


    loss_train, loss_valid = [eval_loss(X_train, t_train, pn_classify, criterion)], [eval_loss(X_valid, t_valid, pn_classify, criterion)]
    acc_train, acc_valid = [eval_acc(X_train, t_train, pn_classify)], [eval_acc(X_valid, t_valid, pn_classify)]
    print 'Epoch {}/{}'.format(0, num_epochs)
    print '-' * 10
    print 'Train Loss: {:.8f} Accuracy: {:.4f}'.format(loss_train[-1], acc_train[-1])
    print 'Valid Loss: {:.8f} Accuracy: {:.4f}\n'.format(loss_valid[-1], acc_valid[-1])

    acc_best = acc_valid[-1]
    # model_best = copy.deepcopy(pn_classify.state_dict())
    since = time.time()

    for epoch in xrange(num_epochs+1):
        if reshuffle:
            idx = torch.randperm(num_images)

        scheduler.step()
        pn_classify.train(True)
        for bn in xrange(num_batches):
            # data augmentation
            R = rot_mat(batch_size)
            X_batch = torch.bmm(R, X_train[idx[bn * batch_size: bn * batch_size + batch_size], :, :])
            noise = torch.normal(means=torch.zeros(X_batch.size()), std=0.02*torch.ones(X_batch.size()))
            noise[noise > 0.05] = 0.05
            noise[noise < -0.05] = -0.05
            X_batch = torch.autograd.Variable(X_batch + noise)

            # X_batch = torch.autograd.Variable(X_train[idx[bn * batch_size: bn * batch_size + batch_size], :, :])
            t_batch = torch.autograd.Variable(t_train[idx[bn * batch_size: bn * batch_size + batch_size]])
            I1 = torch.autograd.Variable(torch.zeros(batch_size, 3, 3) + torch.eye(3))
            I2 = torch.autograd.Variable(torch.zeros(batch_size, 64, 64) + torch.eye(64))
            if torch.cuda.is_available():
                X_batch = X_batch.cuda()
                t_batch = t_batch.cuda()
                I1 = I1.cuda()
                I2 = I2.cuda()

            optimizer.zero_grad()
            y_batch, A1, A2 = pn_classify(X_batch)
            loss = criterion(y_batch, t_batch) + \
                   regularization*torch.sum((I1 - torch.bmm(A1, A1.transpose(1, 2)))**2) + \
                   regularization *torch.sum((I2 - torch.bmm(A2, A2.transpose(1, 2)))**2)

            loss.backward()
            optimizer.step()

        pn_classify.train(False)
        loss_train.append(eval_loss(X_train, t_train, pn_classify, criterion))
        loss_valid.append(eval_loss(X_valid, t_valid, pn_classify, criterion))
        acc_train.append(eval_acc(X_train, t_train, pn_classify))
        acc_valid.append(eval_acc(X_valid, t_valid, pn_classify))
        print 'Epoch {}/{}'.format(epoch+1, num_epochs)
        print '-' * 10
        print 'Train Loss: {:.8f} Accuracy: {:.4f}'.format(loss_train[-1], acc_train[-1])
        print 'Valid Loss: {:.8f} Accuracy: {:.4f}\n'.format(loss_valid[-1], acc_valid[-1])


        cnt = cnt+1 if early_stop and loss_valid[-1] > loss_valid[-2] else 0
        if acc_valid[-1] > acc_best:
            acc_best = acc_valid[-1]
            # model_best = copy.deepcopy(pn_classify.state_dict())
            torch.save(pn_classify.state_dict(), '../models/PointNet_Classifier.pt')
            np.savez('loss', loss_train=loss_train, loss_valid=loss_valid)
            np.savez('accuracy', acc_train=acc_train, acc_valid=acc_valid)

        if cnt > early_stop:
            break

    time_elapsed = time.time() - since
    print 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed / 60, time_elapsed % 60)
    print 'Best val Accuracy: {:4f}'.format(acc_best)
    # pn_classify.load_state_dict(model_best)

    # np.savez('loss', loss_train=loss_train, loss_valid=loss_valid)
    # np.savez('accuracy', acc_train=acc_train, acc_valid=acc_valid)
    # torch.save(pn_classify.state_dict(), '../models/PointNet_Classifier.pt')


# test dimension
if __name__ == '__main__':
    torch.manual_seed(19270817)
    torch.cuda.manual_seed_all(19270817)
    train_classifier(learning_rate=0.001, regularization=0.001, reshuffle=True,
                     batch_size=32, step_size=20, annealing=0.5,
                     num_epochs=250, early_stop=5)
    # train_classifier(learning_rate=0.001, regularization=0.001, reshuffle=True,
    #                  batch_size=32, step_size=6, annealing=0.5,
    #                  num_epochs=250, early_stop=5)

