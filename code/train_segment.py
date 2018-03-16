import torch
import torch.optim.lr_scheduler
import numpy as np
import os, time
from architecture import PointNetSegmentation

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


def train_segment(learning_rate=0.001, regularization=0.001, reshuffle=True,
                     batch_size=32, step_size=20, annealing=0.5,
                     num_epochs=500, early_stop=3):
    pn_segment = PointNetSegmentation(num_classes=13)
    if torch.cuda.is_available():
        pn_segment = pn_segment.cuda()

    # model_best = copy.deepcopy(pn_classify.state_dict())

    # optimizer = torch.optim.Adam(params=pn_classify.parameters(), lr=learning_rate, weight_decay=regularization)
    optimizer = torch.optim.Adam(params=pn_segment.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=annealing)


    # load data
    # data: np.array, num_images x 3 x num_points
    # labels: np.array, num_images
    print 'Loading data...',
    data_dir = '../dataset/indoor3d_sem_seg_hdf5_data'
    f = np.load(os.path.join(data_dir, 'data_train.npz'))
    data, labels = f['data'], f['labels']

    # for cpu test
    if not torch.cuda.is_available():
        data = data[:320, :, :]
        labels = labels[:320]

    # data_test = torch.load(os.path.join(data_dir, 'data_test.pth'))
    # labels_test = torch.load(os.path.join(data_dir, 'labels_test.pth'))

    # augment = 4
    # data = torch.cat([data] + [torch.bmm(R, data)+torch.normal(means=torch.zeros(data.size()), std=0.02*torch.ones(data.size()))
    #                            for R in [rot_mat(data.size(0)) for i in xrange(augment)]], 0)
    # data += torch.normal(means=torch.zeros(data.size()), std=0.02*torch.ones(data.size()))
    # labels = labels.repeat(augment + 1)

    X_train = torch.from_numpy(data[data.shape[0] / 9:, :, :].astype(np.float32))
    t_train = torch.from_numpy(labels[data.shape[0] / 9:].astype(np.int64))

    X_valid = torch.from_numpy(data[:data.shape[0] / 9, :, :].astype(np.float32))
    t_valid = torch.from_numpy(labels[:data.shape[0] / 9].astype(np.int64))

    num_batches = X_train.shape[0] / batch_size
    print 'Done\n'

    loss_train = [eval_loss(X_train, t_train, pn_segment, criterion)]
    loss_valid = [eval_loss(X_valid, t_valid, pn_segment, criterion)]
    acc_train = [eval_acc(X_train, t_train, pn_segment)]
    acc_valid = [eval_acc(X_valid, t_valid, pn_segment)]
    print 'Epoch {}/{}'.format(0, num_epochs)
    print '-' * 10
    print 'Train Loss: {:.8f} Accuracy: {:.4f}'.format(loss_train[-1], acc_train[-1])
    print 'Valid Loss: {:.8f} Accuracy: {:.4f}\n'.format(loss_valid[-1], acc_valid[-1])

    acc_best = acc_valid[-1]
    cnt = 0


    # start training
    since = time.time()
    for epoch in xrange(num_epochs + 1):
        if reshuffle:
            idx = torch.randperm(X_train.shape[0])
        else:
            idx = torch.from_numpy(np.arange(X_train.shape[0]))

        scheduler.step()
        pn_segment.train(True)
        for bn in xrange(num_batches):
            X_batch = X_train[idx[bn * batch_size: bn * batch_size + batch_size], :, :]
            t_batch = t_train[idx[bn * batch_size: bn * batch_size + batch_size]]
            I1 = torch.autograd.Variable(torch.eye(3))
            I2 = torch.autograd.Variable(torch.eye(64))

            if torch.cuda.is_available():
                X_batch = X_batch.cuda()
                t_batch = t_batch.cuda()
                I1 = I1.cuda()
                I2 = I2.cuda()

            optimizer.zero_grad()
            y_batch, A1, A2 = pn_segment(X_batch)
            loss = criterion(y_batch, t_batch) + \
                   regularization * torch.sum((I1 - torch.bmm(A1, A1.transpose(1, 2))) ** 2) + \
                   regularization * torch.sum((I2 - torch.bmm(A2, A2.transpose(1, 2))) ** 2)

            loss.backward()
            optimizer.step()

        pn_segment.train(False)
        loss_train.append(eval_loss(X_train, t_train, pn_segment, criterion))
        loss_valid.append(eval_loss(X_valid, t_valid, pn_segment, criterion))
        acc_train.append(eval_acc(X_train, t_train, pn_segment))
        acc_valid.append(eval_acc(X_valid, t_valid, pn_segment))
        print 'Epoch {}/{}'.format(epoch + 1, num_epochs)
        print '-' * 10
        print 'Train Loss: {:.8f} Accuracy: {:.4f}'.format(loss_train[-1], acc_train[-1])
        print 'Valid Loss: {:.8f} Accuracy: {:.4f}\n'.format(loss_valid[-1], acc_valid[-1])

        cnt = cnt + 1 if early_stop and loss_valid[-1] > loss_valid[-2] else 0
        if acc_valid[-1] > acc_best:
            acc_best = acc_valid[-1]
            # model_best = copy.deepcopy(pn_classify.state_dict())
            torch.save(pn_segment.state_dict(), '../results/PointNet_Classifier.pt')
            np.savez('../results/loss', train=loss_train, valid=loss_valid)
            np.savez('../results/accuracy', train=acc_train, valid=acc_valid)

        if cnt >= early_stop:
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
    train_segment(learning_rate=0.001, regularization=0.001, reshuffle=True,
                     batch_size=32, step_size=20, annealing=0.5,
                     num_epochs=250, early_stop=4)

