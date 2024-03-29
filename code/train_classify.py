import torch
import torch.optim.lr_scheduler
import numpy as np
import os, time
import sklearn
from architecture import PointNetClassification

def rot_mat(num_images):
    pitch = np.random.uniform(-np.pi, np.pi, num_images)
    Ry = np.zeros((num_images, 3, 3), dtype=np.float32)
    Ry[:, 0, 0] = np.cos(pitch)
    Ry[:, 0, 2] = np.sin(pitch)
    Ry[:, 2, 2] = Ry[:, 0, 0]
    Ry[:, 2, 0] = -Ry[:, 0, 2]
    Ry[:, 1, 1] = 1.

    return torch.from_numpy(Ry)


def eval_acc(X, t, classifier):
    classifier.train(False)
    batch_size = 32
    num_batches = X.shape[0] // batch_size
    correct = 0
    for bn in range(num_batches):
        X_batch = torch.autograd.Variable(X[bn * batch_size: bn * batch_size + batch_size, ...])
        t_batch = t[bn * batch_size: bn * batch_size + batch_size]
        if torch.cuda.is_available():
            X_batch = X_batch.cuda()
            t_batch = t_batch.cuda()
        y_batch = classifier(X_batch)[0].data.max(1)[1]
        correct += (y_batch == t_batch).sum()
    return correct / X.shape[0]


def eval_loss(X, t, classifier, criterion):
    classifier.train(False)
    batch_size = 32
    num_batches = X.shape[0] // batch_size
    loss = 0
    for bn in range(num_batches):
        X_batch = torch.autograd.Variable(X[bn * batch_size: bn * batch_size + batch_size, ...])
        t_batch = torch.autograd.Variable(t[bn * batch_size: bn * batch_size + batch_size])
        if torch.cuda.is_available():
            X_batch = X_batch.cuda()
            t_batch = t_batch.cuda()
        y_batch = classifier(X_batch)[0]
        loss += criterion(y_batch, t_batch).data[0]
    return loss / num_batches


def train_classifier(learning_rate=0.001, regularization=0.001, reshuffle=True,
                     batch_size=32, step_size=20, annealing=0.5,
                     num_epochs=500, early_stop=3):

    # load data
    # data: np.array, num_images x 3 x num_points
    # labels: np.array, num_images
    print('Loading data...', end='')
    data_dir = '../dataset/ModelNet40'
    f = np.load(os.path.join(data_dir, 'data_train_1024.npz'))
    data, labels = sklearn.utils.shuffle(f['data'], f['labels'])

    in_features = data.shape[1]

    ########################### for cpu test ###########################
    if not torch.cuda.is_available():
        data = data[:320, :, :]
        labels = labels[:320]
    ####################################################################

    X_train = torch.from_numpy(data[data.shape[0] // 9:, :, :].astype(np.float32))
    t_train = torch.from_numpy(labels[data.shape[0] // 9:].astype(np.int64))

    X_valid = torch.from_numpy(data[:data.shape[0] // 9, :, :].astype(np.float32))
    t_valid = torch.from_numpy(labels[:data.shape[0] // 9].astype(np.int64))

    num_batches = X_train.shape[0] // batch_size
    print('Done\n')

    pn_classify = PointNetClassification(in_features=in_features, num_classes=40)
    if torch.cuda.is_available():
        pn_classify = pn_classify.cuda()


    optimizer = torch.optim.Adam(params=pn_classify.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=annealing)

    loss_train = [eval_loss(X_train, t_train, pn_classify, criterion)]
    loss_valid = [eval_loss(X_valid, t_valid, pn_classify, criterion)]
    acc_train = [eval_acc(X_train, t_train, pn_classify)]
    acc_valid = [eval_acc(X_valid, t_valid, pn_classify)]
    print('Epoch {}/{}'.format(0, num_epochs))
    print('-' * 10)
    print('Train Loss: {:.8f} Accuracy: {:.4f}'.format(loss_train[-1], acc_train[-1]))
    print('Valid Loss: {:.8f} Accuracy: {:.4f}\n'.format(loss_valid[-1], acc_valid[-1]))

    acc_best = acc_valid[-1]
    cnt = 0


    # start training
    since = time.time()
    for epoch in range(num_epochs + 1):
        if reshuffle:
            idx = torch.randperm(X_train.shape[0])
        else:
            idx = torch.from_numpy(np.arange(X_train.shape[0]))

        scheduler.step()
        pn_classify.train(True)
        for bn in range(num_batches):
            X_batch = X_train[idx[bn * batch_size: bn * batch_size + batch_size], :, :]
            t_batch = t_train[idx[bn * batch_size: bn * batch_size + batch_size]]

            I1 = torch.autograd.Variable(torch.eye(in_features))
            I2 = torch.autograd.Variable(torch.eye(64))

            # data augmentation
            R = rot_mat(batch_size)
            noise = torch.from_numpy(np.clip(np.random.normal(loc=0.0, scale=0.02, size=X_batch.shape), -0.05, 0.05).astype(np.float32))
            X_batch = torch.autograd.Variable(torch.cat((X_batch, torch.bmm(R, X_batch) + noise), 0))
            t_batch = torch.autograd.Variable(t_batch.repeat(2))

            if torch.cuda.is_available():
                X_batch = X_batch.cuda()
                t_batch = t_batch.cuda()
                I1 = I1.cuda()
                I2 = I2.cuda()

            optimizer.zero_grad()
            y_batch, A1, A2 = pn_classify(X_batch)
            loss = criterion(y_batch, t_batch) + \
                   regularization * torch.sum((I1 - torch.bmm(A1, A1.transpose(1, 2))) ** 2) + \
                   regularization * torch.sum((I2 - torch.bmm(A2, A2.transpose(1, 2))) ** 2)

            loss.backward()
            optimizer.step()

        pn_classify.train(False)
        loss_train.append(eval_loss(X_train, t_train, pn_classify, criterion))
        loss_valid.append(eval_loss(X_valid, t_valid, pn_classify, criterion))
        acc_train.append(eval_acc(X_train, t_train, pn_classify))
        acc_valid.append(eval_acc(X_valid, t_valid, pn_classify))
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        print('Train Loss: {:.8f} Accuracy: {:.4f}'.format(loss_train[-1], acc_train[-1]))
        print('Valid Loss: {:.8f} Accuracy: {:.4f}\n'.format(loss_valid[-1], acc_valid[-1]))

        cnt = cnt + 1 if early_stop and loss_valid[-1] > loss_valid[-2] else 0
        if acc_valid[-1] > acc_best:
            acc_best = acc_valid[-1]
            torch.save(pn_classify.state_dict(), '../results/classification/PointNetClassify.pt')
            np.savez('../results/classification/loss', train=loss_train, valid=loss_valid)
            np.savez('../results/classification/accuracy', train=acc_train, valid=acc_valid)

        if cnt >= early_stop:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Accuracy: {:4f}'.format(acc_best))


if __name__ == '__main__':
    torch.manual_seed(19270817)
    torch.cuda.manual_seed_all(19270817)
    train_classifier(learning_rate=0.001, regularization=0.001, reshuffle=True,
                     batch_size=32, step_size=20, annealing=0.5,
                     num_epochs=250, early_stop=4)

