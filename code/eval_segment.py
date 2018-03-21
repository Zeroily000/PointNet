import torch
import numpy as np
import sklearn
import os
from architecture import PointNetSegmentation


print('Loading data... ', end='')
data_dir = '../dataset/S3DIS'
f = np.load(os.path.join(data_dir, 'data_test.npz'))
data, labels = sklearn.utils.shuffle(f['data'], f['labels'], random_state=19260817)
in_features = data.shape[1]
batch_size = 32

if not torch.cuda.is_available():
    data = data[:2*batch_size, ...]
    labels = labels[:2*batch_size, ...]

# zero-center xyz and rgb
data[:, :6, :] -= np.mean(data[:, :6, :], axis=2, keepdims=True)
# zero-center location
data[:, 6:, :] = data[:, 6:, :] * 2.0 - 1.0

X_test = torch.from_numpy(data.astype(np.float32))
t_test = torch.from_numpy(labels.astype(np.int64))

print('Done')


if torch.cuda.is_available():
    pn_segment = PointNetSegmentation(in_features=in_features, num_classes=13).cuda()
    pn_segment.load_state_dict(torch.load('../results/segmentation/93.40/PointNetSegment.pt'))
else:
    pn_segment = PointNetSegmentation(in_features=in_features, num_classes=13)
    pn_segment.load_state_dict(torch.load('../results/segmentation/93.40/PointNetSegment.pt', map_location=lambda storage, location: storage))


pn_segment.train(False)
num_batches = X_test.shape[0] // batch_size
C = np.zeros((13, 13))
for bn in range(num_batches):
    print('Batch {}/{}'.format(bn + 1, num_batches))
    X_batch = torch.autograd.Variable(X_test[bn * batch_size: bn * batch_size + batch_size, ...])
    t_batch = t_test[bn * batch_size: bn * batch_size + batch_size, ...]
    if torch.cuda.is_available():
        X_batch = X_batch.cuda()
        t_batch = t_batch.cuda()
    y_batch = pn_segment(X_batch)[0].data.max(1)[1]
    for i in range(13):
        for j in range(13):
            C[i, j] += np.logical_and(t_batch == i, y_batch == j).sum()

IoU = np.zeros(13)
for i in range(13):
    IoU[i] = C[i, i] / (C[i, :].sum() + C[:, i].sum() - C[i, i])
    print('IoU_{}: {:.4f}'.format(i, IoU[i]))
print('Mean IoU: {:.4f}'.format(IoU.mean()))
print('Overall Accuracy: {:.4f}'.format(np.diag(C).sum() / C.sum()))
