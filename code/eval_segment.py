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
batch_size=32

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
    pn_segment.load_state_dict(torch.load('../results/segmentation/PointNet_Segment.pt'))
else:
    pn_segment = PointNetSegmentation(in_features=in_features, num_classes=13)
    pn_segment.load_state_dict(torch.load('../results/segmentation/PointNet_Segment.pt', map_location=lambda storage, location: storage))


pn_segment.train(False)
num_batches = X_test.shape[0] // batch_size
correct = 0
y = []
for bn in range(num_batches):
    print(bn)
    X_batch = torch.autograd.Variable(X_test[bn * batch_size: bn * batch_size + batch_size, ...])
    t_batch = t_test[bn * batch_size: bn * batch_size + batch_size, ...]
    if torch.cuda.is_available():
        X_batch = X_batch.cuda()
        t_batch = t_batch.cuda()
    y_batch = pn_segment(X_batch)[0].data.max(1)[1]
    correct += (y_batch == t_batch).sum()
    y.append(y_batch.cpu().numpy())
print('Test Accuracy: {:.4f}'.format(correct / X_test.shape[0] / X_test.shape[2]))

# np.savez('seg_test', X=X_test[:batch_size*num_batches, ...], y=np.concatenate(y, axis=0), t=t_test[:batch_size*num_batches, ...])
