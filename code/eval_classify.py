import torch
import numpy as np
import os
from architecture import PointNetClassification


print('Loading data...', end='')
data_dir = '../dataset/ModelNet40'
f = np.load(os.path.join(data_dir, 'data_test_1024.npz'))
data, labels = f['data'], f['labels']
in_features = data.shape[1]
batch_size = 32
# for cpu test
if not torch.cuda.is_available():
    data = data[:batch_size*2, :, :]
    labels = labels[:batch_size*2]

X = torch.from_numpy(data.astype(np.float32))
t = torch.from_numpy(labels.astype(np.int64))

num_batches = X.shape[0] // batch_size
print('Done\n')

if torch.cuda.is_available():
    pn_classify = PointNetClassification(in_features=in_features, num_classes=40).cuda()
    pn_classify.load_state_dict(torch.load('../results/classification/86.73/PointNetClassify.pt'))
else:
    pn_classify = PointNetClassification(in_features=in_features, num_classes=40)
    pn_classify.load_state_dict(torch.load('../results/classification/86.73/PointNetClassify.pt',
                                          map_location=lambda storage, location: storage))

pn_classify.train(False)
correct = 0
for bn in range(num_batches):
    X_batch = torch.autograd.Variable(X[bn * batch_size: bn * batch_size + batch_size, ...])
    t_batch = t[bn * batch_size: bn * batch_size + batch_size]
    if torch.cuda.is_available():
        X_batch = X_batch.cuda()
        t_batch = t_batch.cuda()
    y_batch = pn_classify(X_batch)[0].data.max(1)[1]
    correct += (y_batch == t_batch).sum()
print('Test Accuracy: {:.4f}'.format(correct / X.shape[0]))

