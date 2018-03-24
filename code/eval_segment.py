import torch
import numpy as np
import sklearn
import os, random
import matplotlib.pyplot as plt
import trimesh
from architecture import PointNetSegmentation


print('Loading data... ', end='')
cats = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
        'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
data_dir = '../dataset/S3DIS'
f = np.load(os.path.join(data_dir, 'data_test.npz'))
data, labels = sklearn.utils.shuffle(f['data'], f['labels'], random_state=19260817)
in_features = data.shape[1]
batch_size = 32

########################### for cpu test ###########################
if not torch.cuda.is_available():
    data = data[:2*batch_size, ...]
    labels = labels[:2*batch_size, ...]
####################################################################

# zero-center xyz and rgb
data[:, :6, :] -= np.mean(data[:, :6, :], axis=2, keepdims=True)
# zero-center location
data[:, 6:, :] = data[:, 6:, :] * 2.0 - 1.0

X = torch.from_numpy(data.astype(np.float32))
t = torch.from_numpy(labels.astype(np.int64))

print('Done')


if torch.cuda.is_available():
    pn_segment = PointNetSegmentation(in_features=in_features, num_classes=13).cuda()
    pn_segment.load_state_dict(torch.load('../results/segmentation/93.40/PointNetSegment.pt'))
else:
    pn_segment = PointNetSegmentation(in_features=in_features, num_classes=13)
    pn_segment.load_state_dict(torch.load('../results/segmentation/93.40/PointNetSegment.pt',
                                          map_location=lambda storage, location: storage))


pn_segment.train(False)
num_batches = X.shape[0] // batch_size
C = np.zeros((13, 13))
y = []
for bn in range(num_batches):
    print('Batch {}/{}'.format(bn + 1, num_batches))
    X_batch = torch.autograd.Variable(X[bn * batch_size: bn * batch_size + batch_size, ...])
    t_batch = t[bn * batch_size: bn * batch_size + batch_size, ...]
    if torch.cuda.is_available():
        X_batch = X_batch.cuda()
        t_batch = t_batch.cuda()
    y_batch = pn_segment(X_batch)[0].data.max(1)[1]
    y.append(y_batch.cpu().numpy())
    for i in range(13):
        for j in range(13):
            C[i, j] += np.logical_and(t_batch == i, y_batch == j).sum()

IoU = np.zeros(13)
for i in range(13):
    IoU[i] = C[i, i] / (C[i, :].sum() + C[:, i].sum() - C[i, i])
    print('IoU_{}: {:.4f}'.format(i, IoU[i]))
print('\nMean IoU: {:.4f}'.format(IoU.mean()))
print('Overall Accuracy: {:.4f}'.format(np.diag(C).sum() / C.sum()))


plt.figure(figsize=(8,8))
plt.imshow(C)
plt.xticks(np.arange(len(cats)), cats, rotation='vertical')
plt.yticks(np.arange(len(cats)), cats)
plt.show()


cat2clr = {'ceiling':     [0,255,0],
           'floor':       [0,0,255],
           'wall':        [0,255,255],
           'beam':        [255,255,0],
           'column':      [255,0,255],
           'window':      [100,100,255],
           'door':        [200,200,100],
           'table':       [170,120,200],
           'chair':       [255,0,0],
           'sofa':        [200,100,100],
           'bookcase':    [10,200,100],
           'board':       [200,200,200],
           'clutter':     [50,50,50]}
lab2clr = [cat2clr[c] for c in cats]

X_test = np.transpose(X.numpy(), (0, 2, 1)) # num_images x num_points x 9
y_test = np.concatenate(y, axis=0) # num_images x num_points
t_test = t[:y_test.shape[0]].numpy()
i = random.randrange(y_test.shape[0])
points1 = trimesh.points.PointCloud(vertices = X_test[i, :, :3], color = list(map(lambda x : lab2clr[x], y_test[i, :])))
points2 = trimesh.points.PointCloud(vertices = X_test[i, :, :3], color = list(map(lambda x : lab2clr[x], t_test[i, :])))
points1.show()
points2.show()
