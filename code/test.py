# import numpy as np
# import glob
# import os
#
# def get_velo_scans(velo_files):
#     """Generator to parse velodyne binary files into arrays."""
#     for filename in velo_files:
#         scan = np.fromfile(filename, dtype=np.float32)
#         yield scan.reshape((-1, 4))
#
# def velo():
#     """Generator to read velodyne [x,y,z,reflectance] scan data from binary files."""
#     # Find all the Velodyne files
#     base_path = '../dataset/KITTI'
#     date = '2011_09_30'
#     drive = date + '_drive_' + '0016' + '_sync'
#     data_path = os.path.join(base_path, date, drive)
#
#     velo_path = os.path.join( data_path, 'velodyne_points', 'data', '*.bin')
#     velo_files = sorted(glob.glob(velo_path))
#
#     # Subselect the chosen range of frames, if any
#     # if frames is not None:
#     #     velo_files = [velo_files[i] for i in frames]
#
#     # Return a generator yielding Velodyne scans.
#     # Each scan is a Nx4 array of [x,y,z,reflectance]
#
#     for filename in velo_files:
#         scan = np.fromfile(filename, dtype=np.float32)
#         yield scan.reshape((-1, 4))
#
#     # return get_velo_scans(velo_files)
#
#
# if __name__ == '__main__':
#     # data = velo()
#     # print type(data)
#
#
#     # Find all the Velodyne files
#     data_path = '../dataset/KITTI/2011_09_30/2011_09_30_drive_0016_sync'
#     velo_path = os.path.join(data_path, 'velodyne_points', 'data', '*.bin')
#     velo_files = sorted(glob.glob(velo_path))
#
#
#     for filename in velo_files:
#         scan = np.fromfile(filename, dtype=np.float32).reshape((-1, 4))
#         print scan.shape
#         # yield scan.reshape((-1, 4))
#
# import trimesh
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from trimesh.sample import sample_surface
# import os
#
# def data_visualization(points):
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     ax.scatter(np.array(points)[0, :], np.array(points)[1, :], np.array(points)[2, :], s=0.1)
#     plt.show()
#
#
# if __name__ == '__main__':
#     # dataset = '../dataset/ModelNet40'
#     # catset = set([os.path.join(f) for f in os.listdir(dataset) if '.' not in f])
#     # cat2lab = {c: i for c, i in zip(catset, xrange(len(catset)))}
#     # lab2cat = {i: c for c, i in zip(catset, xrange(len(catset)))}
#     #
#     # data_dir = '../dataset/ModelNet40'
#     # f = np.load(os.path.join(data_dir, 'data_train.npz'))
#     # data = f['data']
#     # labels = f['labels']
#     # for i in xrange(data.shape[0]):
#     #     print lab2cat[labels[i]]
#     #     data_visualization(data[i])
#
#     file_dir = '../bed_0001.off'
#     # print file_dir
#     with open(file_dir, 'r') as f:
#         lines = f.readlines()
#         f.close()
#     # if lines[0].strip() != 'OFF':
#     #     lines[0] = lines[0].replace('OFF', '')
#     #     lines = ['OFF\n'] + lines
#     #     with open(file_dir, 'w') as f:
#     #         f.writelines(lines)
#     #         f.close()
#     with open(file_dir, 'r') as f:
#         lines = f.readlines()
#         num_vertices = eval(lines[1].strip().split()[0])
#         num_triangles = eval(lines[1].strip().split()[1])
#
#         vertices = lines[2: num_vertices+2]
#         faces = lines[num_vertices+2:]
#
#         triangles = np.array([[map(float, vertices[int(idx)].strip().split()) for idx in t.strip().split()[1: ]] for t in faces])
#
#         area_faces = np.linalg.norm(np.cross(triangles[:, 0, :] - triangles[:, 1, :], triangles[:, 0, :] - triangles[:, 2, :]), ord=2, axis=1)/2
#
#         mesh = trimesh.load(file_dir)
#
#         t1 = sample_surface(mesh, 1024)[0]
#         f.close()
#


# data_dir = '../dataset/S3DIS'
# f = np.load(os.path.join(data_dir, 'data_train.npz'))
# data, labels = f['data'], f['labels']

# import torch
# import torch.optim.lr_scheduler
# import numpy as np
# import os, time
# import sklearn
# from architecture import PointNetSegmentation
#
#
# torch.manual_seed(19270817)
# torch.cuda.manual_seed_all(19270817)
#
#
#
#
# if __name__ == '__main__':
#     # load data
#     # data: np.array, num_images x num_points x 9
#     # labels: np.array, num_images x num_points
#     print('Loading data... ', end='')
#     data_dir = '../dataset/S3DIS'
#     f = np.load(os.path.join(data_dir, 'data_train.npz'))
#     # data, labels = f['data'], f['labels']
#     data, labels = sklearn.utils.shuffle(f['data'], f['labels'])
#     in_features = data.shape[-1]
#
#     # zero-center xyz and rgb
#     data[:, :, :6] -= np.mean(data[:, :, :6], axis=1, keepdims=True)
#     # zero-center location
#     data[:, :, 6:] = data[:, :, 6:] * 2.0 - 1.0
#
#     data = np.transpose(data, axes=(0, 2, 1))
#
#
#     # for cpu test
#     if not torch.cuda.is_available():
#         data = data[:64, :, :]
#         labels = labels[:64]
#
#     X = torch.from_numpy(data.astype(np.float32))
#     print('Done\n')
#
#
#     pn_segment = PointNetSegmentation(in_features=in_features, num_classes=13)
#     if torch.cuda.is_available():
#         pn_segment = pn_segment.cuda()
#         pn_segment.load_state_dict(torch.load('../results/segmentation/PointNet_Classifier.pt'))
#
#     batch_size = 32
#     # y = []
#     pn_segment.train(False)
#     num_batches = X.shape[0] // batch_size
#     # for bn in range(num_batches):
#     #     print('Batch {}/{}'.format(bn, num_batches))
#     #     y.append(pn_segment(torch.autograd.Variable(X[bn * batch_size: bn * batch_size + batch_size, ...]).cuda())[0].data.max(1)[1]) # batch_size * 4096
#
#     # y = [pn_segment(torch.autograd.Variable(X[bn * batch_size: bn * batch_size + batch_size, ...]).cuda())[0].data.max(
#     #     1)[1].cpu() for bn in range(num_batches)]
#     y = [pn_segment(torch.autograd.Variable(X))[0].data.max(1)[1] for bn in range(num_batches)]
#
#     print(len(y))
#
#     t = np.concatenate(y, axis=0)
#
#     print(t.shape)


import numpy as np
import pandas as pd
from pyntcloud import PyntCloud

f = np.load('data_vis.npz')

i = 0
# positions = np.random.uniform(size=(100, 3)) - 0.5
positions = f['X'][i].T[:, :3]
points = pd.DataFrame(positions, columns=['x', 'y', 'z'])

# colors = (np.random.uniform(size=(100, 3)) * 255).astype(np.uint8)
colors = (f['y'][i].T[:, 3:6] * 255).astype(np.uint8)
points[['red', 'blue', 'green']] = pd.DataFrame(colors, index=points.index)

cloud = PyntCloud(points)
cloud.plot()


