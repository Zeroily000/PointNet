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
#
import torch
print torch.__version__


