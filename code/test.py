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

import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def data_visualization(points):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(np.array(points)[0, :], np.array(points)[1, :], np.array(points)[2, :], s=0.1)
    plt.show()

def sample_surface(mesh, count):
    """
    Sample the surface of a mesh, returning the specified number of points
    For individual triangle sampling uses this method:
    http://mathworld.wolfram.com/TrianglePointPicking.html
    Parameters
    ---------
    mesh: Trimesh object
    count: number of points to return
    Returns
    ---------
    samples: (count,3) points in space on the surface of mesh
    face_index: (count,) indices of faces for each sampled point
    """
    np.random.seed(0)
    # len(mesh.faces) float array of the areas of each face of the mesh
    area = mesh.area_faces
    # total area (float)
    area_sum = np.sum(area)
    # cumulative area (len(mesh.faces))
    area_cum = np.cumsum(area)
    face_pick = np.random.random(count) * area_sum
    face_index = np.searchsorted(area_cum, face_pick)

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = mesh.triangles[:, 0]
    tri_vectors = mesh.triangles[:, 1:].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = np.random.random((len(tri_vectors), 2, 1))

    # points will be distributed on a quadrilateral if we use 2 0-1 samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    return samples, face_index

def mesh_sample(triangles, area_faces, count):
    """
    Sample the surface of a mesh, returning the specified number of points
    For individual triangle sampling uses this method:
    http://mathworld.wolfram.com/TrianglePointPicking.html
    Parameters
    ---------
    mesh: Trimesh object
    count: number of points to return
    Returns
    ---------
    samples: (count,3) points in space on the surface of mesh
    face_index: (count,) indices of faces for each sampled point
    """

    # len(mesh.faces) float array of the areas of each face of the mesh
    np.random.seed(0)
    area = area_faces
    # total area (float)
    area_sum = np.sum(area)
    # cumulative area (len(mesh.faces))
    area_cum = np.cumsum(area)
    face_pick = np.random.random(count) * area_sum
    face_index = np.searchsorted(area_cum, face_pick)

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = triangles[:, 0]
    tri_vectors = triangles[:, 1:].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = np.random.random((len(tri_vectors), 2, 1))

    # points will be distributed on a quadrilateral if we use 2 0-1 samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    return samples

if __name__ == '__main__':
    # dataset = '../dataset/ModelNet40'
    # catset = set([os.path.join(f) for f in os.listdir(dataset) if '.' not in f])
    # cat2lab = {c: i for c, i in zip(catset, xrange(len(catset)))}
    # lab2cat = {i: c for c, i in zip(catset, xrange(len(catset)))}
    #
    # data_dir = '../dataset/ModelNet40'
    # f = np.load(os.path.join(data_dir, 'data_train.npz'))
    # data = f['data']
    # labels = f['labels']
    # for i in xrange(data.shape[0]):
    #     print lab2cat[labels[i]]
    #     data_visualization(data[i])

    file_dir = '../bed_0001.off'
    # print file_dir
    with open(file_dir, 'r') as f:
        lines = f.readlines()
        f.close()
    # if lines[0].strip() != 'OFF':
    #     lines[0] = lines[0].replace('OFF', '')
    #     lines = ['OFF\n'] + lines
    #     with open(file_dir, 'w') as f:
    #         f.writelines(lines)
    #         f.close()
    with open(file_dir, 'r') as f:
        lines = f.readlines()
        num_vertices = eval(lines[1].strip().split()[0])
        num_triangles = eval(lines[1].strip().split()[1])

        vertices = lines[2: num_vertices+2]
        faces = lines[num_vertices+2:]

        triangles = np.array([[map(float, vertices[int(idx)].strip().split()) for idx in t.strip().split()[1: ]] for t in faces])

        area_faces = np.linalg.norm(np.cross(triangles[:, 0, :] - triangles[:, 1, :], triangles[:, 0, :] - triangles[:, 2, :]), ord=2, axis=1)/2

        mesh = trimesh.load(file_dir)

        t1 = sample_surface(mesh, 1024)[0]
        t2 = mesh_sample(triangles, area_faces, 1024)
        f.close()

