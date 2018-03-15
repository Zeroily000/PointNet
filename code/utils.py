# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import trimesh
import numpy as np
import torch
import os, random, time


# def sample_surface(mesh, count):
#     """
#     Sample the surface of a mesh, returning the specified number of points
#     For individual triangle sampling uses this method:
#     http://mathworld.wolfram.com/TrianglePointPicking.html
#     Parameters
#     ---------
#     mesh: Trimesh object
#     count: number of points to return
#     Returns
#     ---------
#     samples: (count,3) points in space on the surface of mesh
#     face_index: (count,) indices of faces for each sampled point
#     """
#
#     # len(mesh.faces) float array of the areas of each face of the mesh
#     area = mesh.area_faces
#     # total area (float)
#     area_sum = np.sum(area)
#     # cumulative area (len(mesh.faces))
#     area_cum = np.cumsum(area)
#     face_pick = np.random.random(count) * area_sum
#     face_index = np.searchsorted(area_cum, face_pick)
#
#     # pull triangles into the form of an origin + 2 vectors
#     tri_origins = mesh.triangles[:, 0]
#     tri_vectors = mesh.triangles[:, 1:].copy()
#     tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))
#
#     # pull the vectors for the faces we are going to sample from
#     tri_origins = tri_origins[face_index]
#     tri_vectors = tri_vectors[face_index]
#
#     # randomly generate two 0-1 scalar components to multiply edge vectors by
#     random_lengths = np.random.random((len(tri_vectors), 2, 1))
#
#     # points will be distributed on a quadrilateral if we use 2 0-1 samples
#     # if the two scalar components sum less than 1.0 the point will be
#     # inside the triangle, so we find vectors longer than 1.0 and
#     # transform them to be inside the triangle
#     random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
#     random_lengths[random_test] -= 1.0
#     random_lengths = np.abs(random_lengths)
#
#     # multiply triangle edge vectors by the random lengths and sum
#     sample_vector = (tri_vectors * random_lengths).sum(axis=1)
#
#     # finally, offset by the origin to generate
#     # (n,3) points in space on the triangle
#     samples = sample_vector + tri_origins
#
#     return samples, face_index

# def read_off(fpath, num_points):
#     '''
#     Read point cloud from a single image file
#     :param filename: filename dir
#     :return: points Nx3 list of string
#     '''
#     # with open(fpath, 'r') as f:
#     #     lines = f.readlines()
#
#     f = open(fpath, 'r')
#     if 'OFF' != f.readline().strip():
#         f.seek(3)
#
#     N = int(f.readline().strip().split(' ')[0])
#
#     if N < num_points:
#         return []
#     points = [[float(x) for x in f.readline().strip().split(' ')] for i in xrange(0, N/num_points*num_points, N/num_points)]
#     f.close()
#     return points
def mesh_sample(off_file, num_points):
    with open(off_file, 'r') as f:
        lines = f.readlines()
        f.close()
    if lines[0].strip() != 'OFF':
        lines[0] = lines[0].replace('OFF', '')
        lines = ['OFF\n'] + lines

    num_vertices = eval(lines[1].strip().split()[0])
    num_triangles = eval(lines[1].strip().split()[1])

    vertices = lines[2: num_vertices+2]
    faces = lines[num_vertices+2: num_vertices+2+num_triangles]

    triangles = np.array([[map(float, vertices[int(idx)].strip().split()) for idx in t.strip().split()[1: ]] for t in faces])

    area_faces = np.linalg.norm(np.cross(triangles[:, 0, :] - triangles[:, 1, :], triangles[:, 0, :] - triangles[:, 2, :]), ord=2, axis=1)/2

    # area = area_faces
    # total area (float)
    area_sum = np.sum(area_faces)
    # cumulative area (len(mesh.faces))
    area_cum = np.cumsum(area_faces)
    face_pick = np.random.random(num_points) * area_sum
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

def prepare_data(dataset, cat2lab, dataset_type = '', num_points = 1024):
    '''
    Load all data
    :param dataset: folder path
    :param batch_size:
    :return: data: BxNx3 list of string; label: BxN
    '''

    # load train file paths
    paths = [(root, filename) for root, _, files in os.walk(dataset, topdown=False) for filename in files if
             dataset_type in root and '.off' in filename]

    # random shuffle
    random.shuffle(paths)

    # data, labels = [], []
    # for root, filename in paths:
    #     points = read_off(os.path.join(root, filename), num_points)
    #     if len(points) != 0:
    #         data.append(points)
    #         labels.append(cat2lab[root.split('/')[-2]])
    data, labels = [], []
    for i, (root, filename) in enumerate(paths):
        print i

        data.append(mesh_sample(os.path.join(root, filename), num_points))
        labels.append(cat2lab[root.split('/')[-2]])
        # file_dir = os.path.join(root, filename)
        # print file_dir
        # with open(file_dir, 'r') as f:
        #     lines = f.readlines()
        #     f.close()
        # if lines[0].strip() != 'OFF':
        #     lines[0] = lines[0].replace('OFF', '')
        #     lines = ['OFF\n'] + lines
        #     with open(file_dir, 'w') as f:
        #         f.writelines(lines)
        #         f.close()

        # mesh = trimesh.load(file_dir)
        # samples = sample_surface(mesh, num_points)[0] # num_points x 3
        # data.append(samples)
        # labels.append(cat2lab[root.split('/')[-2]])

    data = np.array(data)# np.array: num_images x num_points x 3

    # zero-center
    data -= np.mean(data, axis=1, keepdims=True)

    # normalize
    data /= np.max(np.linalg.norm(data, ord=2, axis=2, keepdims=True), axis=1, keepdims=True)

    # # zero-center
    # # data -= (torch.max(data, dim=-2, keepdim=True)[0] + torch.min(data, dim=-2, keepdim=True)[0])/2
    # data -= torch.mean(data, dim=1, keepdim=True)
    #
    # # normalize
    # data /= torch.max(data.norm(p=2, dim=2, keepdim=True), dim=1, keepdim=True)[0]

    return np.transpose(data, axes=(0, 2, 1)), np.array(labels)

# def data_visualization(points):
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     ax.scatter(np.array(points)[0, :], np.array(points)[1, :], np.array(points)[2, :], s=0.1)
#     plt.show()

if __name__ == '__main__':
    random.seed(19260817)
    # torch.manual_seed(19260817)
    np.random.seed(19260817)
    dataset = '../dataset/ModelNet40'
    catset = set([os.path.join(f) for f in os.listdir(dataset) if '.' not in f])
    cat2lab = {c: i for c, i in zip(catset, xrange(len(catset)))}


    since = time.time()
    data_train, labels_train = prepare_data(dataset=dataset, cat2lab=cat2lab, dataset_type='train', num_points=1024)
    data_test, labels_test = prepare_data(dataset=dataset, cat2lab=cat2lab, dataset_type='test', num_points=1024)
    print 'load raw data takes', time.time() - since, 's'

    # since = time.time()
    # np.savez(os.path.join(dataset, 'data_train'), data=data_train, labels=labels_train)
    # np.savez(os.path.join(dataset, 'data_test'), data=data_test, labels=labels_test)
    # print 'save data takes', time.time() - since, 's'

    # torch.save(data_train, os.path.join(dataset, 'data_train.pth'))
    # torch.save(labels_train, os.path.join(dataset, 'labels_train.pth'))
    #
    # torch.save(data_test, os.path.join(dataset, 'data_test.pth'))
    # torch.save(labels_test, os.path.join(dataset, 'labels_test.pth'))

