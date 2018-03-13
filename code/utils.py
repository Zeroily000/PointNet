from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import random
import time

def read_off(fpath, num_points):
    '''
    Read point cloud from a single image file
    :param filename: filename dir
    :return: points Nx3 list of string
    '''
    f = open(fpath, 'r')
    if 'OFF' != f.readline().strip():
        f.seek(3)

    N = int(f.readline().strip().split(' ')[0])

    if N < num_points:
        return []
    points = [[float(x) for x in f.readline().strip().split(' ')] for i in xrange(0, N/num_points*num_points, N/num_points)]
    f.close()
    return points

#
# def rot_mat(num_images):
#     roll = torch.Tensor(num_images, 1, 1).uniform_(0, math.pi)
#     Rx = torch.cat((torch.cat((torch.ones(num_images, 1, 1), torch.zeros(num_images, 1, 1), torch.zeros(num_images, 1, 1)), 2),
#                     torch.cat((torch.zeros(num_images, 1, 1), torch.cos(roll), -torch.sin(roll)), 2),
#                     torch.cat((torch.zeros(num_images, 1, 1), torch.sin(roll), torch.cos(roll)), 2)), 1)
#
#     pitch = torch.Tensor(num_images, 1, 1).uniform_(0, math.pi)
#     Ry = torch.cat((torch.cat((torch.cos(pitch), torch.zeros(num_images, 1, 1), torch.sin(pitch)), 2),
#                     torch.cat((torch.zeros(num_images, 1, 1), torch.ones(num_images, 1, 1), torch.zeros(num_images, 1, 1)), 2),
#                     torch.cat((-torch.sin(pitch), torch.zeros(num_images, 1, 1), torch.cos(pitch)), 2)), 1)
#
#     yaw = torch.Tensor(num_images, 1, 1).uniform_(0, math.pi)
#     Rz = torch.cat((torch.cat((torch.cos(yaw), -torch.sin(yaw), torch.zeros(num_images, 1, 1)), 2),
#                     torch.cat((torch.sin(yaw), torch.cos(yaw), torch.zeros(num_images, 1, 1)), 2),
#                     torch.cat((torch.zeros(num_images, 1, 1), torch.zeros(num_images, 1, 1), torch.ones(num_images, 1, 1)), 2)), 1)
#
#     return torch.bmm(Rx, torch.bmm(Ry, Rz))
#


def prepare_data(dataset, cat2lab, dataset_type = '', num_points = 1024, reshuffle = True):
    '''
    Load all data
    :param dataset: folder path
    :param batch_size:
    :return: data: BxNx3 list of string; label: BxN
    '''

    # load train file paths
    paths = [(root, filename) for root, _, files in os.walk(dataset, topdown=False) for filename in files if
             dataset_type in root]

    # random shuffle
    if reshuffle:
        random.shuffle(paths)

    data, labels = [], []
    for root, filename in paths:
        points = read_off(os.path.join(root, filename), num_points)
        if len(points) != 0:
            data.append(points)
            labels.append(cat2lab[root.split('/')[-2]])

    data = torch.FloatTensor(data)  # num_images x num_points x 3

    # zero-center
    data -= (torch.max(data, dim=-2, keepdim=True)[0] + torch.min(data, dim=-2, keepdim=True)[0])/2

    # normalize
    data /= torch.max(data.norm(p=2, dim=-1, keepdim=True), dim=1, keepdim=True)[0]

    return data.transpose(-1, -2), torch.LongTensor(labels)

def data_visualization(points):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(np.array(points)[0, :], np.array(points)[1, :], np.array(points)[2, :], s=0.1)
    plt.show()

if __name__ == '__main__':
    random.seed(19260817)
    torch.manual_seed(19260817)

    dataset = '../dataset/ModelNet40'
    catset = set([os.path.join(f) for f in os.listdir(dataset)])
    cat2lab = {c: i for c, i in zip(catset, xrange(len(catset)))}


    since = time.time()
    data_train, labels_train = prepare_data(dataset=dataset, cat2lab=cat2lab, dataset_type='train', num_points=1024, reshuffle=True)
    data_test, labels_test = prepare_data(dataset=dataset, cat2lab=cat2lab, dataset_type='test', num_points=1024, reshuffle=True)
    print 'load raw data takes', time.time() - since, 's'

    since = time.time()
    torch.save(data_train, os.path.join(dataset, 'data_train.pth'))
    torch.save(labels_train, os.path.join(dataset, 'labels_train.pth'))

    torch.save(data_test, os.path.join(dataset, 'data_test.pth'))
    torch.save(labels_test, os.path.join(dataset, 'labels_test.pth'))
    print 'save data takes', time.time() - since, 's'


