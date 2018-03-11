from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import random



def read_off(fpath, thresh = 2048):
    '''
    Read point cloud from a single image file
    :param filename: filename dir
    :return: points Nx3 list of string
    '''
    f = open(fpath, 'r')
    if 'OFF' != f.readline().strip():
        f.seek(3)

    N = int(f.readline().strip().split(' ')[0])

    if N < thresh:
        return []
    # thresh = N
    points = [[float(x) for x in f.readline().strip().split(' ')] for i in xrange(0, N/thresh*thresh, N/thresh)]
    # points = [[float(x) for x in f.readline().strip().split(' ')] for i in xrange(N)]
    f.close()
    return points


def prepare_data(dataset, batch_size = 32):
    '''
    Load all data
    :param dataset: folder path
    :param batch_size:
    :return: data: BxNx3 list of string; label: BxN
    '''
    random.seed(19260817)

    paths_train = [os.path.join(root, filename) for root, _, files in os.walk(dataset, topdown=False) for filename in files if 'train' in root]
    paths_test = [os.path.join(root, filename) for root, _, files in os.walk(dataset, topdown=False) for filename in files if 'test' in root]

    # random shuffle
    random.shuffle(paths_train)


    batch_num = 10 #

    data_batch = []
    data_train, cats_train = [], []
    for fp in paths_train:
        if len(data_batch) < batch_size:
            points = read_off(fp)
            if len(points) != 0:
                data_batch.append(points)
                cats_train.append(fp.split('_')[0].split('/')[-1])
        else:
            data_train.append(data_batch)
            data_batch = []

        if len(data_train) >= batch_num:
            break

    data_test, cats_test = [], []
    for fp in paths_test:
        if len(data_batch) < batch_size:
            points = read_off(fp)
            if len(points) != 0:
                data_batch.append(points)
                cats_test.append(fp.split('_')[0].split('/')[-1])
        else:
            data_test.append(data_batch)
            data_batch = []

        if len(data_test) >= batch_num:
            break


    # map string to integer
    cats = set(cats_train+cats_test)
    cat2lab = {c: i for c, i in zip(cats, xrange(len(cats)))}
    labels_train = [[cat2lab[c] for c in cats_train[bn * batch_size: bn * batch_size + batch_size]] for bn in xrange(batch_num)]
    labels_test = [[cat2lab[c] for c in cats_test[bn * batch_size: bn * batch_size + batch_size]] for bn in xrange(batch_num)]


    # return torch.FloatTensor(data_train), torch.FloatTensor(data_test), torch.LongTensor(labels_train), torch.LongTensor(labels_test)
    return data_train, data_test, labels_train, labels_test

def data_visualization(points):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(np.array(points)[0, :], np.array(points)[1, :], np.array(points)[2, :], s=0.1)
    plt.show()

if __name__ == '__main__':
    data_train, data_test, labels_train, labels_test = prepare_data('../dataset/ModelNet40')
    data_visualization(np.array(data_train[0][1], dtype=float).T)