from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import random



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


def prepare_data(dataset, temp = 320, dataset_type = '', num_points = 2048, reshuffle = True):
    '''
    Load all data
    :param dataset: folder path
    :param batch_size:
    :return: data: BxNx3 list of string; label: BxN
    '''



    # load train file paths
    # paths = [os.path.join(root, filename) for root, _, files in os.walk(dataset, topdown=False) for filename in files if dataset_type in root]
    paths = [(root, filename) for root, _, files in os.walk(dataset, topdown=False) for filename in files if
             dataset_type in root]

    # random shuffle
    if reshuffle:
        random.seed(19260817)
        random.shuffle(paths)

    data, cats = [], []
    for root, filename in paths:
        points = read_off(os.path.join(root, filename), num_points)
        if len(points) != 0:
            data.append(points)
            cats.append(root.split('/')[-2])

        # delete later
        if len(data) >= temp:
            break
        # end
    data = torch.FloatTensor(data)  # num_images x num_points x 3

    # zero-center
    data -= (torch.max(data, dim=-2, keepdim=True)[0] + torch.min(data, dim=-2, keepdim=True)[0])/2

    # normalize
    data /= torch.max(data.norm(p=2, dim=-1, keepdim=True), dim=1, keepdim=True)[0]


    # maps category(string) to label(int)
    catset = set([os.path.join(f) for f in os.listdir(dataset)])
    cat2lab = {c: i for c, i in zip(catset, xrange(len(catset)))}
    labels = torch.LongTensor([cat2lab[c] for c in cats])  # num_images


    # paths_test = [os.path.join(root, filename) for root, _, files in os.walk(dataset, topdown=False) for filename in
    #               files if 'test' in root]
    #
    # data_test, cats_test = [], []
    # for fp in paths_test:
    #     points = read_off(fp, num_points)
    #     if len(points) != 0:
    #         data_test.append(points)
    #         cats_test.append(fp.split('_')[0].split('/')[-1])
    #
    #     # delete later
    #     if len(data_test) >= temp:
    #         break
    #     # end
    # data_test = torch.FloatTensor(data_test)  # num_images x num_points x 3
    #
    # # zero-center
    # data_test -= (torch.max(data_test, dim=-2, keepdim=True)[0] + torch.min(data_test, dim=-2, keepdim=True)[0]) / 2
    #
    # # normalize
    # data_test /= torch.max(data_test.norm(p=2, dim=-1, keepdim=True), dim=1, keepdim=True)[0]

    # data_batch = []
    # data_train, cats_train = [], []
    # for fp in paths_train:
    #     if len(data_batch) < batch_size:
    #         points = read_off(fp, num_points)
    #         if len(points) != 0:
    #             data_batch.append(points)
    #             cats_train.append(fp.split('_')[0].split('/')[-1])
    #     else:
    #         data_train.append(data_batch)
    #         data_batch = []
    #
    #     if batch_num and len(data_train) >= batch_num:
    #         break

    # data_test, cats_test = [], []
    # for fp in paths_test:
    #     if len(data_batch) < batch_size:
    #         points = read_off(fp, num_points)
    #         if len(points) != 0:
    #             data_batch.append(points)
    #             cats_test.append(fp.split('_')[0].split('/')[-1])
    #     else:
    #         data_test.append(data_batch)
    #         data_batch = []
    #
    #     if batch_num and len(data_test) >= batch_num:
    #         break


    # map string to integer
    # cats = set(cats_train+cats_test)


    # labels_test = torch.LongTensor([cat2lab[c] for c in cats_test]) # num_images
    # labels_train = [[cat2lab[c] for c in cats_train[bn * batch_size: bn * batch_size + batch_size]] for bn in xrange(batch_num)]
    # labels_test = [[cat2lab[c] for c in cats_test[bn * batch_size: bn * batch_size + batch_size]] for bn in xrange(batch_num)]


    # return torch.FloatTensor(data_train), torch.FloatTensor(data_test), torch.LongTensor(labels_train), torch.LongTensor(labels_test)
    return data.transpose(-1, -2), labels

def data_visualization(points):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(np.array(points)[0, :], np.array(points)[1, :], np.array(points)[2, :], s=0.1)
    plt.show()

if __name__ == '__main__':
    data_train, labels_train = prepare_data('../dataset/ModelNet40', dataset_type='train')
    data_visualization(np.array(data_train[0][1], dtype=float).T)