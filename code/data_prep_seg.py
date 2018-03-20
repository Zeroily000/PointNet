# -----------------------------------------------------------------------------
# Load pre-processed S3DIS Dataset
# -----------------------------------------------------------------------------

import os
import h5py
import numpy as np
import time
import random

def prepare_data(dataset, area_test = 'Area_6'):
    '''

    :param dataset:
    :param area_test:
    :return: data: np.array, num_images x num_points x 9
    :return: labels: np.array, num_images x num_points
    '''
    files = [os.path.join(dataset, f) for f in os.listdir(dataset) if '.h5' in f]
    rooms = [line.rstrip() for line in open( os.path.join(dataset, 'room_filelist.txt'))]
    random.shuffle(files)

    data = np.concatenate([h5py.File(h5)['data'][:] for h5 in files], axis=0)
    labels = np.concatenate([h5py.File(h5)['label'][:] for h5 in files], axis=0)

    # data, labels = [], []
    # for h5 in files:
    #     f = h5py.File(h5)
    #     data.append(f['data'][:])
    #     labels.append(f['label'][:])
    # data = np.concatenate(data, axis=0)
    # labels = np.concatenate(labels, axis=0)

    idx_train = [i for i, r in enumerate(rooms) if area_test not in r]
    idx_test = [i for i, r in enumerate(rooms) if area_test in r]

    data_train = data[idx_train, ...]
    labels_train = labels[idx_train]

    data_test = data[idx_test, ...]
    labels_test = labels[idx_test]

    return data_train, labels_train, data_test, labels_test


if __name__ == '__main__':
    random.seed(19260817)
    np.random.seed(19260817)
    dataset = '../dataset/S3DIS'
    area_test = 'Area_6'
    since = time.time()
    data_train, labels_train, data_test, labels_test = prepare_data(dataset, area_test='Area_6')
    np.savez(os.path.join(dataset, 'data_train'), data=data_train, labels=labels_train)
    np.savez(os.path.join(dataset, 'data_test'), data=data_test, labels=labels_test)
    print(time.time() - since)


