# -----------------------------------------------------------------------------
# Load pre-processed S3DIS Dataset
# -----------------------------------------------------------------------------

import os
import h5py
import numpy as np
import time

def prepare_data(dataset, area_test = 'Area_6'):
    '''
    :param dataset:
    :param area_test:
    :return: data: np.array, num_images x 9 x num_points
    :return: labels: np.array, num_images x num_points
    '''
    files = [os.path.join(dataset, f) for f in os.listdir(dataset) if '.h5' in f]
    rooms = [line.rstrip() for line in open( os.path.join(dataset, 'room_filelist.txt'))]

    data = np.concatenate([h5py.File(h5)['data'][:] for h5 in files], axis=0)
    labels = np.concatenate([h5py.File(h5)['label'][:] for h5 in files], axis=0)

    idx_train = [i for i, r in enumerate(rooms) if area_test not in r]
    idx_test = [i for i, r in enumerate(rooms) if area_test in r]

    data_train = data[idx_train, ...]
    labels_train = labels[idx_train]

    data_test = data[idx_test, ...]
    labels_test = labels[idx_test]

    return np.transpose(data_train, axes=(0, 2, 1)), np.transpose(data_test, axes=(0, 2, 1)), labels_train, labels_test


if __name__ == '__main__':
    dataset = '../dataset/S3DIS'
    area_test = 'Area_6'
    since = time.time()
    data_train, data_test, labels_train, labels_test = prepare_data(dataset, area_test='Area_6')
    np.savez(os.path.join(dataset, 'data_train'), data=data_train, labels=labels_train)
    np.savez(os.path.join(dataset, 'data_test'), data=data_test, labels=labels_test)
    print(time.time() - since)


