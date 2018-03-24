# -----------------------------------------------------------------------------
# Pre-processing ModelNet40 Dataset
# -----------------------------------------------------------------------------

import numpy as np
import trimesh
import os, time


def prepare_data(dataset, cat2lab, dataset_type, num_points = 1024):
    '''
    Load all data
    :param dataset: directory of the dataset
    :param cat2lab: map category to integer
    :param dataset_type: 'train' or 'test'
    :param num_points: number of points
    :return: data: Bx3xN
    :return: labels: B
    '''
    # load train file paths
    paths = [(root, filename) for root, _, files in os.walk(dataset, topdown=False) for filename in files if
             dataset_type in root and '.off' in filename]

    data, labels = [], []
    for i, (root, filename) in enumerate(paths):
        file_dir = os.path.join(root, filename)
        with open(file_dir, 'r') as f:
            lines = f.readlines()
            f.close()
        if lines[0].strip() != 'OFF':
            lines[0] = lines[0].replace('OFF', '')
            lines = ['OFF\n'] + lines
            with open(file_dir, 'w') as f:
                f.writelines(lines)
                f.close()

        mesh = trimesh.load(file_dir)
        samples = trimesh.sample.sample_surface(mesh, num_points)[0]  # num_points x 3
        data.append(samples)
        labels.append(cat2lab[root.split('/')[-2]])

    data = np.array(data)  # num_images x num_points x 3

    # zero-center
    data -= np.mean(data, axis=1, keepdims=True)

    # normalize
    data /= np.max(np.linalg.norm(data, ord=2, axis=2, keepdims=True), axis=1, keepdims=True)

    return np.transpose(data, axes=(0, 2, 1)), np.array(labels)


if __name__ == '__main__':
    num_points = 1024
    dataset = '../dataset/ModelNet40'
    cats = sorted([os.path.join(f) for f in os.listdir(dataset) if '.' not in f])
    cat2lab = {c: i for i, c in enumerate(cats)}

    since = time.time()
    print('Processing... ', end='')  # takes about 17 minutes
    data_train, labels_train = prepare_data(dataset=dataset, cat2lab=cat2lab, dataset_type='train', num_points=num_points)
    data_test, labels_test = prepare_data(dataset=dataset, cat2lab=cat2lab, dataset_type='test', num_points=num_points)
    print('Done')

    np.savez(os.path.join(dataset, 'data_train_'+str(num_points)), data=data_train, labels=labels_train)
    np.savez(os.path.join(dataset, 'data_test_'+str(num_points)), data=data_test, labels=labels_test)
    time_elapsed = time.time() - since
    print('Preparing data takes {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

