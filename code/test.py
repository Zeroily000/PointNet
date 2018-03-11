import numpy as np
import glob
import os

def get_velo_scans(velo_files):
    """Generator to parse velodyne binary files into arrays."""
    for filename in velo_files:
        scan = np.fromfile(filename, dtype=np.float32)
        yield scan.reshape((-1, 4))

def velo():
    """Generator to read velodyne [x,y,z,reflectance] scan data from binary files."""
    # Find all the Velodyne files
    base_path = '../dataset/KITTI'
    date = '2011_09_30'
    drive = date + '_drive_' + '0016' + '_sync'
    data_path = os.path.join(base_path, date, drive)

    velo_path = os.path.join( data_path, 'velodyne_points', 'data', '*.bin')
    velo_files = sorted(glob.glob(velo_path))

    # Subselect the chosen range of frames, if any
    # if frames is not None:
    #     velo_files = [velo_files[i] for i in frames]

    # Return a generator yielding Velodyne scans.
    # Each scan is a Nx4 array of [x,y,z,reflectance]

    for filename in velo_files:
        scan = np.fromfile(filename, dtype=np.float32)
        yield scan.reshape((-1, 4))

    # return get_velo_scans(velo_files)


if __name__ == '__main__':
    # data = velo()
    # print type(data)


    # Find all the Velodyne files
    data_path = '../dataset/KITTI/2011_09_30/2011_09_30_drive_0016_sync'
    velo_path = os.path.join(data_path, 'velodyne_points', 'data', '*.bin')
    velo_files = sorted(glob.glob(velo_path))


    for filename in velo_files:
        scan = np.fromfile(filename, dtype=np.float32).reshape((-1, 4))
        print scan.shape
        # yield scan.reshape((-1, 4))