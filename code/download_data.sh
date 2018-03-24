wget http://modelnet.cs.princeton.edu/ModelNet40.zip
wget https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip

mkdir ../dataset/
mkdir ../results/
mkdir ../results/classification/
mkdir ../results/segmentation/

mv ModelNet40.zip ../dataset/
mv indoor3d_sem_seg_hdf5_data.zip ../dataset/
cd ../dataset/

unzip ModelNet40.zip
rm ModelNet40.zip

unzip indoor3d_sem_seg_hdf5_data.zip
rm indoor3d_sem_seg_hdf5_data.zip
mv indoor3d_sem_seg_hdf5_data S3DIS

