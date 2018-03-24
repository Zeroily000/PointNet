# PointNet
PyTorch Implementation of [PointNet](https://arxiv.org/pdf/1612.00593.pdf).

## Prerequisites
- Python 3.5
- [PyTorch 0.3.1](http://pytorch.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [trimesh](https://github.com/mikedh/trimesh)
- [h5py](https://www.h5py.org/)


## Dataset
Download ModelNet40 and S3DIS datasets:

	sh download_data.sh


## Classification
Train classification network:

    python train_classify.py

Evaluate:

	python eval_classify.py


## Segmentation
Train segmentation network:

    python train_segment.py

Evaluate:

	python eval_segment.py

