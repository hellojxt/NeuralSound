# NeuralSound: Learning-based Modal Sound Synthesis with Acoustic Transfer
[arXiv](https://arxiv.org/abs/2108.07425) | [Project Page](https://hellojxt.github.io/NeuralSound/)
![teaser](https://hellojxt.github.io/NeuralSound/images/teaser.png)

## Introduction
Official Implementation of NeuralSound. NeuralSound includes a mixed vibration solver for modal analysis and a radiation network for acoustic transfer. We also inclue some code for the implementation of the [DeepModal](https://hellojxt.github.io/DeepModal/).

## Environment
- Ubuntu, Python
- Matplotlib, Numpy, Scipy, PyTorch, tensorboard, PyCUDA, Numba, tqdm, meshio 
- [Bempp-cl](https://github.com/bempp/bempp-cl)
- [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine)

For Ubuntu20.04, python 3.7, and CUDA 11.1, a tested script for environment setup is [here](./environment.md).

## Dataset
The dataset of meshes can be download from [ABC Dataset](https://deep-geometry.github.io/abc-dataset/). Assuming the meshes in obj format are in the dataset folder (`dataset/mesh/*.obj`). We provide scripts to generate synthetic data from the meshes for training and testing:

First, you should cd to the folder of the scripts.
```bash
cd dataset_scripts
```

To generate Voxelized Data and save to dataset/voxel/*.npy:
```bash
python voxelize.py ../dataset/mesh/*
```
To generate eigenvectors and eigenvalues from modal analysis and save to dataset/eigen/*.npz:
```bash
python modalAnalysis ../dataset/voxel/*
```
To generate dataset for our radiation solver and save to ```dataset/acousticMap/*.npz```:
```bash
python acousticTransfer.py ../dataset/eigen/*
```

To save matrix of each object to dataset/lobpcg/*.npz (the dataset for our vibration solver)
python lobpcgMatrix.py ../dataset/voxel/*

## Training Vibration Solver
First you
## Training Radiation Solver

## Contact
For bugs and feature requests please visit GitHub Issues or contact [Xutong Jin](https://hellojxt.github.io/) by email at jinxutong@pku.edu.cn.