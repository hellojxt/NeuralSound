# NeuralSound: Learning-based Modal Sound Synthesis with Acoustic Transfer
[arXiv](https://arxiv.org/abs/2108.07425) | [Project Page](https://hellojxt.github.io/NeuralSound/)
![teaser](https://hellojxt.github.io/NeuralSound/images/teaser.png)

## Introduction
Official Implementation of NeuralSound. NeuralSound includes a mixed vibration solver for modal analysis and a radiation network for acoustic transfer. This repository also include some code for the implementation of the [DeepModal](https://hellojxt.github.io/DeepModal/).

## Environment
- Ubuntu, Python
- Matplotlib, Numpy, Scipy, PyTorch, tensorboard, PyCUDA, Numba, tqdm, meshio 
- [Bempp-cl](https://github.com/bempp/bempp-cl)
- [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine)
- [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter)

For Ubuntu20.04, python 3.7, and CUDA 11.1, a tested script for environment setup is [here](./environment.md).

## Dataset
The dataset of meshes can be download from [ABC Dataset](https://deep-geometry.github.io/abc-dataset/). Assuming the meshes in obj format are in the dataset folder (`dataset/mesh/*.obj`). We provide scripts to generate synthetic data from the meshes for training and testing:

First, you should cd to the folder of the scripts.
```bash
cd dataset_scripts
```
To generate voxelized data and save to ```dataset/voxel/*.npy```, run:
```bash
python voxelize.py "../dataset/mesh/*" "../dataset/voxel"
```
To generate eigenvectors and eigenvalues from modal analysis and save to ```dataset/eigen/*.npz```, run:
```bash
python modalAnalysis.py "../dataset/voxel/*" "../dataset/eigen"
```
To generate dataset for our vibration solver and save to ```dataset/lobpcg/*.npz```, run:
```bash
python lobpcgMatrix.py "../dataset/voxel/*" "../dataset/lobpcg"
```
To generate dataset for our radiation solver and save to ```dataset/acousticMap/*.npz```, run:
```bash
python acousticTransfer.py "../dataset/eigen/*" "../dataset/acousticMap"
```
To generate dataset for [DeepModal](https://hellojxt.github.io/DeepModal/) and save to ```dataset/deepmodal/*.npz```, run:
```bash
python deepModal.py "../dataset/eigen/*.npz" "../dataset/deepmodal"
```

## Training Vibration Solver
First you should split the dataset into training, testing, and validation sets.
```bash
cd dataset_scripts
python splitDataset.py "../dataset/lobpcg/*.pt"
```
Then you can train the vibration solver by running:
```bash
cd ../vibration
python train.py --dataset "../dataset/lobpcg" --tag default_tag --net defaultUnet --cuda 0
```
The log file is saved to ```vibration/runs/default_tag/``` and the weights are saved to ```vibration/weights/default_tag.pt```.

In the vibration solver, the matrix multiplication is implemented by a similar way of graph neural network. See function ```spmm_conv``` in ```src/classic/fem/project_util.py```. This is because of the consistency of convolution with matrix multiplication mentioned in our paper.

## Training Radiation Solver
First you should split the dataset into training, testing, and validation sets.
```bash
cd dataset_scripts
python splitDataset.py "../dataset/acousticMap/*.npz"
```
Then you can train the radiation solver by running:
```bash
cd ../acoustic
python train.py --dataset "../dataset/acousticMap" --tag default_tag --cuda 0
```
The log file is saved to ```acoustic/runs/default_tag/``` and the weights are saved to ```acoustic/weights/default_tag.pt```. Visualized FFAT Maps are saved to ```acoustic/images/default_tag/``` (In each image, above is ground-truth and below is prediction).

The network architecture is defined in ```src/net/acousticnet.py``` and is more compact than the original architecture in our paper while achieves similar performance.

## Training DeepModal
First you should split the dataset into training, testing, and validation sets.
```bash
cd dataset_scripts
python splitDataset.py "../dataset/deepmodal/*.pt"
```
Then you can train the deepmodal by running:
```bash
cd ../deepmodal
python train.py --dataset "../dataset/deepmodal" --tag default_tag --net defaultUnet --cuda 0
```
The log file is saved to ```deepmodal/runs/default_tag/``` and the weights are saved to ```deepmodal/weights/default_tag.pt```. DeepModal is faster than the mixed vibration solver in NeuralSound, but the error is larger. This implementation of DeepModal use Sparse-Convolutional Neural Network (SCNN) rather than Convolutional Neural Network (CNN) in the original paper.

## Contact
For bugs and feature requests please visit GitHub Issues or contact [Xutong Jin](https://hellojxt.github.io/) by email at jinxutong@pku.edu.cn.