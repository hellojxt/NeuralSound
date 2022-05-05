The follow script is used to setup the environment for the project for Ubuntu 20.04, python 3.7, and CUDA 11.1.

```bash
# install some common packages
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install scipy matplotlib numba tensorboard tqdm meshio pycuda

# install pytorch_scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl

# install Bempp-cl
pip install plotly pyopencl[pocl] gmsh
git clone https://github.com/bempp/bempp-cl
cd bempp-cl
python setup.py install

# install MinkowskiEngine
sudo apt install libopenblas-dev
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
export CXX=g++-8
export CUDA_HOME=/usr/local/cuda-11.1/
export MAX_JOBS=2; # see https://github.com/NVIDIA/MinkowskiEngine/issues/228
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```