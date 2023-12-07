# Brachistools

Brachistools (BReAst Cancer HIStological image tools) provides tools in segmenting nucleus of and classification of breast cancer histological images.

Brachistools comes in three parts: core function, GUI, and a console entry. You can use its core function part by importing the package in a Jupyter notebook, or use the GUI/console app for a code-light experience.

Please see install instructions [below](README.md/#Installation).

# Installation

## Local installation

### System requirements

Main stream systems (Linux, Windows and macOS) are supposedly supported for running Brachistools. At least 8GB of RAM is required and >=16GB is recommended. The software might have issues on macOS. Please contact Ruihong Yuan via email 3190110636@zju.edu.cn for assistance in installation and/or running.

### Dependencies
Brachistools depends on the following packages:
- [pytorch](https://pytorch.org/)
- [PyQt6](http://pyqt.sourceforge.net/Docs/PyQt6/) or [PyQt5](http://pyqt.sourceforge.net/Docs/PyQt5/)
- [QtPy](https://pypi.org/project/QtPy/)
- [numpy](http://www.numpy.org/) (>=1.20.0)
- [natsort](https://natsort.readthedocs.io/en/master/)
- [scipy](https://www.scipy.org/)

### Instructions

Brachistools uses PyTorch, a deep-learning framework, to implement its algorithms. By default, Brachistools runs on CPU, but you can use a GPU with CUDA to accelerate the computations. See [below](README.md#gpu-version-cuda-on-windows-or-linux) for instructions on configuring your device's CUDA and installing the GPU version of Brachistools.

#### Install via Miniconda
1. Install a [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) distribution of Python.
2. Create a new environment for Brachistools (Python 3.8 is required although later versions might be OK):
```sh
conda create --name YOUR_ENV_NAME 'python>=3.8, <3.9'
```
3. Activate this environment:
```sh
conda activate YOUR_ENV_NAME
```
4. Install the minimal version of Brachistools with
```sh
python -m pip install brachistools
```
5. Or, also install its GUI component with
```sh
python -m pip install brachistools[gui]
```
6. To upgrade `brachistools` (PyPI package [here](https://pypi.org/project/brachistools)), run this:
```sh
python -m pip install brachistools --upgrade
```

Please keep in mind that Brachistools is now installed in this particular conda environment only, so running Brachistools in a Jupyter notebook would additionally require that you run the following command:
```sh
python -m pip install notebook matplotlib
```

#### GPU version (CUDA) on Windows or Linux

If your NVIDIA GPU has CUDA support, you can install GPU version of PyTorch to replace the default CPU version. However, configuring your CUDA device and installing the correct version of `pytorch` can be troublesome. The following is just a brief guide that might have issues on your system.

1. Install the CUDA toolkit (choose one from version 11.x) [here](https://developer.nvidia.com/cuda-toolkit-archive). This is a safe option for configuring your GPU drivers and CUDA.
2. Remove the original CPU version (`torch`) of PyTorch:
```shell
python -m pip uninstall torch
```
3. Find a version of PyTorch that matches your CUDA version [here](https://pytorch.org/get-started/locally/). You might need to search for an older version of PyTorch [here](https://pytorch.org/get-started/previous-versions/). Remember to remove the `torchvision` and `torchaudio` requirements as they are not required by Brachistools. You will end up running a command like this (`conda` is preferred to `pip`)
```shell
conda install pytorch pytorch-cuda=YOUR_CUDA_VERSION -c pytorch -c nvidia
```

After correct installation of `pytorch`, Brachistools will automatically recognize your CUDA device upon being imported.

# Running locally

Get started by opening the GUI from a terminal:
```shell
python -m cellpose --gui
```
