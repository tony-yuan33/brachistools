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
- [tensorflow](https://www.tensorflow.org/)
- [keras](https://keras.io/)
- [PyQt6](http://pyqt.sourceforge.net/Docs/PyQt6/) or [PyQt5](http://pyqt.sourceforge.net/Docs/PyQt5/)
- [QtPy](https://pypi.org/project/QtPy/)
- [numpy](http://www.numpy.org/) (>=1.20.0)
- [natsort](https://natsort.readthedocs.io/en/master/)
- [scipy](https://www.scipy.org/)

### Instructions

Brachistools uses PyTorch, a deep-learning framework, to implement its algorithms. By default, Brachistools runs on CPU, but you can use a GPU with CUDA to accelerate the computations. See [below](README.md#gpu-version-cuda-on-windows-or-linux) for instructions on configuring your device's CUDA and installing the GPU version of Brachistools.

#### `git clone` and install locally

1. Open your terminal. `cd` to a directory for cloning the repo
2. Clone the repo:
```shell
git clone https://github.com/tony-yuan33/brachistools.git
```
3. `cd` into the repo:
```shell
cd brachistools
```
4. Install from source with `pip`:
```shell
python -m pip install .[gui]
```
or if you don't need a GUI:
```shell
python -m pip install .
```
5. Download model parameters from shared OneDrive [link](https://zjuintl-my.sharepoint.com/:u:/g/personal/wenjun_20_intl_zju_edu_cn/EW3hFp7TASBJizvAAHAT0IAB49hWBWKFk_6ZDtLkzhaoUw). Then run the following command to tell Brachistools the location of the downloaded model parameters:
```sh
python -m brachistools config --param_dir FILE_FOLDER_CONTAINING_MODEL_PARAMS
```

#### Install via Miniconda (NOT YET SUPPORTED!)

The following will be supported soon.

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
7. Download model parameters from shared OneDrive [link](TBD). Then run the following command to tell Brachistools the location of the downloaded model parameters:
```sh
python -m brachistools config --param_dir FILE_FOLDER_CONTAINING_MODEL_PARAMS
```

# Running locally

Get started by opening the GUI from a terminal:
```shell
python -m brachistools gui
```

For batch processes, you can directly use the command line version. See help:
```shell
python -m brachistools --help
```

Segment and save numpy arrays for binary mask and labels:
```shell
python -m brachistools segment --dir ... --save_dir ... --save_npy
```

Obtain suggested diagnosis using the deep learning model:
```shell
python -m brachistools classify --dir ... --save_dir ...
```

Convert output XML annotation of instance segmentation with:
```shell
python -m brachistools show --dir ... --save_dir ... --save_npy
```

Configure the directory for deep learning model parameters (default is PACKAGE_PATH/models):
```shell
python -m brachistools config --param_dir ...
```

See help for specific subcommands with:
```shell
python -m brachistools segment --help
```
