"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.

Adapted by: YUAN Ruihong
"""

from importlib.metadata import PackageNotFoundError, version
import sys
from platform import python_version
import torch

try:
    vers = version("brachistools")
except PackageNotFoundError:
    vers = 'unknown'

version_str = f"""
brachistools version:\t{vers}
platform:           \t{sys.platform}
python version:     \t{python_version()}
torch version:      \t{torch.__version__}"""
