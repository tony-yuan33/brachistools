"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.

Adapted by: YUAN Ruihong
"""

from importlib.metadata import PackageNotFoundError, version
import sys
from platform import python_version
import tensorflow

try:
    brachistools_version = version("brachistools")
except PackageNotFoundError:
    brachistools_version = 'unknown'

version_str = f"""
brachistools version:\t{brachistools_version}
platform:           \t{sys.platform}
python version:     \t{python_version()}
tensorflow version:      \t{tensorflow.__version__}"""
