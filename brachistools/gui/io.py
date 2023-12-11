
import os, gc
from natsort import natsorted
import numpy as np


def load_folder(path, file_ext = "") -> 'list[str]':
    """Get all file names with specified extension under a folder"""
    files = filter(lambda fn: fn.endswith(file_ext), os.listdir(path))
    return natsorted(files)

