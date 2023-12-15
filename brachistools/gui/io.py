
import os, gc
from typing import Iterable

from natsort import natsorted
import numpy as np

from ..io import imread, imsave

def load_folder(path, file_ext) -> 'list[str]':
    """Get all file names with specified extension(s) under a folder"""
    if isinstance(file_ext, Iterable):
        file_ext = list(ext.lower() for ext in file_ext)
    else:
        file_ext = [file_ext.lower()]

    files = os.listdir(path)
    def get_results():
        for file in files:
            for ext in file_ext:
                if file.lower().endswith(ext):
                    yield file

    return natsorted(get_results())

