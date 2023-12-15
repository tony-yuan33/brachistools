
import os, gc
from typing import Iterable

from natsort import natsorted
import numpy as np

from brachistools.io import load_folder, imread, imsave


def abbrev_path(path, max_char_length=30):
    if len(path) <= max_char_length:
        return path

    dirs = path.split('/')
    dirs2 = path.split('\\')
    if len(dirs) < 2 and len(dirs2) >= 2:
        dirs = dirs2

    if len(dirs) == 1: # A file name
        clip_len = (max_char_length - 7) // 2
        if clip_len <= 0:
            return path
        return path[:clip_len] + '...' + path[-clip_len-4:]

    result = dirs[-1]
    for dir in reversed(dirs[1:]):
        if len(result) + len(dir) + 4 <= max_char_length:
            result = dir + os.path.sep + result
        else:
            result = dirs[0] + os.path.sep + '...' + os.path.sep + result
            break

    return result
