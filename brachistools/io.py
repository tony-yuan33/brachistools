
import os, sys

import cv2
import logging
import pathlib
from pathlib import Path
import xml.etree.ElementTree as ET
from natsort import natsorted
from skimage.io import imread, imsave

from .version import version_str

try:
    from qtpy.QtWidgets import QMessageBox
    HAVE_GUI = True
except:
    HAVE_GUI = False

try:
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except:
    HAVE_MATPLOTLIB = False

io_logger = logging.getLogger(__name__)

def logger_setup():
    cp_dir = Path.home().joinpath('.brachistools')
    cp_dir.mkdir(exist_ok=True)
    log_file = cp_dir.joinpath('run.log')
    try:
        log_file.unlink()
    except:
        print("Creating new log file")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"WRITING LOG OUTPUT TO {log_file}")
    logger.info(version_str)

    return logger, log_file

def load_folder(path, file_ext) -> 'list[str]':
    """Get all file names with specified extension(s) under a folder"""
    from typing import Iterable

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

# def xml_to_mask(filename):
#     pass

# def mask_to_contour(mask):
#     pass

# def mask_to_bounding_box(mask):
#     pass

