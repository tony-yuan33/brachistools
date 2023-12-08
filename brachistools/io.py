
import cv2
import logging
import pathlib
from pathlib import Path
import xml.etree.ElementTree as ET

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
    cp_dir = Path.home().joinpath('.')

def xml_to_mask(filename):
    pass

def mask_to_contour(mask):
    pass

def mask_to_bounding_box(mask):
    pass

