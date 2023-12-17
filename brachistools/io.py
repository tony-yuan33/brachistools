
import os, sys

import cv2
import logging
import pathlib
from pathlib import Path
import xml.etree.ElementTree as ET
from natsort import natsorted
from skimage.io import imread, imsave
import numpy as np

from .version import version_str
from .transforms import get_hue_colors

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

def load_folder(path, file_ext, absolute_path = False) -> 'list[str]':
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
                    if absolute_path:
                        yield os.path.join(path, file)
                    else:
                        yield file

    return natsorted(get_results())

def labels_to_xml(labels, bg_label=0) -> ET.ElementTree:
    from skimage.measure import find_contours
    root = ET.Element("Annotation", Width=str(labels.shape[1]), Height=str(labels.shape[0]))
    regions_elem = ET.SubElement(root, "Regions")

    label_names = set(labels.flat)
    label_names.remove(bg_label)

    for label in label_names:
        region_mask = (labels == label)

        # Find contour points to write as a polygon
        contours = find_contours(region_mask, level=0.5, fully_connected='low')
        if len(contours) == 0:
            continue

        region_elem = ET.SubElement(regions_elem, "Region", Label=str(label))
        for i, contour in enumerate(contours):
            (rmin, cmin), (rmax, cmax) = np.min(contour, axis=0), np.max(contour, axis=0)

            if i == 0:
                contour_id = str(label)
            else:
                contour_id = f"{label}_{i}"

            vertices_elem = ET.SubElement(region_elem, "Vertices", Id=contour_id)
            for y, x in contour:
                ET.SubElement(vertices_elem, "Vertex", X=str(x), Y=str(y))

            bbox_elem = ET.SubElement(
                region_elem, "BoundingBox",
                X=str(cmin), Y=str(rmin),
                Width=str(cmax-cmin), Height=str(rmax-rmin))

    tree = ET.ElementTree(root)
    return tree

def xml_to_pic(tree: ET.ElementTree, use_tqdm=False):
    from tqdm import tqdm
    from skimage.draw import polygon, polygon2mask

    root = tree.getroot()
    width, height = int(root.attrib['Width']), int(root.attrib['Height'])

    regions = root.findall('Regions/Region')
    im = np.zeros(shape=(height, width, 3)) # Prepare RGB
    colors = get_hue_colors(len(regions))
    np.random.shuffle(colors)

    loopvar = zip(regions, colors)
    if use_tqdm:
        loopvar = tqdm(loopvar, desc="Drawing regions", total=len(regions))
    for region, color in loopvar:
        rows = []
        cols = []
        for point in region.findall('Vertices/Vertex'):
            rows.append(float(point.attrib['Y']))
            cols.append(float(point.attrib['X']))

        ridx, cidx = polygon(rows, cols, shape=(height, width))
        # Draw polygon with the color
        im[ridx, cidx, :] = color

        # Draw bbox
        bbox = region.find('BoundingBox')
        min_y, min_x = float(bbox.attrib['Y']), float(bbox.attrib['X'])
        y_ext, x_ext = float(bbox.attrib['Height']), float(bbox.attrib['Width'])
        min_y, min_x, y_ext, x_ext = int(min_y), int(min_x), int(y_ext), int(x_ext)
        max_y, max_x = min_y + y_ext + 1, min_x + x_ext + 1
        im[min_y:max_y, min_x, :] = color
        im[min_y:max_y, max_x-1, :] = color
        im[min_y, min_x:max_x, :] = color
        im[max_y-1, min_x:max_x, :] = color

    return (im * 255).astype(np.uint8)

# def xml_to_mask(filename):
#     pass

# def mask_to_contour(mask):
#     pass

# def mask_to_bounding_box(mask):
#     pass

