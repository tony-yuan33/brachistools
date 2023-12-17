
from skimage.exposure import equalize_adapthist
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import numpy as np

from .transforms import VahadaneStainDeconvolution, inverted_gray_scale
from .utils import ParamDict

def vahadane(sparsity_regularizer):
    vahadane_transform = VahadaneStainDeconvolution(sparsity_regularizer=sparsity_regularizer)
    def deconvolute_hematoxylin(image):
        return vahadane_transform.F(image=image, method='hematoxylin')
    return deconvolute_hematoxylin

default_segmentation_params = ParamDict({
    'vahadane:sparsity_regularizer': 0.75,
    'equalize_adapthist:clip_limit': 0.01,
    'remove_small_objects:min_size': 250,
    'remove_small_holes:area_threshold': 100,
    'peak_local_max': {
        'min_distance': 12,
        'footprint': np.ones((15, 15)),
        'threshold_rel': 0.2
    },
    'merge_small_labels': {
        'min_size': 300,
        'verbose': False
    }
})

def segmentation_pipeline(input_image, params):
    from skimage import img_as_ubyte

    vahadane_transform = vahadane(**params['vahadane'])

    if len(input_image.shape) < 3 or input_image.shape[-1] < 3:
        raise ValueError("Input image has less than 3 channels")

    input_image = img_as_ubyte(input_image)
    image_H = vahadane_transform(input_image)
    image_H = equalize_adapthist(image_H, **params['equalize_adapthist'])
    image_H = img_as_ubyte(input_image)
    image_H = inverted_gray_scale(image_H)
    nuclei = image_H > threshold_otsu(image_H)
    nuclei = remove_small_objects(nuclei, **params['remove_small_objects'])
    nuclei = remove_small_holes(nuclei, **params['remove_small_holes'])

    distances = distance_transform_edt(nuclei)
    local_maxima_idx = peak_local_max(distances, **params['peak_local_max'])
    markers = peaks_to_markers(local_maxima_idx, shape=nuclei.shape)
    labeled_nuclei = watershed(-distances, markers, mask=nuclei)
    labeled_nuclei = merge_small_labels(labeled_nuclei, **params['merge_small_labels'])
    return nuclei, labeled_nuclei

def peaks_to_markers(peaks, shape):
    from skimage.measure import label
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(peaks.T)] = True
    markers = label(mask)
    return markers

def merge_small_labels(labelled_mask, min_size, verbose = True):
    """Merge labels less than threshold size
    Small labels are merged to their largest neighbor in order to
    discourage over-segmentation.
    Label sizes are dynamically monitored to prevent labels
    that become sufficiently large during merges from unnecessary
    merges

    Author: Ruihong Yuan
    """
    from skimage.measure import regionprops
    from skimage.segmentation import find_boundaries
    from skimage.morphology import dilation, square

    labels = labelled_mask.copy()

    label_areas = []
    small_labels = []
    for region in regionprops(labels, cache=False):
        area = region.area
        label_areas.append(area)
        if area < min_size:
            small_labels.append(region.label)

    merged_count = 0
    skipped_count = 0
    for label in small_labels:
        assert label_areas[label-1] != 0, "Label area is zero only when it has been merged"

        if label_areas[label-1] >= min_size:
            skipped_count += 1
            if verbose:
                print("Skipped", label, "because it has reached min_size after merging")
            continue

        # Find neighboring pixels (label boundary)
        label_boundaries = find_boundaries(labels == label)
        # Expand boundary for robustness
        expanded_boundaries = dilation(label_boundaries, square(3))

        # Collect label types of the boundary pixels
        neighbors = set(labels[expanded_boundaries].flat)
        # Remove background and this label
        neighbors.discard(0)
        neighbors.discard(label)

        if neighbors:
            # Find the largest neighbor label
            largest_neighbor = max(neighbors, key=lambda x: label_areas[x - 1])
            if verbose:
                print("Merged", label, "to", largest_neighbor)
            labels[labels == label] = largest_neighbor
            # Update label areas
            label_areas[largest_neighbor - 1] += label_areas[label - 1]
            label_areas[label - 1] = 0
            merged_count += 1

    if verbose:
        print("Total merged:", merged_count)
        print("Total skipped:", skipped_count)
    return labels

def label2rgb_bbox(labeled_image, image=None, **kwargs):
    """Create bounding boxes based on label2rgb"""
    from skimage import img_as_ubyte
    from skimage.measure import label, regionprops
    from skimage.color import label2rgb

    colors = label2rgb(labeled_image, image=image, **kwargs)
    colors = img_as_ubyte(colors)

    im = colors.copy()
    RED = [255, 0, 0]
    for region in regionprops(labeled_image):
        min_y, min_x, max_y, max_x = region.bbox
        im[min_y:max_y, min_x, :] = RED
        im[min_y:max_y, max_x-1, :] = RED
        im[min_y, min_x:max_x, :] = RED
        im[max_y-1, min_x:max_x, :] = RED

    return im
