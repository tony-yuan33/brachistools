

def merge_small_labels(labelled_mask, min_size, verbose = True):
    """Merge labels less than threshold size
    Small labels are merged to their largest neighbor in order to
    discourage over-segmentation.
    Label sizes are dynamically monitored to prevent labels
    that become sufficiently large from merging from unnecessary
    merging

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
