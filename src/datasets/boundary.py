"""Boundary labels derived from full-resolution expert masks before patch crops."""

import numpy as np
from scipy import ndimage


def boundary_from_mask(mask, radius=1):
    """Morphological gradient with an inner/outer band and replicated scan edges.

    Input and output have shape (1,D,H,W). Radius is measured in voxels of the
    preprocessed 1 mm grid. Full-scan labels avoid artificial patch-cut edges.
    """
    if not isinstance(radius, int) or isinstance(radius, bool) or radius < 1:
        raise ValueError("boundary radius must be a positive integer")
    mask = np.asarray(mask)
    if mask.ndim != 4 or mask.shape[0] != 1 or not np.isin(mask, [0, 1]).all():
        raise ValueError("Boundary labels require a binary mask (1,D,H,W)")
    binary = mask[0].astype(np.uint8)
    size = 2 * radius + 1
    dilated = ndimage.maximum_filter(binary, size=size, mode="nearest")
    eroded = ndimage.minimum_filter(binary, size=size, mode="nearest")
    return (dilated != eroded)[None].astype(np.uint8)
