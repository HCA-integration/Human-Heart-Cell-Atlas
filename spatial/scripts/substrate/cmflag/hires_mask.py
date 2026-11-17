"""Tissue mask of an H&E image from colour saturation.

Used by cmflag_ah.py for the cuts of the hires path (median grey of tissue and background). Saturation works on
a grey canvas as well as on white glass, where a plain grey threshold would mark the whole image.
"""

import numpy as np
from scipy import ndimage as ndi


def tissue_mask(rgb, sat_thr=18, bright_hi=235, bright_lo=20):
    """Largest connected region of stained pixels: channel spread above sat_thr, neither very bright nor dark."""
    rgb = np.asarray(rgb).astype(np.float32)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    rgb = rgb[:, :, :3]
    sat = rgb.max(2) - rgb.min(2)
    bright = rgb.mean(2)
    m = (sat > sat_thr) & (bright < bright_hi) & (bright > bright_lo)
    m = ndi.binary_opening(m, iterations=2)
    m = ndi.binary_fill_holes(m)
    lbl, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(np.ones_like(lbl), lbl, range(1, n + 1))
    return ndi.binary_fill_holes(lbl == (np.argmax(sizes) + 1))
