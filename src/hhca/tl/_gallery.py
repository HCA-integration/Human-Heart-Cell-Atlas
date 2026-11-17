"""Locating an H&E tile on the registered microscopy image of a section (Fig 1 j)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def to_grey(rgb):
    """Luma of a float RGB image."""
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def section_grey(image, um_to_px, extent, um_per_px=1.0, strip_um=400.0) -> np.ndarray:
    """Grey H&E of a whole section in the um frame of the bins (white outside the image).

    Parameters
    ----------
    image
        Registered H&E image (rows x columns x channels); a memory-mapped image can be passed.
    um_to_px
        (matrix, offset) of the affine map from um to H&E pixels.
    extent
        (x0, x1, y0, y1) in um.
    um_per_px
        Resolution of the result.
    strip_um
        Height of the strips in which the image is resampled.
    """
    matrix, offset = um_to_px
    x0, x1, y0, y1 = extent
    height, width = image.shape[:2]
    grid_x = np.arange(x0, x1, um_per_px)
    strips = []
    for strip_y0 in np.arange(y0, y1, strip_um):
        grid_y = np.arange(strip_y0, min(strip_y0 + strip_um, y1), um_per_px)
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
        pixels = np.column_stack([mesh_x.ravel(), mesh_y.ravel()]) @ matrix.T + offset
        px, py = pixels[:, 0], pixels[:, 1]
        covered = (px >= 0) & (px <= width - 1) & (py >= 0) & (py <= height - 1)
        strip = np.full(px.size, 255.0, np.float32)
        if covered.any():
            bx0, bx1 = max(0, int(px[covered].min()) - 2), min(width, int(px[covered].max()) + 3)
            by0, by1 = max(0, int(py[covered].min()) - 2), min(height, int(py[covered].max()) + 3)
            block = np.asarray(image[by0:by1, bx0:bx1, :3]).astype(np.float32)
            strip[covered] = map_coordinates(
                to_grey(block), [py[covered] - by0, px[covered] - bx0], order=1, mode="nearest"
            )
        strips.append(strip.reshape(len(grid_y), len(grid_x)))
    return np.vstack(strips).astype(np.float32)


def locate_tile(section, tile, size_um, um_per_px=1.0) -> dict:
    """Find a square grey tile on the grey image of a section by normalised cross-correlation.

    The tile is tried at 3 scales and 8 orientations (needs OpenCV).

    Returns
    -------
    Dictionary with the best score ``ncc``, the ``runner_up`` score outside a tile-sized radius, the position
    (``column``, ``row``) of the match, the tile side ``n`` in pixels and its ``scale``, ``rot90`` and ``flip``.
    """
    import cv2

    best = None
    for scale in (0.9, 1.0, 1.1):
        n = int(round(size_um * scale / um_per_px))
        base = cv2.resize(tile, (n, n), interpolation=cv2.INTER_AREA)
        for k in range(4):
            for flip in (False, True):
                template = np.rot90(base, k)
                template = np.ascontiguousarray(template[:, ::-1] if flip else template)
                scores = cv2.matchTemplate(section, template, cv2.TM_CCOEFF_NORMED)
                _, score, _, location = cv2.minMaxLoc(scores)
                if best is None or score > best[0]:
                    best = (score, location, scale, k, flip, n, scores)
    score, (column, row), scale, k, flip, n, scores = best
    radius = int(size_um / um_per_px)
    scores = scores.copy()
    scores[max(0, row - radius) : row + radius, max(0, column - radius) : column + radius] = -1
    return {
        "ncc": float(score),
        "runner_up": float(scores.max()),
        "column": column,
        "row": row,
        "n": n,
        "scale": scale,
        "rot90": k,
        "flip": flip,
    }
