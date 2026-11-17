"""Small helpers for reading Space Ranger outputs, shared by the deconvolution scripts.

Usage: from visium_io import dense, find_positions
"""

import os

import numpy as np
from scipy.sparse import issparse


def dense(X):
    """A sparse or dense matrix as a dense array."""
    return X.toarray() if issparse(X) else np.asarray(X)


def find_positions(spatial_dir):
    """Path of the tissue-positions file of a Space Ranger spatial/ folder (either file name), or None."""
    for file_name in ["tissue_positions.csv", "tissue_positions_list.csv"]:
        path = os.path.join(spatial_dir, file_name)
        if os.path.exists(path):
            return path
    return None
