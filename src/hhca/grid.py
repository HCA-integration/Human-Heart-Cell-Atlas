"""Integer keys of the Visium HD bin grids and the CMFLAG spillover mask.

Bins are addressed by one integer per grid cell, ``round(x / size) * KEYMUL + round(y / size)`` with the
coordinates in micrometres. The 2 um grid is the unit of the FICTURE decode and of the CMFLAG flag tables; the
8 um grid is the lattice of the vessel structures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEYMUL = np.int64(10_000_000)
SPILLOVER_FLAGS = ["flag_cm_spillover", "flag_vsmc_spillover", "flag_spill_artEC", "flag_spill_venEC"]


def grid_index(x, y, size: float = 2.0):
    """Grid column, grid row and combined integer key of each coordinate (um) on a `size` um grid."""
    column = np.round(np.asarray(x, float) / size).astype(np.int64)
    row = np.round(np.asarray(y, float) / size).astype(np.int64)
    return column, row, column * KEYMUL + row


def bin_key(x, y, size: float = 2.0) -> np.ndarray:
    """Integer key of the `size` um grid cell of each coordinate (um)."""
    return grid_index(x, y, size)[2]


def spillover_keys(flags_path, size: float = 2.0) -> np.ndarray:
    """Sorted keys of the 2 um bins that carry any of the four CMFLAG spillover flags."""
    flags = pd.read_parquet(flags_path, columns=["X", "Y", *SPILLOVER_FLAGS])
    keys = bin_key(flags["X"].to_numpy(np.float64), flags["Y"].to_numpy(np.float64), size)
    return np.unique(keys[flags[SPILLOVER_FLAGS].to_numpy().any(axis=1)])
