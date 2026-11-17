"""File names and readers shared by the certainty and threshold scripts.

Import after the thread variables are set: this module imports numpy.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DECODE_NAME = "atlas_L3.decode.prj_12.r_4.0_5.0.pixel.tsv.gz"


def load_factor_order(atlas_dir: Path):
    """Level-3 type names in factor order."""
    return json.loads((atlas_dir / "atlas_L3_factor_order.json").read_text(encoding="utf-8"))


def load_model(atlas_dir: Path, factor_order):
    """The model matrix (genes x types), columns in factor order."""
    model = pd.read_csv(atlas_dir / "atlas_L3_pseudobulk.tsv", sep="\t", index_col=0)
    missing = [name for name in factor_order if name not in model.columns]
    if missing:
        raise ValueError(f"Missing factors in pseudobulk: {missing}")
    return model[factor_order]


def checkpoint_slides(checkpoint_dir: Path):
    """Sorted ids of the slides with a feature checkpoint."""
    return sorted(path.name.replace(".features.npz", "") for path in checkpoint_dir.glob("*.features.npz"))


def read_decode(path, columns, dtype):
    """The given columns of a pixel decode, with float32 as the default dtype."""
    dtypes = {column: dtype.get(column, np.float32) for column in columns}
    return pd.read_csv(path, sep="\t", usecols=list(columns), dtype=dtypes)
