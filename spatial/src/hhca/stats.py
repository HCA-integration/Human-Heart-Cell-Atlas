"""Small statistics helpers."""

from __future__ import annotations

import numpy as np


def stars(p: float) -> str:
    """Significance stars of a p or q value: ``***`` < 0.001, ``**`` < 0.01, ``*`` < 0.05, otherwise ``ns``."""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def robust_norm(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Scale `values` to [0, 1] between `low` and `high` and clip; zeros when the range is empty."""
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)
