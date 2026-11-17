"""Lumen-inclusive calibre of vessel structures, calibre classes and composition along calibre."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

# vote-fraction column of a structure -> label of the subtype in the composition tables
SUBTYPE_LABELS = {
    "frac_VSMC": "VSMC (mural)",
    "frac_Pericyte": "Pericyte (mural)",
    "frac_ArterialEC": "Arterial EC",
    "frac_CapillaryEC": "Capillary EC",
    "frac_VenousEC": "Venous EC",
}
FRACTIONS = list(SUBTYPE_LABELS)
CLASSES = ["large_conductance", "intermediate", "small_resistance"]


def _hull_area(points) -> float:
    """Convex-hull area of 2-D points (0 for fewer than three points or a degenerate hull)."""
    if len(points) < 3:
        return 0.0
    try:
        return float(ConvexHull(points).volume)
    except QhullError:  # collinear points
        return 0.0


def vessel_calibre(vessels, lattice_um=8.0, filled_diameter=True) -> pd.DataFrame:
    """Calibre of every structure of one section.

    The area of a structure is the convex hull of its lattice-cell centres, or the area of the cells themselves
    if that is larger, so a ring of wall cells counts with its lumen. Calibre is the diameter of the circle of
    that area, ``2 * sqrt(area / pi)``.

    Returns
    -------
    One row per structure: ``struct``, ``n_bins``, centroid ``x_um`` / ``y_um``, ``mural_VSMC_frac`` (share of
    cells whose dominant type is VSMC), ``hull_area_um2`` and ``lumen_equiv_diam_um``; with `filled_diameter`
    also ``filled_equiv_diam_um``, the diameter from the cells alone.
    """
    cell_area = lattice_um**2
    rows = []
    for struct, members in vessels.obs[["struct", "x_um", "y_um", "dom_subtype"]].groupby("struct"):
        points = members[["x_um", "y_um"]].values
        n_bins = len(members)
        filled = n_bins * cell_area
        area = max(_hull_area(points), filled)
        row = {
            "struct": int(struct),
            "n_bins": n_bins,
            "x_um": float(points[:, 0].mean()),
            "y_um": float(points[:, 1].mean()),
            "mural_VSMC_frac": float((members["dom_subtype"] == "VSMC").mean()),
        }
        if filled_diameter:
            row["filled_equiv_diam_um"] = 2.0 * np.sqrt(filled / np.pi)
        row.update(hull_area_um2=area, lumen_equiv_diam_um=2.0 * np.sqrt(area / np.pi))
        rows.append(row)
    return pd.DataFrame(rows)


def calibre_classes(structures, d_large, d_small, large_min_vsmc=0.20, small_max_vsmc=0.05) -> pd.DataFrame:
    """Add ``vessel_class`` to a calibre table (in place) and return it.

    ``large_conductance``: diameter >= `d_large` and VSMC share >= `large_min_vsmc`; ``small_resistance``:
    diameter <= `d_small` and VSMC share < `small_max_vsmc`; ``intermediate`` otherwise.
    """
    structures["vessel_class"] = "intermediate"
    large = (structures.mural_VSMC_frac >= large_min_vsmc) & (structures.lumen_equiv_diam_um >= d_large)
    structures.loc[large, "vessel_class"] = "large_conductance"
    small = (structures.mural_VSMC_frac < small_max_vsmc) & (structures.lumen_equiv_diam_um <= d_small)
    structures.loc[small, "vessel_class"] = "small_resistance"
    return structures


def structure_composition(calibre, votes) -> pd.DataFrame:
    """Join the calibre of every structure to its composition, the share of its clean votes per subtype.

    Structures that lose every vascular pixel to the spillover mask get no composition (NaN).
    """
    subtypes = [fraction[len("frac_") :] for fraction in FRACTIONS]
    vote_columns = [f"vote_{subtype}" for subtype in subtypes]
    clean = votes[["slide", "struct", *vote_columns]].copy()
    kept = clean[vote_columns].sum(axis=1).values
    for subtype in subtypes:
        clean[f"frac_{subtype}"] = np.where(kept > 0, clean[f"vote_{subtype}"].values / np.maximum(kept, 1), np.nan)
    return calibre[["slide", "struct", "n_bins", "lumen_equiv_diam_um", "vessel_class"]].merge(
        clean[["slide", "struct", *FRACTIONS]], on=["slide", "struct"], how="left", validate="1:1"
    )


def composition_by_calibre(composition, by, edges, min_structures=20, name=None) -> pd.DataFrame:
    """Mean composition per bin of a size measure.

    Parameters
    ----------
    composition
        Structures with their vote fractions (:func:`structure_composition`).
    by
        Column to bin, e.g. ``"lumen_equiv_diam_um"``.
    edges
        Bin edges.
    min_structures
        Structures needed for a bin to count as ``enough``.
    name
        Name of the measure in the column names of the result (default: `by`).

    Returns
    -------
    One row per non-empty bin: edges, median of `by`, ``n``, ``enough``, and per subtype the mean share
    (``<subtype>__pct``) with the half-width of its 95 % confidence interval (``<subtype>__ci95_pct``).
    """
    name = name or by
    x = composition[by].values.astype(float)
    bin_index = np.digitize(x, edges) - 1
    records = []
    for b in range(len(edges) - 1):
        in_bin = bin_index == b
        n = int(in_bin.sum())
        if n == 0:
            continue
        record = {
            f"{name}_lo": float(edges[b]),
            f"{name}_hi": float(edges[b + 1]),
            f"median_{name}": float(np.median(x[in_bin])),
            "n": n,
            "enough": n >= min_structures,
        }
        for fraction, label in SUBTYPE_LABELS.items():
            values = composition[fraction].values[in_bin]
            values = values[np.isfinite(values)]  # structures without a composition
            k = len(values)
            sem = float(values.std(ddof=1) / np.sqrt(k)) if k > 1 else np.nan
            record[f"{label}__pct"] = 100 * float(values.mean())
            record[f"{label}__ci95_pct"] = 100 * 1.96 * sem if k > 1 else np.nan
        record["n_with_composition"] = int(np.isfinite(composition[FRACTIONS[0]].values[in_bin]).sum())
        records.append(record)
    bins = pd.DataFrame(records)
    value_columns = [column for column in bins.columns if column.endswith("_pct")]
    bins[value_columns] = bins[value_columns].round(3)
    return bins


def composition_by_class(composition) -> pd.DataFrame:
    """Mean composition (%), structure count and median diameter per calibre class."""
    by_class = (composition.groupby("vessel_class")[FRACTIONS].mean() * 100).round(2)
    by_class["n_structures"] = composition.vessel_class.value_counts()
    by_class["median_diam_um"] = composition.groupby("vessel_class").lumen_equiv_diam_um.median().round(2)
    return by_class.reindex(CLASSES)
