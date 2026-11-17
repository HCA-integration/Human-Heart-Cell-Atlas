"""Marker counts per vessel-wall compartment from the Space Ranger 2 um matrix of one section."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from hhca.grid import KEYMUL, bin_key

CAPILLARY = "Capillary Endothelial cells"
PERICYTE = "Pericytes"
MUSCULAR_VEIN = "Venous EC (muscular vein)"
# compartments along the arterio-venous axis, then the specialised endothelia: (name in the tables, label)
COMPARTMENTS = [
    ("Vascular Smooth Muscle cells", "Smooth muscle"),
    ("Arterial Endothelial cells", "Arterial EC"),
    ("Pericytes", "Pericytes"),
    ("Capillary EC (pericyte+)", "Capillary EC,\npericyte-carrying"),
    ("Capillary EC (pericyte-)", "Capillary EC,\nnon-pericyte"),
    ("Venous Endothelial cells", "Venous EC"),
    (MUSCULAR_VEIN, "Venous EC,\nmuscular vein"),
    ("Lymphatic Endothelial Cells", "Lymphatic EC"),
    ("Endocardial Endothelial Cells", "Endocardial EC"),
]
COMPARTMENT_LABELS = dict(COMPARTMENTS)


def _compartment_of_key(pixels, factor_order, vessel_types, spillover, pericyte_radius_um):
    """Vessel-wall compartment of each 2 um key, from the highest-P1 decode pixel of the key."""
    bins = pd.DataFrame(
        {
            "K1": pixels["K1"].values,
            "P1": pixels["P1"].values,
            "key": bin_key(pixels["X"].values, pixels["Y"].values),
        }
    )
    if spillover is not None:
        bins = bins[~np.isin(bins["key"].values, spillover)]
    bins = bins.sort_values("P1", ascending=False).drop_duplicates("key", keep="first")
    bins["type"] = bins["K1"].map(lambda code: factor_order[int(code)])
    bins["x_um"] = (bins["key"].values // KEYMUL).astype(np.float64) * 2.0
    bins["y_um"] = (bins["key"].values % KEYMUL).astype(np.float64) * 2.0
    bins = bins[bins["type"].isin(vessel_types)].copy()

    pericyte_xy = bins.loc[bins["type"] == PERICYTE, ["x_um", "y_um"]].values
    capillary = (bins["type"] == CAPILLARY).values
    capillary_xy = bins.loc[capillary, ["x_um", "y_um"]].values
    if len(pericyte_xy) and len(capillary_xy):
        distance, _ = cKDTree(pericyte_xy).query(capillary_xy, k=1, distance_upper_bound=pericyte_radius_um)
        with_pericyte = np.isfinite(distance)
    else:
        with_pericyte = np.zeros(len(capillary_xy), bool)
    compartment = bins["type"].astype(object).values
    compartment[np.where(capillary)[0]] = np.where(
        with_pericyte, "Capillary EC (pericyte+)", "Capillary EC (pericyte-)"
    )
    key_to_compartment = pd.Series(compartment, index=bins["key"].values)
    key_to_compartment = key_to_compartment[~key_to_compartment.index.duplicated(keep="first")]
    return key_to_compartment, int(capillary.sum()), int(with_pericyte.sum())


def wall_compartments(
    adata,
    pixels,
    factor_order,
    vessels,
    structures,
    vessel_types,
    spillover=None,
    pericyte_radius_um=6.0,
    lattice_um=8.0,
) -> None:
    """Assign the 2 um bins of a section to vessel-wall compartments.

    Each 2 um bin gets the Level-3 type of its highest-probability decode pixel. Bins of the `vessel_types` are
    kept, and capillary EC bins are split by a pericyte bin within `pericyte_radius_um`. Bins whose lattice cell
    belongs to a ``Venous(muscular)`` structure form a further compartment, whatever their own type. With
    `spillover`, the 2 um bins that carry a CMFLAG spillover flag are left out of every compartment.

    Adds ``adata.obs["compartment"]`` (missing for bins outside the vessel wall), ``adata.obs["muscular_vein"]``
    and the bin counts in ``adata.uns["wall_compartments"]``.

    Parameters
    ----------
    adata
        Space Ranger 2 um bins of the section with ``obsm["spatial"]`` in um (:func:`hhca.io.read_visium_hd`).
    pixels
        Decode pixels of the section (``X``, ``Y``, ``K1``, ``P1``).
    factor_order
        Level-3 types in factor order.
    vessels, structures
        Lattice cells and structure table of the section.
    """
    key_to_compartment, n_capillary, n_with_pericyte = _compartment_of_key(
        pixels, factor_order, vessel_types, spillover, pericyte_radius_um
    )
    muscular = structures.loc[structures["vessel_type"] == "Venous(muscular)", "struct"].astype(np.int64)
    in_muscular = vessels.obs["struct"].astype(np.int64).isin(set(muscular.tolist())).values
    muscular_keys = set(vessels.obs["key"].astype(np.int64)[in_muscular].tolist()) if len(muscular) else set()

    x_um, y_um = adata.obsm["spatial"][:, 0], adata.obsm["spatial"][:, 1]
    key = bin_key(x_um, y_um)
    lattice_key = bin_key(x_um, y_um, lattice_um)
    in_muscular_vein = (
        np.fromiter((k in muscular_keys for k in lattice_key), dtype=bool, count=len(lattice_key))
        if muscular_keys
        else np.zeros(len(lattice_key), bool)
    )
    if spillover is not None:
        in_muscular_vein &= ~np.isin(key, spillover)
    adata.obs["compartment"] = key_to_compartment.reindex(key).values
    adata.obs["muscular_vein"] = in_muscular_vein
    adata.uns["wall_compartments"] = {
        "n_mapped": int(adata.obs["compartment"].notna().sum()),
        "n_cap": n_capillary,
        "n_periplus": n_with_pericyte,
        "n_mv_structs": int(len(muscular)),
        "n_mv_bins2um": int(in_muscular_vein.sum()),
    }


def marker_counts(adata, genes) -> pd.DataFrame:
    """Sum the counts of marker genes over the bins of each vessel-wall compartment.

    Uses ``adata.obs["compartment"]`` and ``adata.obs["muscular_vein"]`` of :func:`wall_compartments`.

    Returns
    -------
    One row per compartment and gene: ``type``, ``gene``, ``marker_sum`` (counts), ``n_expr`` (bins with a count),
    ``n_bins`` and ``total_counts`` (all features) of the compartment. ``attrs["genes_present"]`` tells which
    genes are in the matrix.
    """
    X = adata.X.tocsr()
    unique_names = pd.Index(ad.utils.make_index_unique(pd.Index(adata.var_names)))
    base = unique_names.str.replace(r"-\d+$", "", regex=True)
    total_per_bin = np.asarray(X.sum(1)).ravel()
    columns_of = {gene: np.where(base.values == gene)[0] for gene in genes}
    per_bin = {
        gene: (np.asarray(X[:, columns].sum(1)).ravel() if len(columns) else np.zeros(adata.n_obs))
        for gene, columns in columns_of.items()
    }

    compartment = adata.obs["compartment"].values
    mapped = pd.notna(compartment)
    mapped_index = np.where(mapped)[0]
    groups = [(name, mapped_index[compartment[mapped] == name]) for name in sorted(pd.unique(compartment[mapped]))]
    in_muscular_vein = adata.obs["muscular_vein"].values
    if in_muscular_vein.any():
        groups.append((MUSCULAR_VEIN, np.where(in_muscular_vein)[0]))

    rows = []
    for name, bin_index in groups:
        n_bins = int(len(bin_index))
        total = float(total_per_bin[bin_index].sum())
        for gene in genes:
            counts = per_bin[gene][bin_index]
            rows.append((name, gene, float(counts.sum()), int((counts > 0).sum()), n_bins, total))
    table = pd.DataFrame(rows, columns=["type", "gene", "marker_sum", "n_expr", "n_bins", "total_counts"])
    table.attrs["genes_present"] = {gene: bool(len(columns) > 0) for gene, columns in columns_of.items()}
    return table


def pool_markers(per_section) -> pd.DataFrame:
    """Pool the per-section marker tables into one compartment x gene table.

    ``cp10k`` = 1e4 * summed marker counts / summed total counts, ``pct`` = bins with a count / bins * 100,
    ``logcp`` = log1p(cp10k), all over the sections.
    """
    # n_bins and total_counts are repeated on every gene row of a (slide, type): take the first
    per_compartment = (
        per_section.groupby(["slide", "type"])
        .agg(n_bins=("n_bins", "first"), total_counts=("total_counts", "first"))
        .reset_index()
    )
    n_bins = per_compartment.groupby("type")["n_bins"].sum()
    total_counts = per_compartment.groupby("type")["total_counts"].sum()
    pooled = (
        per_section.groupby(["type", "gene"])
        .agg(marker_sum=("marker_sum", "sum"), n_expr=("n_expr", "sum"))
        .reset_index()
    )
    pooled["cp10k"] = pooled["marker_sum"] / pooled["type"].map(total_counts) * 1e4
    pooled["pct"] = pooled["n_expr"] / pooled["type"].map(n_bins) * 100
    pooled["logcp"] = np.log1p(pooled["cp10k"])
    return pooled


def marker_detection(pooled, genes, min_counts=20, min_pct=0.5) -> pd.DataFrame:
    """Per gene of a family: counts, best compartment and whether it is detected.

    Detected = at least `min_counts` counts over the compartments and at least `min_pct` % of bins in one.
    """
    records = []
    for gene in genes:
        sub = pooled[(pooled.gene == gene) & (pooled.type.isin(list(COMPARTMENT_LABELS)))]
        if len(sub) == 0:
            records.append(
                {
                    "gene": gene,
                    "total_counts": 0,
                    "n_expr": 0,
                    "max_cp10k": 0.0,
                    "max_pct": 0.0,
                    "best_col": "-",
                    "detected": False,
                }
            )
            continue
        total = float(sub.marker_sum.sum())
        best = sub["cp10k"].idxmax()
        records.append(
            {
                "gene": gene,
                "total_counts": total,
                "n_expr": int(sub.n_expr.sum()),
                "max_cp10k": float(sub.loc[best, "cp10k"]),
                "max_pct": float(sub["pct"].max()),
                "best_col": COMPARTMENT_LABELS[sub.loc[best, "type"]].replace("\n", " "),
                "detected": bool(total >= min_counts and sub["pct"].max() >= min_pct),
            }
        )
    return pd.DataFrame(records).sort_values("total_counts", ascending=False)
