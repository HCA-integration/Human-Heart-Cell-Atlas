"""Per-type expression of the snRNA atlas and the cardiomyocyte specificity of its genes."""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from hhca.tl._bins import CARDIOMYOCYTES, VENTRICULAR_CM


def _read_labels(obs, column):
    """Values of a categorical obs column of an open h5ad, for both on-disk layouts of anndata."""
    node = obs[column]
    if isinstance(node, h5py.Group):  # categorical stored as a group (anndata >= 0.8)
        categories = node["categories"][:].astype(str)
        return categories[node["codes"][:]]
    codes = node[:]
    categories = obs["__categories"][column][:].astype(str)
    return categories[codes]


def type_means(atlas, levels, layer="normcounts", chunk=20_000, log=None) -> dict:
    """Mean expression and detection fraction of every gene per cell type of the atlas.

    The layer is read from the h5ad file in row blocks, so the atlas is never held in memory. Mean = sum over
    the nuclei of a type / number of nuclei; detection = fraction of the nuclei with a value above 0.

    Parameters
    ----------
    atlas
        Path of the atlas h5ad.
    levels
        ``obs`` columns with cell-type labels; one set of tables per column.
    layer
        Layer to average.
    chunk
        Number of nuclei read at a time.
    log
        Function called with progress messages.

    Returns
    -------
    Mapping of level -> ``{"mean": genes x types, "detection": genes x types, "n_cells": nuclei per type}``.
    """
    with h5py.File(atlas, "r") as handle:
        var = handle["var"]["_index"][:].astype(str)
        matrix = handle["layers"][layer]
        n_cells, n_genes = (int(v) for v in matrix.attrs["shape"])
        if log:
            log(f"atlas {n_cells:,} cells x {n_genes:,} genes, layer {layer}")
        labels = {level: _read_labels(handle["obs"], level) for level in levels}
        types = {level: np.array(sorted(set(labels[level]))) for level in levels}
        codes = {level: np.searchsorted(types[level], labels[level]) for level in levels}
        sums = {level: np.zeros((len(types[level]), n_genes)) for level in levels}
        detected = {level: np.zeros((len(types[level]), n_genes)) for level in levels}
        indptr = matrix["indptr"]
        for start in range(0, n_cells, chunk):
            stop = min(n_cells, start + chunk)
            p0, p1 = int(indptr[start]), int(indptr[stop])
            block = sp.csr_matrix(
                (matrix["data"][p0:p1].astype(np.float64), matrix["indices"][p0:p1], indptr[start : stop + 1] - p0),
                shape=(stop - start, n_genes),
            )
            positive = (block > 0).astype(np.float64)
            for level in levels:
                membership = sp.csr_matrix(
                    (np.ones(stop - start), (codes[level][start:stop], np.arange(stop - start))),
                    shape=(len(types[level]), stop - start),
                )
                sums[level] += (membership @ block).toarray()
                detected[level] += (membership @ positive).toarray()
            if log and (start // chunk) % 10 == 0:
                log(f"  {stop:,}/{n_cells:,}")
    tables = {}
    for level in levels:
        n_per_type = np.bincount(codes[level], minlength=len(types[level])).astype(float)
        tables[level] = {
            "mean": pd.DataFrame((sums[level] / n_per_type[:, None]).T, index=var, columns=types[level]),
            "detection": pd.DataFrame((detected[level] / n_per_type[:, None]).T, index=var, columns=types[level]),
            "n_cells": pd.Series(n_per_type.astype(int), index=types[level], name="n_cells").rename_axis(level),
        }
    return tables


def _specificity_call(row):
    """Text label of the expressed / specific combination."""
    if row.cm_expressed and row.cm_specific:
        return "CM-specific"
    if row.cm_expressed and not row.cm_specific:
        return "CM-expressed but broad (not specific)"
    if (not row.cm_expressed) and row.cm_specific:
        return "CM-highest but low-expressed"
    return "non-CM (likely contamination)"


def cardiomyocyte_specificity(atlas_means, min_ventricular_mean=0.05, min_ratio=1.0) -> pd.DataFrame:
    """Call every gene of the atlas as cardiomyocyte-specific or not.

    A gene is ``cm_expressed`` when its mean in ventricular cardiomyocytes reaches `min_ventricular_mean`,
    ``cm_specific`` when the higher of the two cardiomyocyte means is at least `min_ratio` times the highest mean
    of any other type, and ``cm_safe`` when both hold.

    Parameters
    ----------
    atlas_means
        Genes x Level-3 types table of mean expression (:func:`type_means`).

    Returns
    -------
    One row per gene: the cardiomyocyte means, the top other type and its mean, the ratio, the three flags and
    a text call.
    """
    other = [column for column in atlas_means.columns if column not in CARDIOMYOCYTES]
    table = pd.DataFrame(
        {
            "gene": atlas_means.index,
            "VentCM": atlas_means[VENTRICULAR_CM].values,
            "AtrCM": atlas_means["Atrial Cardiomyocytes"].values,
            "CM_max": atlas_means[CARDIOMYOCYTES].max(axis=1).values,
            "top_nonCM": atlas_means[other].idxmax(axis=1).values,
            "top_nonCM_val": atlas_means[other].max(axis=1).values,
            "overall_top": atlas_means.idxmax(axis=1).values,
        }
    )
    table["CM_over_topNonCM"] = table["CM_max"] / (table["top_nonCM_val"] + 1e-9)
    table["cm_is_overall_top"] = table["overall_top"].isin(CARDIOMYOCYTES)
    table["cm_expressed"] = table["VentCM"] >= min_ventricular_mean
    table["cm_specific"] = table["CM_over_topNonCM"] >= min_ratio
    table["cm_safe"] = table["cm_expressed"] & table["cm_specific"]
    table["cm_call"] = table.apply(_specificity_call, axis=1)
    return table
