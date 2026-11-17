"""Pseudo-cells of one cell type from the decode pixels, their surroundings and marker regressions."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
from statsmodels.stats.multitest import multipletests

from hhca.tl._bins import _coordinates
from hhca.tl._nerves import _of_types

CENTROID = ["centroid_x_um", "centroid_y_um"]


def _group_pixels(xy, rows, eps_um, min_pixels, split_pixels, tile_um):
    """Cluster pixels into pseudo-cells: list of (count-matrix rows of the member pixels, centroid)."""
    from sklearn.cluster import DBSCAN

    labels = DBSCAN(eps=eps_um, min_samples=min_pixels, n_jobs=1).fit_predict(xy)
    groups = []
    for label in np.unique(labels):
        if label < 0:
            continue
        member = labels == label
        member_xy, member_rows = xy[member], rows[member]
        if member.sum() <= split_pixels:
            groups.append((member_rows, member_xy.mean(0)))
            continue
        tile_x = np.floor(member_xy[:, 0] / tile_um).astype(np.int64)
        tile_y = np.floor(member_xy[:, 1] / tile_um).astype(np.int64)
        tile = tile_x * 1000003 + tile_y
        for key in np.unique(tile):
            in_tile = tile == key
            if in_tile.sum() >= min_pixels:
                groups.append((member_rows[in_tile], member_xy[in_tile].mean(0)))
    return groups


def pseudocells(
    adata, pixels, factor, eps_um=10.0, min_pixels=3, split_pixels=50, tile_um=12.0, min_counts=100
) -> ad.AnnData:
    """Build pseudo-cells of one cell type on one section.

    The decode pixels whose most probable type is `factor` are clustered with DBSCAN (`eps_um`, `min_pixels`);
    a cluster of more than `split_pixels` pixels is cut into tiles of `tile_um` and the tiles with at least
    `min_pixels` pixels are kept. The counts of a pseudo-cell are the summed counts of the 2 um bins nearest to
    its pixels, each bin once; pseudo-cells with fewer than `min_counts` counts are dropped.

    Parameters
    ----------
    adata
        Space Ranger 2 um bins of the section (:func:`hhca.io.read_visium_hd`).
    pixels
        Decode pixels of the section (``X``, ``Y``, ``K1``).
    factor
        Code of the cell type in the factor order.

    Returns
    -------
    AnnData of pseudo-cells x genes with summed counts; ``obs``: ``centroid_x_um``, ``centroid_y_um`` (mean
    position of the member pixels), ``n_bins`` and ``total_counts``.
    """
    xy = pixels[["X", "Y"]].to_numpy().astype(np.float64)
    selected = np.where(np.isin(pixels["K1"].to_numpy(), [factor]))[0]
    _, rows = cKDTree(adata.obsm["spatial"]).query(xy[selected], k=1)
    groups = _group_pixels(xy[selected], rows, eps_um, min_pixels, split_pixels, tile_um)

    cell_index, bin_index, centroids, n_bins = [], [], [], []
    for cell, (member_rows, centroid) in enumerate(groups):
        bins = np.unique(member_rows)
        cell_index.append(np.full(len(bins), cell))
        bin_index.append(bins)
        centroids.append(centroid)
        n_bins.append(len(bins))
    cell_index, bin_index = np.concatenate(cell_index), np.concatenate(bin_index)
    membership = sp.coo_matrix(
        (np.ones(len(cell_index), np.float32), (cell_index, bin_index)), shape=(len(groups), adata.n_obs)
    ).tocsr()
    counts = sp.csr_matrix(membership.dot(adata.X))
    centroids, n_bins = np.asarray(centroids), np.asarray(n_bins)
    total = np.asarray(counts.sum(1)).ravel()

    keep = total >= min_counts
    obs = pd.DataFrame(
        {
            "centroid_x_um": centroids[keep, 0],
            "centroid_y_um": centroids[keep, 1],
            "n_bins": n_bins[keep],
            "total_counts": total[keep],
        },
        index=[str(i) for i in range(int(keep.sum()))],
    )
    return ad.AnnData(X=counts[keep], obs=obs, var=pd.DataFrame(index=np.asarray(adata.var_names)))


def nearest_structure(table, structures=None) -> None:
    """Add the nearest vessel structure of one section to a per-cell table.

    Parameters
    ----------
    table
        Per-cell table with the centroids (``cells.obs``).
    structures
        Calibre table of the section (``x_um``, ``y_um``, ``lumen_equiv_diam_um``, ``vessel_class``); without
        it the columns are empty.

    Adds ``dist_vessel_struct_um`` (distance to the centre of the nearest structure), ``diam_near`` (its
    diameter) and ``class_near`` (its calibre class).
    """
    if structures is None:
        table["dist_vessel_struct_um"] = np.full(len(table), np.nan)
        table["diam_near"] = np.full(len(table), np.nan)
        table["class_near"] = np.full(len(table), "NA", dtype=object)
        return
    distance, nearest = cKDTree(structures[["x_um", "y_um"]].values.astype(float)).query(table[CENTROID].to_numpy())
    table["dist_vessel_struct_um"] = distance
    table["diam_near"] = structures["lumen_equiv_diam_um"].values[nearest]
    table["class_near"] = structures["vessel_class"].values[nearest]


def distance_to_bins(table, sections, types, key) -> pd.Series:
    """Add the distance from every cell to the nearest 2 um bin of some Level-3 types.

    Parameters
    ----------
    table
        Per-cell table with ``slide`` and the centroids (``cells.obs``).
    sections
        Mapping of section name -> section.
    types
        Level-3 labels of the bins.
    key
        Name of the added column.

    Returns
    -------
    Number of bins of `types` per section.
    """
    distance = np.full(len(table), np.nan)
    n_bins = {}
    for slide in table["slide"].unique():
        adata = sections[slide]
        x, y = _coordinates(adata)
        of_type = _of_types(adata, types)
        on_slide = (table["slide"] == slide).to_numpy()
        n_bins[slide] = int(of_type.sum())
        if of_type.any():
            distance[on_slide] = cKDTree(np.c_[x[of_type], y[of_type]]).query(table.loc[on_slide, CENTROID].values)[0]
    table[key] = distance
    return pd.Series(n_bins, name="n_bins")


def nerve_distance(table, neural_bins, min_sizes=(3,)) -> pd.DataFrame:
    """Add the distance from every cell to the nearest nerve, for nerves of several minimum sizes.

    For every `k` of `min_sizes`, ``dist_nerve_k<k>`` is the distance to the nearest neural bin that is linked
    with at least `k` neural bins, itself included; ``k = 1`` counts every neural bin.

    Parameters
    ----------
    table
        Per-cell table with ``slide`` and the centroids (``cells.obs``).
    neural_bins
        Mapping of section name -> neural bins of :func:`hhca.tl.call_nerves`.

    Returns
    -------
    One row per section: the number of neural bins and linked groups, and per `k` the bins (``n_bins_k<k>``)
    and groups (``n_nerves_k<k>``) that are kept.
    """
    for k in min_sizes:
        table[f"dist_nerve_k{k}"] = np.nan
    retention = []
    for slide in table["slide"].unique():
        bins = neural_bins[slide]
        xy = bins[["x_um", "y_um"]].to_numpy()
        comp_size = bins["comp_size"].to_numpy()
        # number of groups of each size: bins of that size divided by the size
        sizes = np.arange(comp_size.max() + 1)
        n_groups = np.bincount(comp_size) // np.maximum(sizes, 1)
        on_slide = (table["slide"] == slide).to_numpy()
        centroids = table.loc[on_slide, CENTROID].values
        row = {"slide": slide, "n_neural": len(xy), "n_components": int(n_groups.sum())}
        for k in min_sizes:
            keep = comp_size >= k
            row[f"n_bins_k{k}"] = int(keep.sum())
            row[f"n_nerves_k{k}"] = int(n_groups[sizes >= k].sum())
            if keep.sum() == 0:
                continue
            table.loc[on_slide, f"dist_nerve_k{k}"] = cKDTree(xy[keep]).query(centroids)[0]
        retention.append(row)
    return pd.DataFrame(retention)


def marker_expression(cells, markers) -> pd.DataFrame:
    """Expression of a few genes in every cell, in double precision.

    Returns
    -------
    Cells x three columns per gene: ``<gene>_raw`` (counts), ``<gene>_log1p_cp10k`` (counts scaled to 10,000
    per cell over all genes of `cells`, log1p) and ``<gene>_det`` (1 if the cell has a count).
    """
    matrix = cells.X.tocsc()
    var_names = np.asarray(cells.var_names)
    total = np.asarray(cells.X.sum(1)).ravel().astype(float)
    table = {}
    for gene in markers:
        raw = np.asarray(matrix[:, np.where(var_names == gene)[0][0]].todense()).ravel()
        table[f"{gene}_raw"] = raw
        table[f"{gene}_log1p_cp10k"] = np.log1p(raw / total * 1e4)
        table[f"{gene}_det"] = (raw > 0).astype(int)
    return pd.DataFrame(table, index=cells.obs_names)


def marker_effects(table, markers, term=None, adjust=("z_lc",), column="{marker}_log1p_cp10k") -> pd.DataFrame:
    """Regress the expression of every marker on one term, adjusted for covariates and the section.

    Per marker an ordinary least-squares model ``expression ~ term + covariates + C(slide)`` with
    heteroscedasticity-robust (HC3) standard errors.

    Parameters
    ----------
    table
        Per-cell table with the expression columns, `term`, the covariates and ``slide``.
    term
        Column whose coefficient is reported; None fits the covariates only.
    adjust
        Covariate columns.
    column
        Name of the expression column of a marker.

    Returns
    -------
    One row per marker: ``beta`` (coefficient of `term`), ``ci_lo``, ``ci_hi`` (95 % interval), ``p``, ``padj``
    (Benjamini-Hochberg over the markers), ``n`` (cells in the model) and ``r2``.
    """
    import statsmodels.formula.api as smf

    terms = [*([] if term is None else [term]), *adjust, "C(slide)"]
    rows = []
    for marker in markers:
        fit = smf.ols(f"{column.format(marker=marker)} ~ {' + '.join(terms)}", data=table).fit(cov_type="HC3")
        row = {"marker": marker, "beta": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan}
        if term is not None:
            interval = fit.conf_int()
            row.update(
                beta=fit.params[term], ci_lo=interval.loc[term, 0], ci_hi=interval.loc[term, 1], p=fit.pvalues[term]
            )
        rows.append({**row, "n": int(fit.nobs), "r2": fit.rsquared})
    effects = pd.DataFrame(rows)
    effects.insert(5, "padj", np.nan)
    if term is not None:
        effects["padj"] = multipletests(effects["p"], method="fdr_bh")[1]
    return effects


def marker_means(table, markers, group="peri", column="{marker}_log1p_cp10k") -> pd.DataFrame:
    """Mean expression of every marker in the cells with and without a 0/1 indicator, per section.

    Returns
    -------
    One row per marker and section: ``slide``, ``marker``, ``n_perineural`` and ``n_other`` (cells with and
    without the indicator) and their means ``mean_perineural`` and ``mean_other``.
    """
    rows = []
    for marker in markers:
        values = column.format(marker=marker)
        for slide, cells in table.groupby("slide", observed=True):
            inside = cells[group] == 1
            rows.append(
                {
                    "slide": slide,
                    "marker": marker,
                    "n_perineural": int(cells[group].sum()),
                    "n_other": int((1 - cells[group]).sum()),
                    "mean_perineural": float(cells.loc[inside, values].mean()) if cells[group].any() else np.nan,
                    "mean_other": float(cells.loc[cells[group] == 0, values].mean()),
                }
            )
    return pd.DataFrame(rows)


def expression_by_distance(table, column, distance, edges) -> pd.DataFrame:
    """Mean of an expression column in bins of a distance column.

    Returns
    -------
    One row per distance bin (left-closed, labelled ``<low>-<high>``): ``mean``, ``sem`` and ``size``.
    """
    labels = [f"{low}-{high}" for low, high in zip(edges[:-1], edges[1:])]
    distance_bin = pd.cut(table[distance], bins=list(edges), labels=labels, right=False)
    return table.groupby(distance_bin, observed=True)[column].agg(["mean", "sem", "size"]).reindex(labels)
