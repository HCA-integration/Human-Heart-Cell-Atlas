"""AnnData objects of a Visium HD section: the FICTURE decode joined to the Space Ranger bins at 2 um and 8 um."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from hhca.grid import grid_index

# 8 um block id = block row * BLOCK_KEYMUL + block column
BLOCK_KEYMUL = np.int64(100_000)
SCORE_NOTE = "certainty_score (0.15*margin+0.35*depth+0.30*marker+0.20*spatial)"


def read_bin_positions(directory) -> pd.DataFrame:
    """All bins of a Space Ranger ``square_NNNum`` folder, in tissue or not, with their centres in um.

    Returns
    -------
    ``spatial/tissue_positions.parquet`` with the columns ``x_um`` and ``y_um`` added (full-resolution pixel
    times ``microns_per_pixel``); ``attrs["microns_per_pixel"]`` holds the scale.
    """
    import pyarrow.parquet as pq

    with open(f"{directory}/spatial/scalefactors_json.json") as handle:
        microns_per_pixel = float(json.load(handle)["microns_per_pixel"])
    positions = pq.read_table(f"{directory}/spatial/tissue_positions.parquet").to_pandas()
    positions["x_um"] = positions["pxl_col_in_fullres"].values * microns_per_pixel
    positions["y_um"] = positions["pxl_row_in_fullres"].values * microns_per_pixel
    positions.attrs["microns_per_pixel"] = microns_per_pixel
    return positions


def read_bins(directory):
    """Read the filtered bins of a Space Ranger ``square_NNNum`` folder as AnnData.

    Returns
    -------
    Bins x features with raw counts (float32 CSR); ``obs["array_row"]``, ``obs["array_col"]`` (position on the
    bin grid), ``obsm["spatial"]`` (bin centres in um), unique ``var_names`` and ``uns["microns_per_pixel"]``.
    """
    import anndata as ad
    import h5py

    with h5py.File(f"{directory}/filtered_feature_bc_matrix.h5", "r") as handle:
        matrix = handle["matrix"]
        n_genes, n_barcodes = (int(v) for v in matrix["shape"][:])
        counts = sp.csc_matrix(
            (matrix["data"][:], matrix["indices"][:], matrix["indptr"][:]), shape=(n_genes, n_barcodes)
        )
        barcodes = np.array([b.decode() for b in matrix["barcodes"][:]])
        features = {
            name: np.array([b.decode() for b in matrix["features"][name][:]])
            for name in ("name", "id", "feature_type", "genome")
        }
    positions = read_bin_positions(directory)
    microns_per_pixel = positions.attrs["microns_per_pixel"]
    positions = positions.set_index("barcode").reindex(barcodes)
    var = pd.DataFrame(
        {
            "gene_ids": features["id"],
            "feature_types": pd.Categorical(features["feature_type"]),
            "genome": pd.Categorical(features["genome"]),
        },
        index=pd.Index(features["name"]),
    )
    adata = ad.AnnData(
        X=counts.T.tocsr().astype(np.float32),
        obs=pd.DataFrame(
            {
                "array_row": positions["array_row"].values.astype(np.int64),
                "array_col": positions["array_col"].values.astype(np.int64),
            },
            index=pd.Index(barcodes),
        ),
        var=var,
    )
    adata.var_names_make_unique()
    adata.obsm["spatial"] = np.column_stack([positions["x_um"].values, positions["y_um"].values])
    adata.uns["microns_per_pixel"] = microns_per_pixel
    return adata


def _collapse_decode(decode):
    """One decode row per unique 2 um key: highest P1, first decode row on ties (lexsort is stable)."""
    grid_x, grid_y, key = grid_index(decode["X"].values, decode["Y"].values)
    order = np.lexsort((-decode["P1"].values, key))
    sorted_keys = key[order]
    selected = order[np.r_[True, sorted_keys[1:] != sorted_keys[:-1]]]  # sorted by key
    return selected, grid_x[selected], grid_y[selected], key[selected]


def _match_keys(table_keys, keys):
    """Row of `table_keys` for each of `keys` (first row wins on duplicates): found mask and rows of the found."""
    key_to_row = pd.Series(np.arange(len(table_keys), dtype=np.int64), index=table_keys)
    key_to_row = key_to_row[~key_to_row.index.duplicated(keep="first")]
    row = key_to_row.reindex(keys).values
    found = np.isfinite(row)
    return found, row[found].astype(np.int64)


def _row_scores(certainty, n_rows):
    """Certainty score per decode row as an array; the table must have one value per decode row."""
    scores = np.asarray(certainty)
    if len(scores) != n_rows:
        raise ValueError(f"{len(scores)} certainty scores for {n_rows} decode rows")
    return scores


def bins_2um(decode, bins, factor_order, section, region, certainty=None):
    """Join the FICTURE decode of a section to its Space Ranger 2 um bins.

    The decode has several rows per 2 um bin. A bin is represented by its row with the highest posterior ``P1``
    (the first one on ties) and addressed by the grid key ``round(x / 2), round(y / 2)`` of its coordinates in um;
    the same key of the Space Ranger bin centres gives the row of the count matrix. Decode bins without a Space
    Ranger bin are dropped.

    Parameters
    ----------
    decode
        Pixel decode of the section (``X``, ``Y``, ``K1``, ``P1``).
    bins
        Space Ranger 2 um bins (:func:`read_bins`).
    factor_order
        Level-3 types in factor order.
    section
        Section id, the prefix of the observation names.
    region
        Position of the section in the wall, stored in ``uns["build"]``.
    certainty
        Certainty score per decode row (optional).

    Returns
    -------
    AnnData of 2 um bins with raw counts; ``obs``: ``L3_label`` (top factor), ``gcert`` (certainty score),
    ``posterior_max`` (``P1``), ``x_um``, ``y_um`` (decode coordinates), ``slide``; the join statistics are in
    ``uns["build"]``.
    """
    import anndata as ad

    selected, grid_x, grid_y, key = _collapse_decode(decode)
    bin_x, bin_y = bins.obsm["spatial"][:, 0], bins.obsm["spatial"][:, 1]
    _, _, bin_key = grid_index(bin_x, bin_y)
    matched, rows = _match_keys(bin_key, key)
    selected, grid_x, grid_y = selected[matched], grid_x[matched], grid_y[matched]

    X = bins.X[rows]
    X.indices = X.indices.astype(np.int32)
    X.indptr = X.indptr.astype(np.int32)
    decode_x = decode["X"].values[selected]
    decode_y = decode["Y"].values[selected]
    if certainty is not None:
        gcert = _row_scores(certainty, len(decode))[selected].astype(np.float32)
        gcert_note = SCORE_NOTE
    else:
        gcert = np.full(len(selected), np.nan, np.float32)
        gcert_note = "NaN (no certainty checkpoint for this slide)"
    obs = pd.DataFrame(
        {
            "L3_label": pd.Categorical.from_codes(
                decode["K1"].values[selected].astype(np.int64), categories=factor_order
            ),
            "gcert": gcert,
            "posterior_max": decode["P1"].values[selected].astype(np.float32),
            "x_um": decode_x.astype(np.float32),
            "y_um": decode_y.astype(np.float32),
            "slide": pd.Categorical([section] * len(selected)),
        },
        index=pd.Index([f"{section}__{gx}_{gy}" for gx, gy in zip(grid_x, grid_y)]),
    )
    residual = np.hypot(decode_x.astype(np.float32) - bin_x[rows], decode_y.astype(np.float32) - bin_y[rows])
    adata = ad.AnnData(X=X, obs=obs, var=bins.var.copy())
    adata.uns["build"] = {
        "source": "fixed-model FICTURE 2um L3 decode JOIN square_002um RAW counts",
        "join": "grid2key round(um/2)*1e7; one obs per unique 2um bin, max-P1 decode row; key2row first-wins",
        "decode_rows": int(len(decode)),
        "unique_2um_bins": int(len(matched)),
        "matched_bins": int(matched.sum()),
        "coverage_pct": float(matched.sum() / len(matched) * 100.0),
        "join_residual_um_median": float(np.median(residual)),
        "join_residual_um_frac_gt1": float((residual > 1).mean()),
        "mpp": float(bins.uns["microns_per_pixel"]),
        "region": region,
        "gcert": gcert_note,
        "L3_factor_order": f"atlas_L3_factor_order.json ({len(factor_order)} types)",
    }
    return adata


def aggregate_bins(bins, factor: int = 4):
    """Sum Space Ranger bins into blocks of `factor` x `factor` bins (a section without ``square_008um``).

    Returns
    -------
    AnnData of the blocks with at least one bin, sorted by block row and column: summed counts,
    ``obs["array_row"]``, ``obs["array_col"]`` (block position) and ``obsm["spatial"]`` (mean of the bin centres).
    """
    import anndata as ad

    array_row, array_col = bins.obs["array_row"].values, bins.obs["array_col"].values
    block = (array_row // factor).astype(np.int64) * BLOCK_KEYMUL + array_col // factor
    blocks, block_of_bin = np.unique(block, return_inverse=True)
    n_bins = len(block_of_bin)
    membership = sp.csr_matrix(
        (np.ones(n_bins, np.float32), (block_of_bin, np.arange(n_bins))), shape=(len(blocks), n_bins)
    )
    bins_per_block = np.bincount(block_of_bin)
    aggregated = ad.AnnData(
        X=(membership @ bins.X).tocsr(),
        obs=pd.DataFrame(
            {
                "array_row": (blocks // BLOCK_KEYMUL).astype(np.int64),
                "array_col": (blocks % BLOCK_KEYMUL).astype(np.int64),
            },
            index=pd.Index(blocks.astype(str)),
        ),
        var=bins.var.copy(),
    )
    aggregated.obsm["spatial"] = np.column_stack(
        [
            np.bincount(block_of_bin, bins.obsm["spatial"][:, 0]) / bins_per_block,
            np.bincount(block_of_bin, bins.obsm["spatial"][:, 1]) / bins_per_block,
        ]
    )
    aggregated.uns["aggregated"] = True
    return aggregated


def bins_8um(decode, positions, bins, factor_order, section, certainty=None, decode_label=""):
    """Aggregate the 2 um labels of a section to its 8 um bins.

    The decode is collapsed to one row per 2 um bin as in :func:`bins_2um`. Each 2 um bin falls into the 8 um
    bin at ``array_row // 4``, ``array_col // 4`` of the 2 um bin grid. An 8 um bin takes the most frequent label
    of its 2 um bins (the first type on ties).

    Parameters
    ----------
    decode
        Pixel decode of the section (``X``, ``Y``, ``K1``, ``P1``).
    positions
        All 2 um bins of the section (:func:`read_bin_positions`).
    bins
        The 8 um bins with their counts: Space Ranger ``square_008um`` (:func:`read_bins`) or
        :func:`aggregate_bins` of the 2 um bins.
    factor_order
        Level-3 types in factor order.
    section
        Section id, the prefix of the observation names.
    certainty
        Certainty score per decode row (optional).
    decode_label
        Path of the decode as written to ``uns["build"]``.

    Returns
    -------
    AnnData of 8 um bins; ``obs``: ``L3_label`` (majority label), ``L3_top1_frac``, ``gcert_mean``, ``n_2um_bins``,
    ``x_um``, ``y_um``, ``slide``; ``obsm["L3_composition"]`` holds the share of each label among the 2 um bins
    (columns in the order of ``uns["L3_composition_order"]``).
    """
    import anndata as ad

    n_types = len(factor_order)
    selected, _, _, key = _collapse_decode(decode)
    label_2um = decode["K1"].values[selected].astype(np.int64)
    gcert_2um = None if certainty is None else _row_scores(certainty, len(decode))[selected].astype(np.float32)

    _, _, bin_key = grid_index(positions["x_um"].values, positions["y_um"].values)
    matched, rows = _match_keys(bin_key, key)
    array_row, array_col = positions["array_row"].values[rows], positions["array_col"].values[rows]
    block_of_2um = (array_row // 4).astype(np.int64) * BLOCK_KEYMUL + array_col // 4

    X = bins.X.copy()
    X.sort_indices()
    X.indices = X.indices.astype(np.int32)
    X.indptr = X.indptr.astype(np.int32)
    block_row, block_col = bins.obs["array_row"].values, bins.obs["array_col"].values
    x_um, y_um = bins.obsm["spatial"][:, 0], bins.obsm["spatial"][:, 1]
    if bins.uns.get("aggregated", False):
        counts_note = "raw counts = 4x4 aggregation of square_002um (no square_008um for AH slides)"
    else:
        counts_note = "raw square_008um filtered counts (SpaceRanger)"
    n_blocks = len(block_row)

    position = pd.Index(block_row * BLOCK_KEYMUL + block_col).get_indexer(block_of_2um)
    inside = position >= 0
    position, label = position[inside], label_2um[matched][inside]
    n_2um = np.bincount(position, minlength=n_blocks)
    type_counts = np.zeros((n_blocks, n_types), np.float64)
    np.add.at(type_counts, (position, label), 1)
    covered = n_2um > 0
    majority = np.where(covered, type_counts.argmax(1), -1)
    composition = np.where(covered[:, None], type_counts / np.maximum(n_2um, 1)[:, None], 0.0).astype(np.float32)
    top1_frac = np.where(covered, type_counts.max(1) / np.maximum(n_2um, 1), np.nan).astype(np.float32)
    if gcert_2um is not None:
        gcert_sum = np.zeros(n_blocks, np.float64)
        np.add.at(gcert_sum, position, gcert_2um[matched][inside].astype(np.float64))
        gcert_mean = np.where(covered, gcert_sum / np.maximum(n_2um, 1), np.nan).astype(np.float32)
    else:
        gcert_mean = np.full(n_blocks, np.nan, np.float32)

    obs = pd.DataFrame(
        {
            "L3_label": pd.Categorical.from_codes(majority, categories=factor_order),
            "L3_top1_frac": top1_frac,
            "gcert_mean": gcert_mean,
            "n_2um_bins": n_2um.astype(np.int32),
            "x_um": x_um.astype(np.float32),
            "y_um": y_um.astype(np.float32),
            "slide": pd.Categorical([section] * n_blocks),
        },
        index=pd.Index([f"{section}__{row}_{col}" for row, col in zip(block_row, block_col)]),
    )
    adata = ad.AnnData(X=X, obs=obs, var=bins.var.copy())
    adata.obsm["L3_composition"] = composition
    adata.obsm["spatial"] = np.c_[x_um, y_um].astype(np.float32)
    adata.uns["L3_composition_order"] = np.array(factor_order, dtype=object)
    adata.uns["slide"] = section
    adata.uns["build"] = {
        "resolution_um": 8,
        "aggregated_from": (
            "fixed-model FICTURE 2um L3 decode; r8=array_row//4, c8=array_col//4 (square_002um barcode grid)"
        ),
        "decode": decode_label,
        "X": counts_note,
        "L3_label": "dominant (majority) L3 across constituent unique 2um bins (max-P1 per bin)",
        "L3_composition": "obsm (n8 x 23) fraction of each L3 type; order = uns L3_composition_order",
        "gcert": (
            "mean certainty_score (0.15*margin+0.35*depth+0.30*marker+0.20*spatial) over constituent bins; "
            "NaN for AH (no checkpoint)"
        ),
        "coords": "physical SpaceRanger microns (pxl_in_fullres * microns_per_pixel)",
        "L3_factor_order": f"atlas_L3_factor_order.json ({n_types} types)",
        "n_8um_bins": int(n_blocks),
        "coverage_pct": float(100.0 * covered.sum() / n_blocks),
    }
    return adata
