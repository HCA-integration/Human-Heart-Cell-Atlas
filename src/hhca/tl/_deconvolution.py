"""Deconvolution of standard Visium spots with the atlas reference (Fig 3 c, ED6 c, g, h, ED8 b-c).

A 55 um Visium spot covers several cells, so its expression is a mixture of cell types. The functions here build
the single-cell reference and the gene panel shared by the methods, fit the mixture by non-negative least squares
(NNLS), build spots of known composition from Visium HD to score a method against, compare the per-spot
proportions of several methods, and use the proportions to select cardiomyocyte-dominant and pericyte-high spots.

Per-spot proportions live in ``adata.obsm[<method>]`` as a table of spots x Level-3 types; the known composition
of simulated spots is ``adata.obsm["truth"]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hhca.io import decode

# ---------------------------------------------------------------------------------------------------------------
# reference and gene panel


def _read_obs_column(obs_group, column) -> np.ndarray:
    """One obs column of an open h5ad as an object array (categoricals decoded, missing -> 'NA')."""
    import h5py

    node = obs_group[column]
    if isinstance(node, h5py.Group) and "categories" in node:
        categories = np.array([decode(value) for value in node["categories"][:]], dtype=object)
        codes = node["codes"][:]
        out = np.empty(len(codes), dtype=object)
        valid = codes >= 0
        out[valid] = categories[codes[valid]]
        out[~valid] = "NA"
        return out
    values = node[:]
    if values.dtype.kind in ("S", "O"):
        return np.array([decode(value) for value in values], dtype=object)
    return values


def _read_count_rows(counts_group, rows, n_genes, log=None, step=2000):
    """Rows of a CSR count matrix of an open h5ad, read one by one."""
    import scipy.sparse as sp

    assert dict(counts_group.attrs).get("encoding-type") == "csr_matrix", dict(counts_group.attrs)
    indptr = counts_group["indptr"][:]
    data_ds, indices_ds = counts_group["data"], counts_group["indices"]
    row_data, row_indices, row_indptr = [], [], [0]
    for block_start in range(0, len(rows), step):
        block = rows[block_start : block_start + step]
        for row in block:
            start, end = int(indptr[row]), int(indptr[row + 1])
            row_data.append(data_ds[start:end])
            row_indices.append(indices_ds[start:end])
            row_indptr.append(row_indptr[-1] + (end - start))
        if log is not None:
            log(f"  read rows {block_start + len(block)}/{len(rows)}")
    data = np.concatenate(row_data) if row_data else np.array([], dtype=data_ds.dtype)
    indices = np.concatenate(row_indices) if row_indices else np.array([], dtype=np.int64)
    return sp.csr_matrix((data, indices, np.array(row_indptr, dtype=np.int64)), shape=(len(rows), n_genes))


def sample_reference(
    atlas, label, n_per_type, seed=0, exclude=("Unassigned",), obs_columns=(), layer="counts", log=None
):
    """Draw the single-cell reference of the deconvolution methods from the atlas.

    At most `n_per_type` nuclei are drawn per cell type; their raw counts are read from the atlas file without
    loading it.

    Parameters
    ----------
    atlas
        Path of the atlas h5ad.
    label
        obs column with the cell types.
    n_per_type
        Nuclei drawn per cell type (all of them for smaller types).
    seed
        Seed of the draw.
    exclude
        Labels that are left out.
    obs_columns
        Further obs columns to keep, such as the donor.
    layer
        Layer with the raw counts.

    Returns
    -------
    AnnData of the drawn nuclei: raw counts in ``X``, the cell type in ``obs["cell_type_L3"]``, the nuclei kept
    per type in ``uns["n_cells"]``.
    """
    import anndata as ad
    import h5py

    rng = np.random.default_rng(seed)
    with h5py.File(atlas, "r") as handle:
        obs_group = handle["obs"]
        labels = _read_obs_column(obs_group, label)
        drawn, n_cells = [], {}
        for cell_type in sorted(set(labels) - set(exclude) - {"NA"}):
            rows = np.where(labels == cell_type)[0]
            if len(rows) > n_per_type:
                rows = rng.choice(rows, size=n_per_type, replace=False)
            drawn.append(rows)
            n_cells[cell_type] = int(len(rows))
        rows = np.sort(np.concatenate(drawn))
        meta = {column: _read_obs_column(obs_group, column)[rows] for column in obs_columns if column in obs_group}
        index_name = handle["var"].attrs.get("_index", "_index")
        genes = np.array([decode(gene) for gene in handle["var"][decode(index_name)][:]], dtype=object)
        counts = _read_count_rows(handle["layers"][layer], rows, len(genes), log)

    obs = pd.DataFrame({"cell_type_L3": pd.Categorical(labels[rows].astype(str))})
    for column, values in meta.items():
        obs[column] = pd.Categorical(values.astype(str))
    obs.index = [f"ref_{i}" for i in range(len(rows))]
    reference = ad.AnnData(X=counts.astype(np.float32), obs=obs, var=pd.DataFrame(index=pd.Index(genes.astype(str))))
    reference.var_names_make_unique()
    reference.uns["n_cells"] = n_cells
    return reference


def gene_panel(reference, gene_lists, n_hvg, markers, min_cells=5) -> list[str]:
    """Gene panel shared by cell2location and stereoscope.

    The highly variable genes of the reference (scanpy, dispersion on log1p CP10k) among the genes measured in
    every section, plus the marker genes measured in every section.

    Parameters
    ----------
    reference
        Single-cell reference with raw counts (:func:`sample_reference`).
    gene_lists
        Genes of each section.
    n_hvg
        Number of highly variable genes.
    markers
        Marker genes that are always kept.
    min_cells
        Reference nuclei a gene needs to be considered.

    Returns
    -------
    The genes of the panel, sorted.
    """
    import scanpy as sc

    common = set(reference.var_names)
    for genes in gene_lists:
        common &= set(genes)
    shared = reference[:, sorted(common)].copy()
    sc.pp.filter_genes(shared, min_cells=min_cells)
    sc.pp.normalize_total(shared, target_sum=1e4)
    sc.pp.log1p(shared)
    sc.pp.highly_variable_genes(shared, n_top_genes=n_hvg)
    variable = list(shared.var_names[shared.var["highly_variable"].values])
    return sorted(set(variable) | (set(markers) & common))


# ---------------------------------------------------------------------------------------------------------------
# NNLS


def _dense(matrix) -> np.ndarray:
    """A sparse or dense matrix as a dense array."""
    from scipy.sparse import issparse

    return matrix.toarray() if issparse(matrix) else np.asarray(matrix)


def visium_spots(directory, min_cells, min_counts):
    """Read the spots of one Space Ranger output folder with the filters of the deconvolution.

    Reads ``filtered_feature_bc_matrix.h5`` and, where the folder has ``spatial/``, the spot positions. Genes
    detected in fewer than `min_cells` spots and spots with fewer than `min_counts` counts are left out.

    Returns
    -------
    AnnData with raw counts in ``X`` and ``obs["total_counts"]``; ``obsm["spatial"]`` (full-resolution pixels),
    ``obs["array_row"]`` and ``obs["array_col"]`` where the positions exist.
    """
    import os

    import scanpy as sc

    adata = sc.read_10x_h5(f"{directory}/filtered_feature_bc_matrix.h5")
    adata.var_names_make_unique()
    for file_name in ("tissue_positions.csv", "tissue_positions_list.csv"):
        path = f"{directory}/spatial/{file_name}"
        if not os.path.exists(path):
            continue
        positions = pd.read_csv(path)
        if list(positions.columns[:2]) != ["barcode", "in_tissue"]:
            columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
            positions = pd.read_csv(path, header=None, names=columns)
        positions = positions.set_index("barcode").reindex(adata.obs_names)
        adata = adata[positions["in_tissue"].notna().values].copy()
        positions = positions.loc[adata.obs_names]
        adata.obs["array_row"] = positions["array_row"].astype(int).values
        adata.obs["array_col"] = positions["array_col"].astype(int).values
        adata.obsm["spatial"] = positions[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=float)
        break
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_counts=min_counts)
    adata.obs["total_counts"] = np.asarray(adata.X.sum(1)).ravel()
    return adata


def reference_signatures(reference, label="cell_type_L3") -> pd.DataFrame:
    """Expression signature of every cell type of the reference: mean CP10k of its nuclei (genes x cell types)."""
    import scanpy as sc

    normalised = reference.copy()
    sc.pp.normalize_total(normalised, target_sum=1e4)
    expression = _dense(normalised.X)
    labels = reference.obs[label].astype(str).values
    cell_types = sorted(set(labels))
    signatures = np.zeros((normalised.n_vars, len(cell_types)), dtype=np.float64)
    for k, cell_type in enumerate(cell_types):
        signatures[:, k] = expression[labels == cell_type].mean(0)
    return pd.DataFrame(signatures, index=normalised.var_names, columns=cell_types)


def signature_genes(signatures, genes, n_top_genes, markers) -> list[str]:
    """Genes of the NNLS fit: the `n_top_genes` signature genes that vary most across cell types, plus the markers.

    Only genes that are in the signatures and in `genes` (the genes of the section) are used; variation is the
    coefficient of variation of a gene across the cell types.
    """
    shared = sorted(set(signatures.index) & set(genes))
    shared_signatures = signatures.loc[shared]
    mean = shared_signatures.mean(1) + 1e-9
    cv = shared_signatures.std(1) / mean
    informative = set(cv.sort_values(ascending=False).head(n_top_genes).index) | (set(markers) & set(shared))
    return sorted(informative & set(shared))


def nnls_deconvolution(adata, signatures, n_top_genes, markers, key_added="NNLS") -> None:
    """Deconvolve the spots of one section by non-negative least squares.

    The expression of every spot (CP10k, linear scale) is fitted as a non-negative mixture of the cell-type
    signatures (``scipy.optimize.nnls``) on the genes of :func:`signature_genes`. The weight of a cell type times
    the total of its signature is its share of the mRNA of the spot; these shares are normalised to sum to 1.

    Parameters
    ----------
    adata
        Spots with raw counts in ``X``.
    signatures
        Genes x cell types, mean CP10k per type.
    n_top_genes
        Number of most variable signature genes used.
    markers
        Marker genes that are always used.
    key_added
        Name of the result.

    Returns
    -------
    Nothing; adds ``adata.obsm[key_added]`` (spots x cell types, mRNA fractions) and ``adata.uns[key_added]``
    with the genes used (``genes``) and the root mean square of the residual norms of the fits (``rmse``).
    """
    import anndata as ad
    import scanpy as sc
    from scipy.optimize import nnls

    cell_types = list(signatures.columns)
    normalised = ad.AnnData(X=adata.X.copy(), var=pd.DataFrame(index=adata.var_names))
    sc.pp.normalize_total(normalised, target_sum=1e4)
    genes = signature_genes(signatures, adata.var_names, n_top_genes, markers)
    S = signatures.loc[genes, cell_types].to_numpy(dtype=np.float64)
    Y = _dense(normalised[:, genes].X).astype(np.float64)
    weights = np.zeros((adata.n_obs, len(cell_types)))
    residual = np.zeros(adata.n_obs)
    for j in range(adata.n_obs):
        weights[j], residual[j] = nnls(S, Y[j])
    mrna = weights * S.sum(0)[None, :]
    mrna_sums = mrna.sum(1, keepdims=True)
    mrna_sums[mrna_sums == 0] = 1.0
    adata.obsm[key_added] = pd.DataFrame(mrna / mrna_sums, index=adata.obs_names, columns=cell_types)
    adata.uns[key_added] = {"genes": genes, "rmse": float(np.sqrt((residual**2).mean()))}


# ---------------------------------------------------------------------------------------------------------------
# spots of known composition from Visium HD


def _tile_ids(x, y, xmin, ymin, nx, ny, tile_um):
    """Linear tile id (ix * ny + iy) of each coordinate."""
    ix = np.clip(((x - xmin) / tile_um).astype(np.int64), 0, nx - 1)
    iy = np.clip(((y - ymin) / tile_um).astype(np.int64), 0, ny - 1)
    return ix * ny + iy


def simulate_spots(
    transcripts,
    pixels,
    genes,
    bounds,
    factor_order,
    tile_um,
    min_counts,
    min_pixels,
    min_cells,
    chunksize=8_000_000,
    log=None,
):
    """Aggregate one Visium HD section into spots of known composition.

    The section is cut into square tiles of `tile_um`. The counts of a simulated spot are the transcripts of its
    tile summed per gene; its known composition is the share of the decoded 2 um bins of the tile per Level-3
    type (an area fraction).

    Parameters
    ----------
    transcripts
        Path of the transcripts of the section (``X``, ``Y`` in um, ``gene``, ``Count``), read in blocks.
    pixels
        Path of the FICTURE pixel decode (``X``, ``Y`` in um, ``K1`` = index into `factor_order`), read in blocks.
    genes
        Genes of the section; transcripts of other genes are left out.
    bounds
        ``xmin``, ``xmax``, ``ymin``, ``ymax`` of the section in um.
    factor_order
        Level-3 types in factor order.
    tile_um
        Side of a tile.
    min_counts, min_pixels
        Counts and decoded bins a tile needs to be kept.
    min_cells
        Simulated spots a gene needs to be kept.

    Returns
    -------
    AnnData of the simulated spots: counts in ``X``, tile centres in ``obsm["spatial"]`` (um), grid position in
    ``obs["array_row"]`` / ``obs["array_col"]``, ``obs["total_counts"]``, ``obs["n_bins"]``; the known
    composition in ``obsm["truth"]`` and the bins per type in ``obsm["truth_pixels"]``; grid size and the number
    of transcript rows and decoded bins read in ``uns["grid"]``.
    """
    import anndata as ad
    import scipy.sparse as sp

    xmin, xmax, ymin, ymax = (float(bounds[key]) for key in ("xmin", "xmax", "ymin", "ymax"))
    nx = int(np.floor((xmax - xmin) / tile_um)) + 1
    ny = int(np.floor((ymax - ymin) / tile_um)) + 1
    n_tiles, n_factors = nx * ny, len(factor_order)
    genes = list(genes)
    gene_index = {gene: i for i, gene in enumerate(genes)}

    factor_counts = np.zeros((n_tiles, n_factors), dtype=np.int64)
    n_pixels = 0
    for chunk in pd.read_csv(
        pixels,
        sep="\t",
        usecols=["X", "Y", "K1"],
        dtype={"X": np.float64, "Y": np.float64, "K1": np.int16},
        chunksize=chunksize,
    ):
        tile = _tile_ids(chunk["X"].values, chunk["Y"].values, xmin, ymin, nx, ny, tile_um)
        counted = np.bincount(tile * n_factors + chunk["K1"].values.astype(np.int64), minlength=n_tiles * n_factors)
        factor_counts += counted.reshape(n_tiles, n_factors)
        n_pixels += len(chunk)
        if log is not None:
            log(f"  pixels {n_pixels:,}")

    counts = sp.csr_matrix((n_tiles, len(genes)), dtype=np.int64)
    n_transcripts = 0
    for chunk in pd.read_csv(
        transcripts,
        sep="\t",
        usecols=["X", "Y", "gene", "Count"],
        dtype={"X": np.float64, "Y": np.float64, "gene": str, "Count": np.int32},
        chunksize=chunksize,
    ):
        column = chunk["gene"].map(gene_index)
        known = column.notna().values
        if not known.all():
            chunk, column = chunk[known], column[known]
        tile = _tile_ids(chunk["X"].values, chunk["Y"].values, xmin, ymin, nx, ny, tile_um)
        block = (chunk["Count"].values.astype(np.int64), (tile, column.values.astype(np.int64)))
        counts = counts + sp.coo_matrix(block, shape=(n_tiles, len(genes))).tocsr()
        n_transcripts += len(chunk)
        if log is not None:
            log(f"  transcripts {n_transcripts:,}")

    total_counts = np.asarray(counts.sum(1)).ravel()
    total_pixels = factor_counts.sum(1)
    keep = np.where((total_counts >= min_counts) & (total_pixels >= min_pixels))[0]
    spot_counts, spot_factors = counts[keep], factor_counts[keep]
    ix, iy = keep // ny, keep % ny  # tile id = ix * ny + iy
    barcodes = [f"sim_{a}_{b}" for a, b in zip(ix, iy)]
    gene_keep = np.where(np.asarray((spot_counts > 0).sum(0)).ravel() >= min_cells)[0]
    spot_counts = spot_counts[:, gene_keep]

    obs = pd.DataFrame(index=barcodes)
    obs["array_row"] = iy.astype(int)
    obs["array_col"] = ix.astype(int)
    obs["total_counts"] = np.asarray(spot_counts.sum(1)).ravel()
    obs["n_bins"] = total_pixels[keep]
    spots = ad.AnnData(
        X=sp.csr_matrix(spot_counts).astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=pd.Index([genes[i] for i in gene_keep], name=None)),
    )
    spots.obsm["spatial"] = np.c_[xmin + (ix + 0.5) * tile_um, ymin + (iy + 0.5) * tile_um].astype(float)
    spots.var_names_make_unique()
    spots.obsm["truth"] = pd.DataFrame(
        spot_factors / spot_factors.sum(1, keepdims=True), index=barcodes, columns=factor_order
    )
    spots.obsm["truth_pixels"] = pd.DataFrame(spot_factors, index=barcodes, columns=factor_order)
    spots.uns["grid"] = {
        "nx": nx,
        "ny": ny,
        "n_tiles": n_tiles,
        "n_genes": len(genes),
        "n_transcripts": int(n_transcripts),
        "n_decoded_pixels": int(n_pixels),
    }
    return spots


def subsample_spots(adata, max_spots, rng):
    """At most `max_spots` spots of a section, drawn without replacement from the generator `rng`, in spot order."""
    if adata.n_obs <= max_spots:
        return adata
    return adata[np.sort(rng.choice(adata.n_obs, max_spots, replace=False))].copy()


# ---------------------------------------------------------------------------------------------------------------
# scoring against the known composition


def read_proportions(path, cell_types, normalise=True) -> pd.DataFrame:
    """Read the per-spot proportions of one method on one section (spots x `cell_types`).

    Cell types a method did not report are 0. With `normalise` every spot is scaled to sum to 1 (spots without
    any proportion stay 0).
    """
    table = pd.read_csv(path, index_col=0)
    proportions = table[[cell_type for cell_type in cell_types if cell_type in table.columns]].copy()
    for cell_type in cell_types:
        if cell_type not in proportions.columns:
            proportions[cell_type] = 0.0
    proportions = proportions[list(cell_types)]
    if not normalise:
        return proportions
    row_sums = proportions.sum(1).replace(0, np.nan)
    return proportions.div(row_sums, axis=0).fillna(0.0)


def collect_proportions(tables):
    """Put the per-spot proportions of several methods on one section into one AnnData.

    Parameters
    ----------
    tables
        Mapping of method -> spots x cell types (:func:`read_proportions`).

    Returns
    -------
    AnnData without expression: the spots of the first table as ``obs`` and one table per method in
    ``obsm[<method>]`` (NaN for spots a method has no result for).
    """
    import anndata as ad

    spots = next(iter(tables.values())).index if tables else pd.Index([])
    adata = ad.AnnData(obs=pd.DataFrame(index=spots))
    for method, table in tables.items():
        adata.obsm[method] = table.reindex(adata.obs_names)
    return adata


def _shared_spots(adata, key_a, key_b):
    """The tables of two methods on the spots both have values for."""
    a, b = adata.obsm[key_a], adata.obsm[key_b]
    complete = a.notna().all(axis=1).values & b.notna().all(axis=1).values
    return (a, b) if complete.all() else (a[complete], b[complete])


def score_proportions(truth, estimate) -> tuple[dict, pd.DataFrame]:
    """Score the estimated proportions of the spots of one section against their known composition.

    Parameters
    ----------
    truth, estimate
        Spots x cell types, the same spots and cell types in both.

    Returns
    -------
    overall
        ``n_spots``; ``rmse`` and ``mae`` over all spots and cell types; ``jsd``, the mean Jensen-Shannon
        distance between the estimated and the known composition of a spot; ``pearson`` and ``spearman``, the
        mean over the cell types of the correlation across spots.
    per_type
        One row per cell type: ``rmse``, ``mae``, ``pearson``, ``spearman`` (NaN for a cell type that is constant
        in either table), ``truth_mean``, ``pred_mean``.
    """
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import pearsonr, spearmanr

    estimate = estimate[truth.columns]
    rows = {}
    for cell_type in truth.columns:
        x, y = truth[cell_type].values, estimate[cell_type].values
        varies = x.std() > 1e-9 and y.std() > 1e-9
        rows[cell_type] = {
            "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
            "mae": float(np.mean(np.abs(x - y))),
            "pearson": float(pearsonr(x, y)[0]) if varies else np.nan,
            "spearman": float(spearmanr(x, y)[0]) if varies else np.nan,
            "truth_mean": float(x.mean()),
            "pred_mean": float(y.mean()),
        }
    per_type = pd.DataFrame(rows).T
    X, Y = truth.values, estimate.values
    pearsons = [row["pearson"] for row in rows.values() if np.isfinite(row["pearson"])]
    spearmans = [row["spearman"] for row in rows.values() if np.isfinite(row["spearman"])]
    overall = {
        "n_spots": truth.shape[0],
        "rmse": float(np.sqrt(np.mean((X - Y) ** 2))),
        "mae": float(np.mean(np.abs(X - Y))),
        "jsd": float(np.nanmean(jensenshannon(X + 1e-12, Y + 1e-12, axis=1))),
        "pearson": float(np.mean(pearsons)) if pearsons else np.nan,
        "spearman": float(np.mean(spearmans)) if spearmans else np.nan,
    }
    return overall, per_type


def score_methods(sections, methods, regions, truth_key="truth", min_spots=5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every method on every simulated section (:func:`score_proportions`).

    Parameters
    ----------
    sections
        Mapping of section -> AnnData with the known composition in ``obsm[truth_key]`` and the proportions of
        each method in ``obsm[<method>]`` (methods without a result on a section are skipped).
    methods
        Methods, in the order of the rows.
    regions
        Mapping of section -> position in the wall (``epi``, ``myo``, ``endo``).

    Returns
    -------
    overall
        One row per method and section: ``method``, ``slide``, ``region`` and the overall scores.
    per_type
        One row per method, section and cell type: ``method``, ``slide``, ``region``, ``celltype`` and the scores
        of the cell type.
    """
    overall_rows, type_rows = [], []
    for method in methods:
        for slide, adata in sections.items():
            if method not in adata.obsm:
                continue
            truth, estimate = _shared_spots(adata, truth_key, method)
            if truth.shape[0] < min_spots:
                continue
            overall, per_type = score_proportions(truth, estimate)
            key = {"method": method, "slide": slide, "region": regions[slide]}
            overall_rows.append({**key, **overall})
            for cell_type, row in per_type.iterrows():
                type_rows.append({**key, "celltype": cell_type, **row.to_dict()})
    return pd.DataFrame(overall_rows), pd.DataFrame(type_rows)


def leaderboard(overall) -> pd.DataFrame:
    """Rank the methods on the simulated sections.

    The scores of :func:`score_methods` are averaged over the sections; ``score`` is the mean of the ranks by
    RMSE, Jensen-Shannon distance and Pearson r (1 = best), and sorts the table.
    """
    board = (
        overall.groupby("method")
        .agg(
            rmse=("rmse", "mean"),
            mae=("mae", "mean"),
            jsd=("jsd", "mean"),
            pearson=("pearson", "mean"),
            spearman=("spearman", "mean"),
            n_slides=("slide", "nunique"),
        )
        .reset_index()
    )
    board["rank_rmse"] = board["rmse"].rank()
    board["rank_jsd"] = board["jsd"].rank()
    board["rank_pearson"] = board["pearson"].rank(ascending=False)
    board["score"] = board[["rank_rmse", "rank_jsd", "rank_pearson"]].mean(1)
    return board.sort_values("score")


def celltype_scores(per_type, cell_types) -> dict[str, pd.DataFrame]:
    """Scores per cell type and method, averaged over the simulated sections.

    Returns
    -------
    ``rmse`` and ``pearson`` (cell types x methods) and ``best``: per cell type the method with the lowest RMSE
    and the one with the highest Pearson r, with the mean known proportion of the type.
    """
    rmse = per_type.groupby(["celltype", "method"])["rmse"].mean().unstack().reindex(cell_types)
    pearson = per_type.groupby(["celltype", "method"])["pearson"].mean().unstack().reindex(cell_types)
    best = pd.DataFrame(
        {
            "best_rmse_method": rmse.idxmin(1),
            "best_rmse": rmse.min(1),
            "best_corr_method": pearson.idxmax(1),
            "best_corr": pearson.max(1),
            "truth_mean": per_type.groupby("celltype")["truth_mean"].mean().reindex(cell_types),
        }
    )
    return {"rmse": rmse, "pearson": pearson, "best": best}


def region_scores(overall) -> pd.DataFrame:
    """RMSE, Jensen-Shannon distance and Pearson r per method and position in the wall (mean over sections)."""
    return (
        overall.groupby(["region", "method"])
        .agg(rmse=("rmse", "mean"), jsd=("jsd", "mean"), pearson=("pearson", "mean"))
        .reset_index()
    )


# ---------------------------------------------------------------------------------------------------------------
# comparison of the methods on the tissue sections


def section_chambers(sections, atrial_type, ventricular_type, method="NNLS", atrial_sections=()) -> dict[str, str]:
    """Call every section atrial or ventricular from its deconvolution.

    A section is atrial when the mean proportion of atrial cardiomyocytes of `method` exceeds that of ventricular
    cardiomyocytes; sections without a result of `method` are atrial when listed in `atrial_sections`.
    """
    chambers = {}
    for slide, adata in sections.items():
        if method in adata.obsm:
            proportions = adata.obsm[method]
            atrial = proportions[atrial_type].mean() > proportions[ventricular_type].mean()
        else:
            atrial = slide in atrial_sections
        chambers[slide] = "atrial" if atrial else "ventricular"
    return chambers


def method_concordance(sections, methods, min_spots=5) -> dict:
    """Agreement between the per-spot proportions of every pair of methods.

    For each pair, the Pearson correlation across the spots of a section is computed per cell type and averaged
    over cell types and sections; the Jensen-Shannon distance between the two estimates of a cell type across
    the spots is averaged in the same way.

    Returns
    -------
    ``pearson`` and ``jsd`` (methods x methods) and ``per_type``, the mean correlation of each cell type over
    all pairs of different methods.
    """
    from scipy.spatial.distance import jensenshannon

    cell_types = None
    pearson = pd.DataFrame(np.nan, index=methods, columns=methods)
    jsd = pd.DataFrame(np.nan, index=methods, columns=methods)
    per_type_r = {}
    for method1 in methods:
        for method2 in methods:
            correlations, divergences = [], []
            for adata in sections.values():
                if method1 not in adata.obsm or method2 not in adata.obsm:
                    continue
                a, b = _shared_spots(adata, method1, method2)
                if a.shape[0] < min_spots:
                    continue
                if cell_types is None:
                    cell_types = list(a.columns)
                    per_type_r = {cell_type: [] for cell_type in cell_types}
                for cell_type in cell_types:
                    x, y = a[cell_type].values, b[cell_type].values
                    if x.std() > 1e-9 and y.std() > 1e-9:
                        r = np.corrcoef(x, y)[0, 1]
                        correlations.append(r)
                        if method1 != method2:
                            per_type_r[cell_type].append(r)
                divergences.append(np.nanmean(jensenshannon(a.values.T + 1e-12, b.values.T + 1e-12, axis=0)))
            if correlations:
                pearson.loc[method1, method2] = np.nanmean(correlations)
            if divergences:
                jsd.loc[method1, method2] = np.nanmean(divergences)
    per_type = pd.Series({cell_type: (np.nanmean(r) if r else np.nan) for cell_type, r in per_type_r.items()})
    return {"pearson": pearson, "jsd": jsd, "per_type": per_type}


def background_floor(
    sections, methods, chambers, cell_types, min_ventricular, min_ratio, max_cv
) -> tuple[pd.DataFrame, pd.Series]:
    """Find cell types that a method reports at a uniform background level on ventricular sections.

    A cell type forms such a floor when its mean proportion on the ventricular sections is above
    `min_ventricular`, more than `min_ratio` times the mean on the atrial sections, and varies little between
    the ventricular sections (coefficient of variation below `max_cv`).

    Returns
    -------
    One row per method and cell type (``atrial_mean``, ``ventricular_mean``, ``vent_over_atrial``, ``vent_CV``,
    ``uniform_floor``) and the number of floor types per method.
    """
    rows = []
    for method in methods:
        available = sorted(slide for slide, adata in sections.items() if method in adata.obsm)
        for cell_type in cell_types:
            means = {slide: sections[slide].obsm[method][cell_type].mean() for slide in available}
            atrial = [means[slide] for slide in available if chambers[slide] == "atrial"]
            ventricular = [means[slide] for slide in available if chambers[slide] == "ventricular"]
            atrial_mean = float(np.mean(atrial)) if atrial else np.nan
            ventricular_mean = float(np.mean(ventricular)) if ventricular else np.nan
            ventricular_cv = (
                float(np.std(ventricular) / (np.mean(ventricular) + 1e-9)) if len(ventricular) > 1 else np.nan
            )
            ratio = ventricular_mean / (atrial_mean + 1e-6) if atrial_mean == atrial_mean else np.nan
            rows.append(
                {
                    "method": method,
                    "cell_type": cell_type,
                    "atrial_mean": atrial_mean,
                    "ventricular_mean": ventricular_mean,
                    "vent_over_atrial": ratio,
                    "vent_CV": ventricular_cv,
                    "uniform_floor": bool(
                        (ventricular_mean > min_ventricular) and (ratio > min_ratio) and (ventricular_cv < max_cv)
                    ),
                }
            )
    floor = pd.DataFrame(rows)
    return floor, floor.groupby("method")["uniform_floor"].sum().rename("n_floor_types_of_4")


def compartment_composition(sections, methods, compartments, chambers) -> pd.DataFrame:
    """Mean proportion of every compartment per method and section.

    Parameters
    ----------
    sections
        Mapping of section -> AnnData with the proportions of each method in ``obsm[<method>]``.
    methods
        Methods, in the order of the rows.
    compartments
        Mapping of compartment -> Level-3 types; the proportions of the types are summed per spot.
    chambers
        Mapping of section -> ``atrial`` / ``ventricular`` (:func:`section_chambers`).

    Returns
    -------
    One row per method and section: ``method``, ``slide``, ``chamber`` and the mean proportion of each
    compartment over the spots. The mean over the sections of a method is the bar of ED6 h.
    """
    rows = []
    for method in methods:
        for slide in sorted(slide for slide, adata in sections.items() if method in adata.obsm):
            proportions = sections[slide].obsm[method]
            row = {"method": method, "slide": slide, "chamber": chambers[slide]}
            for compartment, members in compartments.items():
                present = [cell_type for cell_type in members if cell_type in proportions.columns]
                row[compartment] = float(proportions[present].sum(1).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def marker_support(sections, methods, cell_types, scores_key="marker_scores", min_spots=10) -> pd.DataFrame:
    """Agreement of the estimated proportion of a cell type with the expression of its marker genes.

    Spearman correlation across the spots of a section between the proportion and the marker score in
    ``obsm[scores_key]``, as the median over the sections, per method and cell type.
    """
    from scipy.stats import spearmanr

    rows = []
    for method in methods:
        for cell_type in cell_types:
            correlations = []
            for slide in sorted(slide for slide, adata in sections.items() if method in adata.obsm):
                adata = sections[slide]
                if scores_key not in adata.obsm or cell_type not in adata.obsm[scores_key].columns:
                    continue
                a, b = adata.obsm[method][cell_type], adata.obsm[scores_key][cell_type]
                if a.shape[0] < min_spots or a.std() < 1e-9 or b.std() < 1e-9:
                    continue
                correlations.append(spearmanr(a.values, b.values).correlation)
            rows.append(
                {
                    "method": method,
                    "cell_type": cell_type,
                    "median_spearman": float(np.nanmedian(correlations)) if correlations else np.nan,
                    "n_slides": len(correlations),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------
# Fig 3 c: chamber markers in cardiomyocyte-dominant spots

# cell2location types of Kanemaru et al. 2023 that are named, not prefixed
KANEMARU_NAMED = {
    "SAN_P_cell": "Conduction",
    "AVN_bundle_cell": "Conduction",
    "AVN_P_cell": "Conduction",
    "B": "B",
    "B_plasma": "Plasma",
    "Mast": "Mast",
    "Meso": "Mesothelial",
    "Neut": "Neutrophil",
    "DC": "DC",
    "ILC": "ILC",
    "MAIT-like": "T/NK",
    "gdT": "T/NK",
    "T/NK_cycling": "T/NK",
}
KANEMARU_PREFIX = {
    "aCM": "Atrial-CM",
    "vCM": "Ventricular-CM",
    "EC": "Endothelial",  # with EC10_CMC-like, EC9_FB-like
    "FB": "Fibroblast",
    "PC": "Pericyte",  # PC1_vent to PC4_CMC-like
    "SMC": "SMC",
    "Adip": "Adipocyte",
    "NC": "Neural",
}


def kanemaru_lineage(cell_type: str) -> str:
    """Coarse lineage of a cell2location type of Kanemaru et al. 2023, comparable to the Level-3 types.

    Only ``aCM*`` and ``vCM*`` are cardiomyocytes; pacemaker and conduction cells and the cardiomyocyte-like
    pericyte and endothelial states are not.
    """
    for prefix in ("aCM", "vCM"):
        if cell_type.startswith(prefix):
            return KANEMARU_PREFIX[prefix]
    if cell_type in ("SAN_P_cell", "AVN_bundle_cell", "AVN_P_cell"):
        return KANEMARU_NAMED[cell_type]
    for prefix in ("EC", "FB", "PC", "SMC", "Adip", "NC"):
        if cell_type.startswith(prefix):
            return KANEMARU_PREFIX[prefix]
    if cell_type in KANEMARU_NAMED:
        return KANEMARU_NAMED[cell_type]
    if ("Mo" in cell_type) or ("MP" in cell_type):
        return "Myeloid"  # CD14+Mo, CD16+Mo, MoMP, LYVE1+ macrophages
    if cell_type.startswith(("CD4", "CD8", "NK")):
        return "T/NK"
    return "OTHER::" + cell_type


def lineage_proportions(proportions) -> pd.DataFrame:
    """Sum the cell2location proportions of Kanemaru et al. 2023 (spots x fine types) to coarse lineages."""
    lineages = {}
    for cell_type in proportions.columns:
        lineages.setdefault(kanemaru_lineage(cell_type), []).append(cell_type)
    unknown = [lineage for lineage in lineages if lineage.startswith("OTHER::")]
    assert not unknown, unknown
    return pd.DataFrame(
        {lineage: proportions[cell_types].sum(axis=1, skipna=True).values for lineage, cell_types in lineages.items()},
        index=proportions.index,
    )


def cardiomyocyte_dominant(adata, cardiomyocytes, proportions=None, dominant="dominant_L3") -> None:
    """Mark the spots whose top deconvolved cell type is a cardiomyocyte.

    Parameters
    ----------
    adata
        Spots of one section.
    cardiomyocytes
        Cell types that count as cardiomyocytes.
    proportions
        Spots x cell types. Without it the top type is read from ``adata.obs[dominant]`` and the proportions of
        the cardiomyocyte types from the obs columns of their names.

    Returns
    -------
    Nothing; adds ``adata.obs["cm_dominant"]`` and ``adata.obs["cm_proportion"]``, the summed proportion of the
    cardiomyocyte types.
    """
    if proportions is None:
        top = adata.obs[dominant].astype(str).values
        cm_proportion = adata.obs[list(cardiomyocytes)].sum(axis=1).values
    else:
        top = np.asarray(proportions.columns[np.asarray(proportions.values).argmax(axis=1)])
        present = [cell_type for cell_type in cardiomyocytes if cell_type in proportions.columns]
        cm_proportion = proportions[present].sum(axis=1).values
    adata.obs["cm_dominant"] = np.isin(top, list(cardiomyocytes))
    adata.obs["cm_proportion"] = cm_proportion


def chamber_gene_panel(concordance, extra=()) -> pd.Series:
    """Chamber genes of Fig 3 c in plot order, with their direction in the atlas.

    The cardiomyocyte genes of the panel (``cm_class`` and ``in_panel`` of the gene table): atrial genes, then
    the atrial genes `extra` that are shown with the panel, then ventricular genes.

    Returns
    -------
    Series gene -> ``atria`` / ``ventricle``.
    """
    panel = concordance[(concordance.cm_class) & (concordance.in_panel)]
    atrial = panel[panel.snuc_dir == "atria"].gene.tolist() + list(extra)
    ventricular = panel[panel.snuc_dir == "ventricle"].gene.tolist()
    direction = dict.fromkeys(atrial, "atria")
    direction.update(dict.fromkeys(ventricular, "ventricle"))
    return pd.Series(direction, name="snuc_dir").rename_axis("gene")


def chamber_sums(adata, genes, mask="cm_dominant") -> pd.DataFrame:
    """Sum the expression of the panel genes over the spots of one section.

    Parameters
    ----------
    adata
        Spots with log-normalised expression in ``X``.
    genes
        Panel genes; genes the section does not measure get no sum.
    mask
        Boolean obs column of the selected spots.

    Returns
    -------
    One row per gene: ``n_all`` and ``n_cm`` (spots of the section, selected spots) and ``sum_all`` and
    ``sum_cm``, the summed expression over them (NaN for a gene that is not measured).
    """
    from scipy.sparse import issparse

    selected = adata.obs[mask].to_numpy(dtype=bool)
    column = {gene: i for i, gene in enumerate(adata.var_names)}
    rows = []
    for gene in genes:
        row = {"gene": gene, "n_all": int(selected.shape[0]), "n_cm": int(selected.sum())}
        if gene in column:
            values = adata.X[:, column[gene]]
            values = np.asarray(values.todense()).ravel() if issparse(values) else np.asarray(values).ravel()
            values = values.astype(np.float64)
            row["sum_all"], row["sum_cm"] = float(values.sum()), float(values[selected].sum())
        else:
            row["sum_all"], row["sum_cm"] = np.nan, np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def region_means(sums, regions, which="cm") -> pd.DataFrame:
    """Mean expression of the panel genes per chamber region, pooled over the sections of all cohorts.

    Parameters
    ----------
    sums
        Tables of :func:`chamber_sums` concatenated, with the columns ``cohort``, ``region`` and ``slide``.
    regions
        Regions, in the order of the columns.
    which
        ``"cm"`` for the selected spots, ``"all"`` for all spots.

    Returns
    -------
    Genes x regions: summed expression divided by the number of spots of the region (NaN where no section of
    the region measures the gene).
    """
    genes = list(dict.fromkeys(sums["gene"]))
    cohorts = list(dict.fromkeys(sums["cohort"]))
    means = pd.DataFrame(index=genes, columns=list(regions), dtype=float)
    for region in regions:
        of_region = sums[sums["region"] == region]
        total = int(of_region.drop_duplicates(["cohort", "slide"])[f"n_{which}"].sum())
        for gene in genes:
            rows = of_region[(of_region["gene"] == gene) & of_region[f"sum_{which}"].notna()]
            if rows.empty or not total:
                means.loc[gene, region] = np.nan
                continue
            pooled = 0
            for cohort in cohorts:
                cohort_sum = 0.0
                for value in rows.loc[rows["cohort"] == cohort, f"sum_{which}"]:
                    cohort_sum += float(value)
                pooled = pooled + cohort_sum
            means.loc[gene, region] = pooled / total
    return means


def region_zscores(means) -> pd.DataFrame:
    """Z-score of every gene across the regions (rows of :func:`region_means`)."""
    return means.sub(means.mean(1), axis=0).div(means.std(1).replace(0, np.nan), axis=0)


def region_spot_counts(sums, regions, cohorts) -> pd.DataFrame:
    """Spots and cardiomyocyte-dominant spots per chamber region, per cohort and combined.

    Parameters
    ----------
    sums
        Tables of :func:`chamber_sums` concatenated, with the columns ``cohort``, ``region`` and ``slide``.
    regions
        Regions, in the order of the rows.
    cohorts
        Mapping of cohort (as in ``sums``) -> name used in the column names.
    """
    slides = sums.drop_duplicates(["cohort", "slide"])
    rows = []
    for region in regions:
        row = {"region": region}
        total = selected = 0
        for cohort, name in cohorts.items():
            of_cohort = slides[(slides["region"] == region) & (slides["cohort"] == cohort)]
            n_all, n_cm = int(of_cohort["n_all"].sum()), int(of_cohort["n_cm"].sum())
            row[f"{name}_total"], row[f"{name}_cm"] = n_all, n_cm
            row[f"{name}_frac"] = n_cm / n_all if n_all else np.nan
            total, selected = total + n_all, selected + n_cm
        row["combined_total"], row["combined_cm"] = total, selected
        row["combined_frac"] = selected / total if total else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------
# ED8 b-c: atrial-pericyte signature


def score_signature(adata, key_added="score") -> None:
    """Score a gene signature per spot.

    Every gene of `adata` is z-scored across all spots, so that no highly expressed gene dominates, and the
    z-scores are averaged per spot.

    Returns
    -------
    Nothing; adds ``adata.layers["z"]`` and ``adata.obs[key_added]``.
    """
    # one contiguous column per gene, so that the sums over spots run in the same order for any input layout
    values = np.asfortranarray(np.asarray(_dense(adata.X), dtype=np.float64))
    expression = pd.DataFrame(values, columns=adata.var_names, index=adata.obs_names)
    z = (expression - expression.mean(0)) / expression.std(0).replace(0, np.nan)
    adata.layers["z"] = z.to_numpy()
    adata.obs[key_added] = z.mean(1)


def abundance_groups(obs, abundance, groupby, quantiles, labels, key_added) -> None:
    """Split the spots into a low and a high group of a cell-type abundance, within each group of `groupby`.

    Spots up to the lower quantile get ``labels[0]``, spots from the upper quantile on ``labels[1]``; the spots
    in between stay unlabelled.

    Returns
    -------
    Nothing; adds ``obs[key_added]``.
    """
    obs[key_added] = pd.Series(np.nan, index=obs.index, dtype=object)
    for _, group in obs.groupby(groupby, observed=True):
        low, high = group[abundance].quantile(list(quantiles))
        obs.loc[group.index[group[abundance] <= low], key_added] = labels[0]
        obs.loc[group.index[group[abundance] >= high], key_added] = labels[1]


def signature_by_chamber(adata, chambers, sort_by, chamber_key="chamber") -> pd.DataFrame:
    """Mean z-scored expression of the signature genes per chamber (genes x chambers).

    Uses ``adata.layers["z"]`` of :func:`score_signature`; the genes are sorted by their mean over the chambers
    `sort_by`, highest first.
    """
    z = pd.DataFrame(np.asfortranarray(adata.layers["z"]), columns=adata.var_names, index=adata.obs_names)
    means = pd.DataFrame(
        {chamber: z.loc[adata.obs_names[(adata.obs[chamber_key] == chamber).values]].mean(0) for chamber in chambers}
    )
    return means.loc[means[list(sort_by)].mean(1).sort_values(ascending=False).index]


def _median(values) -> float:
    """Median, NaN for an empty group."""
    return float(np.median(values)) if len(values) else float("nan")


def compare_scores(reference, group) -> dict:
    """Compare the scores of `group` with those of `reference` (two-sided Mann-Whitney U test).

    Returns
    -------
    ``n`` and ``median`` of both groups (``*_group``, ``*_reference``), ``delta_median`` (group minus reference),
    ``mwu_p`` and ``mwu_U`` (NaN when a group is empty).
    """
    from scipy import stats

    if min(len(reference), len(group)):
        test = stats.mannwhitneyu(group, reference, alternative="two-sided")
        p, u = float(test.pvalue), float(test.statistic)
    else:
        p, u = float("nan"), float("nan")
    return {
        "n_group": len(group),
        "n_reference": len(reference),
        "median_group": _median(group),
        "median_reference": _median(reference),
        "delta_median": float(_median(group) - _median(reference)),
        "mwu_p": p,
        "mwu_U": u,
    }


def paired_donor_test(spots, value="score", groups=("atria", "ventricle"), group_key="region", min_donors=3):
    """Compare the per-donor mean of `value` between two groups of spots (Wilcoxon signed-rank test).

    Spots of one donor are not independent, so the donor is the unit here.

    Returns
    -------
    None when fewer than `min_donors` donors have spots in both groups; otherwise ``n_donors``, the median of
    the donor means per group, ``wilcoxon_p`` and the donor means (``donors``).
    """
    from scipy import stats

    donor_means = spots.groupby(["donor", group_key])[value].mean().unstack()
    if not (donor_means.shape[1] == 2 and len(donor_means.dropna())):
        return None
    both = donor_means.dropna()
    if len(both) < min_donors:
        return None
    test = stats.wilcoxon(both[groups[0]], both[groups[1]])
    return {
        "n_donors": int(len(both)),
        f"median_{groups[0]}": float(both[groups[0]].median()),
        f"median_{groups[1]}": float(both[groups[1]].median()),
        "wilcoxon_p": float(test.pvalue),
        "donors": donor_means.round(6).reset_index().to_dict(orient="records"),
    }
