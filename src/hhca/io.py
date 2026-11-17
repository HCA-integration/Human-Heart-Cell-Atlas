"""Reading and writing the on-disk formats used across the pipelines."""

from __future__ import annotations

import gzip
import json
import os

import numpy as np
import pandas as pd

DECODE_DTYPES = {"X": np.float32, "Y": np.float32, "K1": np.int16, "P1": np.float32}


def decode(value):
    """A bytes value as str; anything else is returned unchanged."""
    return value.decode("utf-8", "replace") if isinstance(value, (bytes, bytearray)) else value


def read_categorical(group, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Categories (str) and integer codes of a categorical column of an open h5ad group such as ``f["obs"]``."""
    node = group[column]
    categories = np.array([decode(category) for category in node["categories"][:]], dtype=object).astype(str)
    return categories, node["codes"][:]


def write_mtx_gz(matrix, path: str) -> None:
    """Write a sparse matrix of integer counts as gzipped MatrixMarket (`path` ends with ``.gz``)."""
    import scipy.io
    import scipy.sparse

    counts = scipy.sparse.csr_matrix(matrix).astype(np.int32)
    plain = path[:-3] if path.endswith(".gz") else path
    scipy.io.mmwrite(plain, counts, field="integer")
    with open(plain, "rb") as source, gzip.open(path, "wb") as target:
        target.writelines(source)
    os.remove(plain)


def read_decode(path, columns=("X", "Y", "K1")) -> pd.DataFrame:
    """Read the FICTURE pixel decode of one section (``.pixel.tsv.gz``).

    One row per decode pixel: ``X``, ``Y`` (um), ``K1`` (code of the most probable Level-3 type in the factor
    order) and ``P1`` (its probability).
    """
    columns = list(columns)
    return pd.read_csv(path, sep="\t", usecols=columns, dtype={column: DECODE_DTYPES[column] for column in columns})


def read_visium_hd(directory):
    """Read the Space Ranger 2 um bins of one section (``binned_outputs/square_002um``) as AnnData.

    The feature names are kept as in the matrix (they can repeat); ``obsm["spatial"]`` holds the bin centres in
    um and ``uns["microns_per_pixel"]`` the scale they were converted with.
    """
    import scanpy as sc

    with open(f"{directory}/spatial/scalefactors_json.json") as handle:
        microns_per_pixel = json.load(handle)["microns_per_pixel"]
    adata = sc.read_10x_h5(f"{directory}/filtered_feature_bc_matrix.h5")
    positions = pd.read_parquet(f"{directory}/spatial/tissue_positions.parquet").set_index("barcode")
    positions = positions.reindex(adata.obs_names)
    adata.obsm["spatial"] = np.column_stack(
        [
            (positions["pxl_col_in_fullres"].values * microns_per_pixel).astype(np.float64),
            (positions["pxl_row_in_fullres"].values * microns_per_pixel).astype(np.float64),
        ]
    )
    adata.uns["microns_per_pixel"] = microns_per_pixel
    return adata


def read_tissue_positions(directory) -> tuple[np.ndarray, np.ndarray]:
    """Centres (x, y in um) of the in-tissue 2 um bins of one section (``binned_outputs/square_002um``)."""
    with open(f"{directory}/spatial/scalefactors_json.json") as handle:
        microns_per_pixel = json.load(handle)["microns_per_pixel"]
    positions = pd.read_parquet(
        f"{directory}/spatial/tissue_positions.parquet",
        columns=["in_tissue", "pxl_row_in_fullres", "pxl_col_in_fullres"],
    )
    positions = positions[positions.in_tissue == 1]
    return (
        positions.pxl_col_in_fullres.to_numpy() * microns_per_pixel,
        positions.pxl_row_in_fullres.to_numpy() * microns_per_pixel,
    )


def write_vessels(vessels, path: str) -> None:
    """Write the lattice cells of :func:`hhca.tl.vascular_lattice` as a table (``<slide>_cells.parquet``)."""
    by_code = vessels.var.sort_values("K1").index
    votes = pd.DataFrame(np.asarray(vessels.X), columns=[f"vote_{name}" for name in vessels.var_names])
    obs = vessels.obs.reset_index(drop=True)
    cells = pd.concat([obs[["key", "gx", "gy"]], votes[[f"vote_{name}" for name in by_code]]], axis=1)
    cells["n_votes"] = obs["n_votes"]
    cells["dom_subtype"] = obs["dom_subtype"].astype(str).astype(object)
    for column in ("x_um", "y_um", "struct"):
        cells[column] = obs[column]
    cells.to_parquet(path + ".part", index=False)
    os.replace(path + ".part", path)


def read_vessels(path: str, vascular_types: dict, factor_order):
    """Read a table written by :func:`write_vessels` back as the AnnData of lattice cells."""
    import anndata as ad

    cells = pd.read_parquet(path)
    names = list(vascular_types)
    code_of = {level3_type: code for code, level3_type in enumerate(factor_order)}
    obs = cells[["key", "gx", "gy", "n_votes", "dom_subtype", "x_um", "y_um", "struct"]].copy()
    obs["dom_subtype"] = pd.Categorical(obs["dom_subtype"], categories=names)
    obs.index = cells["key"].astype(str).values
    var = pd.DataFrame(
        {"cell_type": list(vascular_types.values()), "K1": [code_of[t] for t in vascular_types.values()]}, index=names
    )
    return ad.AnnData(
        X=cells[[f"vote_{name}" for name in names]].to_numpy(np.float64),
        obs=obs,
        var=var,
        obsm={"spatial": cells[["x_um", "y_um"]].to_numpy(np.float64)},
    )


def read_xenium_cohort(directory, counts: bool = True):
    """Read the extraction of a Xenium Seurat object (``scripts/xenium/extract_mtx.R``) as AnnData.

    Cells x genes with the raw counts in ``X`` (left out with ``counts=False``), the author metadata of every
    cell in ``obs`` and the cell centroids (um) in ``obsm["spatial"]``.
    """
    import anndata as ad
    import scipy.io

    directory = os.fspath(directory).rstrip("/")
    with open(f"{directory}/barcodes.txt") as handle:
        barcodes = [line.strip() for line in handle]
    with open(f"{directory}/genes.txt") as handle:
        genes = [line.strip() for line in handle]
    obs = pd.read_csv(f"{directory}/obs.csv").set_index("barcode", drop=False).reindex(barcodes)
    obs.index.name = None
    geometry = pd.read_csv(f"{directory}/cells_geom.csv", usecols=["cell_id", "x_centroid", "y_centroid"])
    geometry = geometry.set_index("cell_id").reindex(barcodes)
    matrix = scipy.io.mmread(f"{directory}/counts.mtx").tocsr().T.tocsr() if counts else None
    adata = ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=genes))
    adata.obsm["spatial"] = geometry[["x_centroid", "y_centroid"]].to_numpy(np.float64)
    return adata
