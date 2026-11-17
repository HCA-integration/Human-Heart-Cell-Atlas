"""Reading Space Ranger square_NNNum folders and writing the tables FICTURE takes as input."""

import json
import os

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp


def microns_per_pixel(square_dir):
    """microns_per_pixel of the full-resolution image, from spatial/scalefactors_json.json."""
    with open(os.path.join(square_dir, "spatial", "scalefactors_json.json")) as handle:
        return float(json.load(handle)["microns_per_pixel"])


def read_10x(h5_path):
    """Count matrix (genes x barcodes, CSC), barcodes and feature table of a 10x h5 file."""
    with h5py.File(h5_path, "r") as f:
        matrix = f["matrix"]
        n_genes, n_barcodes = [int(v) for v in matrix["shape"][:]]
        counts = sp.csc_matrix(
            (matrix["data"][:], matrix["indices"][:], matrix["indptr"][:]), shape=(n_genes, n_barcodes)
        )
        barcodes = np.array([b.decode() for b in matrix["barcodes"][:]])
        features = {
            name: np.array([b.decode() for b in matrix["features"][name][:]])
            for name in ("name", "id", "feature_type", "genome")
        }
    return counts, barcodes, features


def pixel_positions(square_dir, barcodes):
    """Full-resolution pixel row and column (float64) of each barcode, from spatial/tissue_positions.parquet."""
    positions = pq.read_table(
        os.path.join(square_dir, "spatial", "tissue_positions.parquet"),
        columns=["barcode", "pxl_row_in_fullres", "pxl_col_in_fullres"],
    ).to_pandas()
    row_of = dict(zip(positions["barcode"], positions["pxl_row_in_fullres"].values))
    col_of = dict(zip(positions["barcode"], positions["pxl_col_in_fullres"].values))
    pxl_row = np.fromiter((row_of[b] for b in barcodes), float, len(barcodes))
    pxl_col = np.fromiter((col_of[b] for b in barcodes), float, len(barcodes))
    return pxl_row, pxl_col


def is_mitochondrial(gene_names):
    """True for the genes whose symbol starts with MT-."""
    return np.char.startswith(np.char.upper(gene_names.astype(str)), "MT-")


def write_coordinate_minmax(path, x, y, precision):
    """Write coordinate_minmax.tsv: xmin / xmax / ymin / ymax (um) with `precision` decimals."""
    with open(path, "w") as handle:
        handle.write(f"xmin\t{x.min():.{precision}f}\n")
        handle.write(f"xmax\t{x.max():.{precision}f}\n")
        handle.write(f"ymin\t{y.min():.{precision}f}\n")
        handle.write(f"ymax\t{y.max():.{precision}f}\n")


def write_feature_table(path, gene_names, gene_ids, totals):
    """Write feature.clean.tsv.gz: detected non-mitochondrial genes by decreasing total; return their number."""
    features = pd.DataFrame({"gene": gene_names, "gene_id": gene_ids, "Count": totals})
    features = features[(features["Count"] > 0) & (~is_mitochondrial(gene_names))].sort_values("Count", ascending=False)
    features.to_csv(path, sep="\t", index=False, compression="gzip")
    return len(features)
