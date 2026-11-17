#!/usr/bin/env python
"""Build the CellPin query AnnData of the GSE290577 rejection cohort from the extract_mtx.R outputs.

counts.mtx is genes x cells; the AnnData is cells x genes with raw integer counts in .X and .layers['counts']
(the same layout as build_healthy_h5ad.py), author obs (obs.csv) and centroids in obsm['spatial'] (cells_geom.csv).

Inputs:
    --extracted  folder with counts.mtx, genes.txt, barcodes.txt, obs.csv, cells_geom.csv (extract_mtx.R)

Outputs:
    --out        xenium_genes.h5ad

Usage: python build_rejection_h5ad.py --extracted EXTRACTED_DIR --out xenium_genes.h5ad
"""

import argparse

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--extracted",
        required=True,
        help="directory with counts.mtx, genes.txt, barcodes.txt, obs.csv, cells_geom.csv",
    )
    parser.add_argument("--out", required=True, help="output xenium_genes.h5ad")
    return parser.parse_args()


def read_names(path):
    """Non-empty lines of a text file."""
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]


def main():
    args = parse_args()
    extracted = args.extracted.rstrip("/")
    genes = read_names(f"{extracted}/genes.txt")
    barcodes = read_names(f"{extracted}/barcodes.txt")
    counts = scipy.io.mmread(f"{extracted}/counts.mtx").tocsr()  # genes x cells
    assert counts.shape == (len(genes), len(barcodes)), (counts.shape, len(genes), len(barcodes))
    X = sp.csr_matrix(counts.T, dtype=np.float32)
    assert np.all(X.data == np.round(X.data)), "counts are not integer-valued"

    obs = pd.read_csv(f"{extracted}/obs.csv", dtype={"biopsy_antibody_grading": str})
    obs = obs.set_index("barcode", drop=False)
    obs.index.name = None
    assert list(obs.index) == barcodes
    geometry = pd.read_csv(
        f"{extracted}/cells_geom.csv", usecols=["barcode", "x_centroid", "y_centroid", "cell_area", "nucleus_area"]
    )
    geometry = geometry.set_index("barcode").loc[barcodes]

    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    adata.obs["x_centroid"] = geometry["x_centroid"].to_numpy()
    adata.obs["y_centroid"] = geometry["y_centroid"].to_numpy()
    adata.obs["cell_area"] = geometry["cell_area"].to_numpy()
    adata.obs["nucleus_area"] = geometry["nucleus_area"].to_numpy()
    adata.obsm["spatial"] = geometry[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    adata.layers["counts"] = adata.X.copy()
    adata.obs["total_counts_panel"] = np.asarray(adata.X.sum(1)).ravel()
    adata.obs["n_genes"] = np.asarray((adata.X > 0).sum(1)).ravel()
    for column in adata.obs.columns:
        if adata.obs[column].dtype == object:
            adata.obs[column] = adata.obs[column].astype(str)
    print("[rej] ", adata, flush=True)
    print(
        "[rej] cores:",
        adata.obs["Sample"].nunique(),
        " slides:",
        adata.obs["slide"].value_counts().to_dict(),
        flush=True,
    )
    print("[rej] median counts/cell:", float(np.median(adata.obs["total_counts_panel"])), flush=True)
    adata.write_h5ad(args.out)
    print("[rej] wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
