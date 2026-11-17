#!/usr/bin/env python
"""Build the CellPin query AnnData of the healthy 10x Xenium section.

Keeps the gene features of the 10x output bundle (control probes, blank and deprecated codewords are dropped) and
writes raw counts in .X and .layers['counts'] with the cell centroids in obsm['spatial'].

Inputs:
    --matrix  cell_feature_matrix.h5 of the 10x output bundle
    --cells   cells.parquet of the 10x output bundle (centroids, areas)

Outputs:
    --out     xenium_genes.h5ad

Usage: python build_healthy_h5ad.py --matrix cell_feature_matrix.h5 --cells cells.parquet --out xenium_genes.h5ad
"""

import argparse

import numpy as np
import pandas as pd
import scanpy as sc

# name fragments of the control features, used when the matrix has no feature_types column
CONTROL_NAME_PARTS = ("NegControl", "Negative", "antisense", "BLANK", "Blank", "Unassigned", "Deprecated", "Codeword")
CELL_COLUMNS = ["x_centroid", "y_centroid", "transcript_counts", "cell_area", "nucleus_area"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--matrix", required=True, help="10x cell_feature_matrix.h5")
    parser.add_argument("--cells", required=True, help="10x cells.parquet (centroids, areas)")
    parser.add_argument("--out", required=True, help="output xenium_genes.h5ad")
    return parser.parse_args()


def read_gene_counts(matrix_path):
    """Cell x gene counts of the 10x matrix without the control features."""
    print("[xenium] reading cell_feature_matrix.h5 ...")
    adata = sc.read_10x_h5(matrix_path, gex_only=False)
    adata.var_names_make_unique()
    print("[xenium] raw matrix:", adata.shape)
    print(
        "[xenium] feature_types:",
        adata.var["feature_types"].value_counts().to_dict() if "feature_types" in adata.var else "no feature_types col",
    )
    # Xenium types its control features 'Negative Control Probe', 'Negative Control Codeword', 'Unassigned
    # Codeword', 'Antisense', 'Blank Codeword' and 'Deprecated Codeword'; genes are 'Gene Expression'.
    if "feature_types" in adata.var:
        is_gene = adata.var["feature_types"].astype(str) == "Gene Expression"
    else:
        is_gene = ~adata.var_names.str.contains("|".join(CONTROL_NAME_PARTS), case=False, regex=True)
    adata = adata[:, is_gene.values].copy()
    print("[xenium] gene-only matrix:", adata.shape)
    return adata


def add_cell_table(adata, cells_path):
    """Add the centroids (obsm['spatial']) and the per-cell columns of cells.parquet."""
    cells = pd.read_parquet(cells_path)
    cells["cell_id"] = cells["cell_id"].astype(str)
    cells = cells.set_index("cell_id")
    shared = adata.obs_names.intersection(cells.index)
    assert len(shared) == adata.n_obs, f"centroid mismatch: {len(shared)} vs {adata.n_obs}"
    cells = cells.loc[adata.obs_names]
    adata.obsm["spatial"] = cells[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    for column in CELL_COLUMNS:
        if column in cells.columns:
            adata.obs[column] = cells[column].to_numpy()


def main():
    args = parse_args()
    adata = read_gene_counts(args.matrix)
    add_cell_table(adata, args.cells)

    # raw counts in X and in a layer of their own
    adata.layers["counts"] = adata.X.copy()
    adata.obs["total_counts"] = np.asarray(adata.X.sum(1)).ravel()
    adata.obs["n_genes"] = np.asarray((adata.X > 0).sum(1)).ravel()

    print("[xenium] median counts/cell:", float(np.median(adata.obs["total_counts"])))
    print("[xenium] median genes/cell :", float(np.median(adata.obs["n_genes"])))
    adata.write_h5ad(args.out)
    print("[xenium] wrote", args.out, adata)


if __name__ == "__main__":
    main()
