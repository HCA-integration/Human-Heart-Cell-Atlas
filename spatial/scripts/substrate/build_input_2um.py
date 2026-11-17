#!/usr/bin/env python
"""FICTURE input files from the square_002um matrix of a whole Visium HD slide.

The count matrix is expanded to long format in one pass, sorted by Y and written gzipped in chunks; genes
stay integer indices until each chunk is written, which bounds the memory.

Inputs: a Space Ranger square_002um folder.
Outputs (in --out-dir): transcripts.tsv.gz (X, Y, gene, Count; um, sorted by Y, mitochondrial genes dropped),
feature.clean.tsv.gz, coordinate_minmax.tsv.

Usage: python build_input_2um.py --slide-dir SQUARE_002UM --out-dir DIR
"""

import argparse
import gzip
import os

import h5py
import numpy as np
import pandas as pd

from hhca.log import get_logger
from visium_hd import (
    is_mitochondrial,
    microns_per_pixel,
    pixel_positions,
    write_coordinate_minmax,
    write_feature_table,
)

log = get_logger("input")


def read_matrix(h5_path):
    """Shape, column pointers, barcodes, gene names / ids and the non-zero entries (int32) of a 10x h5 matrix."""
    with h5py.File(h5_path, "r") as f:
        matrix = f["matrix"]
        n_genes, n_bins = [int(x) for x in matrix["shape"][:]]
        indptr = matrix["indptr"][:]
        barcodes = np.array([b.decode() for b in matrix["barcodes"][:]])
        gene_name = np.array([b.decode() for b in matrix["features"]["name"][:]])
        gene_id = np.array([b.decode() for b in matrix["features"]["id"][:]])
        log(f"matrix {n_genes} genes x {n_bins} bins; nnz={int(indptr[-1])}; loading data/indices ...")
        data = matrix["data"][:].astype(np.int32)
        indices = matrix["indices"][:].astype(np.int32)  # gene index per non-zero entry
    return n_genes, n_bins, indptr, barcodes, gene_name, gene_id, data, indices


def write_transcripts(path, x, y, gene_name, gene, count, chunk, precision):
    """Write the long table gzipped, `chunk` rows at a time."""
    with gzip.open(path, "wt") as gz:
        gz.write("X\tY\tgene\tCount\n")
        for start in range(0, gene.size, chunk):
            end = min(gene.size, start + chunk)
            pd.DataFrame(
                {"X": x[start:end], "Y": y[start:end], "gene": gene_name[gene[start:end]], "Count": count[start:end]}
            ).to_csv(gz, sep="\t", index=False, header=False, float_format=f"%.{precision}f")


def parse_args():
    """Command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slide-dir", required=True, help="Space Ranger square_002um folder")
    parser.add_argument("--out-dir", required=True, help="folder for the FICTURE input files")
    parser.add_argument("--precision", type=int, default=2, help="decimals of the written coordinates")
    parser.add_argument("--chunk", type=int, default=8_000_000, help="rows per written chunk")
    return parser.parse_args()


def main():
    """Write the three FICTURE input files of a whole slide."""
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    mpp = microns_per_pixel(args.slide_dir)
    log(f"microns_per_pixel = {mpp:.6f}")

    n_genes, n_bins, indptr, barcodes, gene_name, gene_id, data, indices = read_matrix(
        os.path.join(args.slide_dir, "filtered_feature_bc_matrix.h5")
    )
    pxl_row, pxl_col = pixel_positions(args.slide_dir, barcodes)
    bin_x = (pxl_col * mpp).astype(np.float32)
    bin_y = (pxl_row * mpp).astype(np.float32)

    # gene totals over the slide, before the mitochondrial genes are dropped from the rows
    gene_totals = np.bincount(indices, weights=data, minlength=n_genes).astype(np.int64)

    # one row per non-zero entry
    bin_of_entry = np.repeat(np.arange(n_bins, dtype=np.int32), np.diff(indptr).astype(np.int64))
    log(f"expanded to {bin_of_entry.size} nnz rows")
    keep = ~is_mitochondrial(gene_name)[indices]
    gene = indices[keep]
    count = data[keep]
    x = bin_x[bin_of_entry[keep]]
    y = bin_y[bin_of_entry[keep]]
    del bin_of_entry
    log(f"without mitochondrial genes: {gene.size} rows; sorting by Y ...")
    order = np.argsort(y, kind="stable")
    x = x[order]
    y = y[order]
    gene = gene[order]
    count = count[order]
    del order

    out_transcripts = os.path.join(args.out_dir, "transcripts.tsv.gz")
    log(f"writing {out_transcripts} in chunks of {args.chunk} ...")
    write_transcripts(out_transcripts, x, y, gene_name, gene, count, args.chunk, args.precision)
    log(f"wrote {out_transcripts}")

    out_minmax = os.path.join(args.out_dir, "coordinate_minmax.tsv")
    write_coordinate_minmax(out_minmax, bin_x, bin_y, args.precision)
    log(f"wrote {out_minmax}: X[{bin_x.min():.0f},{bin_x.max():.0f}] Y[{bin_y.min():.0f},{bin_y.max():.0f}] um")

    n_features = write_feature_table(
        os.path.join(args.out_dir, "feature.clean.tsv.gz"), gene_name, gene_id, gene_totals
    )
    log(f"wrote feature.clean.tsv.gz: {n_features} genes (MT- dropped)")
    log("done")


if __name__ == "__main__":
    main()
