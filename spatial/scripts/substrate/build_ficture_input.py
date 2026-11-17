#!/usr/bin/env python
"""FICTURE input files from a Visium HD square_002um matrix, for the whole slide or a crop.

Used with --crop to define the gene set of the model (genes detected in one 800 um field).
Coordinates are pxl_{row,col}_in_fullres * microns_per_pixel, so FICTURE runs with --mu_scale 1.

Inputs: a Space Ranger square_002um folder.
Outputs (in --out-dir):
  transcripts.tsv.gz      X, Y, gene, Count (um), sorted by Y
  coordinate_minmax.tsv   xmin / xmax / ymin / ymax (um)
  feature.clean.tsv.gz    gene, gene_id, Count (totals per gene, mitochondrial genes dropped)

Usage: python build_ficture_input.py --slide-dir SQUARE_002UM --out-dir DIR [--crop X0 X1 Y0 Y1]
"""

import argparse
import os

import numpy as np
import pandas as pd

from hhca.log import get_logger
from visium_hd import microns_per_pixel, pixel_positions, read_10x, write_coordinate_minmax, write_feature_table

log = get_logger("input")


def parse_args():
    """Command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slide-dir", required=True, help="Space Ranger square_002um folder")
    parser.add_argument("--out-dir", required=True, help="folder for the FICTURE input files")
    parser.add_argument(
        "--crop",
        nargs=4,
        type=float,
        default=None,
        metavar=("X0", "X1", "Y0", "Y1"),
        help="crop in um; whole slide if not given",
    )
    parser.add_argument("--precision", type=int, default=2, help="decimals of the written coordinates")
    return parser.parse_args()


def main():
    """Write the three FICTURE input files of a slide or of a crop of it."""
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    mpp = microns_per_pixel(args.slide_dir)
    log(f"microns_per_pixel = {mpp:.6f}")

    log("loading h5 ...")
    counts, barcodes, features = read_10x(os.path.join(args.slide_dir, "filtered_feature_bc_matrix.h5"))
    gene_name, gene_id = features["name"], features["id"]
    n_genes, n_bins = counts.shape
    log(f"matrix {n_genes} genes x {n_bins} bins, nnz={counts.nnz}")

    log("loading coords ...")
    pxl_row, pxl_col = pixel_positions(args.slide_dir, barcodes)
    bin_x = pxl_col * mpp
    bin_y = pxl_row * mpp

    if args.crop:
        x0, x1, y0, y1 = args.crop
        keep = (bin_x >= x0) & (bin_x < x1) & (bin_y >= y0) & (bin_y < y1)
        log(f"crop X[{x0:.0f},{x1:.0f}] Y[{y0:.0f},{y1:.0f}] -> {keep.sum()} bins")
    else:
        keep = np.ones(n_bins, bool)
        log(f"whole slide -> {keep.sum()} bins")

    # one row per non-zero entry of the kept bins
    kept_bins = np.where(keep)[0]
    entries = counts[:, kept_bins].tocoo()  # genes x kept bins
    gene = entries.row
    count = entries.data.astype(np.int32)
    x = bin_x[kept_bins][entries.col].astype(np.float32)
    y = bin_y[kept_bins][entries.col].astype(np.float32)
    log(f"emitting {len(count)} (X,Y,gene,Count) rows")

    # FICTURE expects the transcripts sorted along Y
    order = np.argsort(y, kind="stable")
    x, y, gene, count = x[order], y[order], gene[order], count[order]

    out_transcripts = os.path.join(args.out_dir, "transcripts.tsv.gz")
    transcripts = pd.DataFrame(
        {"X": np.round(x, args.precision), "Y": np.round(y, args.precision), "gene": gene_name[gene], "Count": count}
    )
    log(f"writing {out_transcripts} ...")
    transcripts.to_csv(out_transcripts, sep="\t", index=False, float_format=f"%.{args.precision}f", compression="gzip")

    out_minmax = os.path.join(args.out_dir, "coordinate_minmax.tsv")
    write_coordinate_minmax(out_minmax, x, y, args.precision)
    log(f"wrote {out_minmax}")

    # gene totals over the kept bins
    gene_totals = np.asarray(entries.sum(axis=1)).ravel().astype(np.int64)
    out_features = os.path.join(args.out_dir, "feature.clean.tsv.gz")
    n_features = write_feature_table(out_features, gene_name, gene_id, gene_totals)
    log(f"wrote {out_features}: {n_features} genes (MT- dropped)")

    log("done")


if __name__ == "__main__":
    main()
