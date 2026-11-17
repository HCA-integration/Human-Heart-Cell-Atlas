"""ED6 d, e: per-label pseudobulk of one CMFLAG-clean 2 um h5ad.

Raw counts are summed over the bins of each Level-3 label; the matrix is read in row blocks
(hhca.tl.label_pseudobulk).

Inputs: 2 um h5ad of one slide (obs L3_label, raw counts).
Outputs: genes x labels counts (csv.gz) and <out>.nbins.json (bins per label).

Usage: label_pseudobulk.py --h5ad SLIDE.h5ad --out SLIDE_label_pseudobulk.csv.gz
"""

import argparse
import json
import os

import anndata as ad

import hhca
from hhca.config import load_config


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="Raw counts per Level-3 label of one 2 um h5ad")
    parser.add_argument("--h5ad", required=True, help="2 um h5ad of one slide")
    parser.add_argument("--out", required=True, help="output csv.gz")
    return parser.parse_args()


def main():
    """Sum the counts per label and write the table and the bin counts."""
    args = parse_args()
    adata = ad.read_h5ad(args.h5ad, backed="r")
    counts = hhca.tl.label_pseudobulk(adata, chunk=load_config("cardiomyocyte")["ed6d"]["chunk"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    counts.to_csv(args.out, compression="gzip")
    with open(args.out + ".nbins.json", "w") as handle:
        json.dump(counts.attrs["n_bins"], handle, indent=1)
    print(
        f"wrote {args.out}: {counts.shape[0]} genes x {counts.shape[1]} labels, {adata.n_obs:,} bins, "
        f"total counts {counts.to_numpy().sum():.0f}"
    )


if __name__ == "__main__":
    main()
