#!/usr/bin/env python
"""Build the single-cell reference shared by all deconvolution methods.

Streams the HHCA atlas h5ad with h5py, draws at most N cells per Level-3 type (without 'Unassigned') and writes
their raw counts (layers['counts']) with donor metadata as a compact AnnData (hhca.tl.sample_reference; config
reference).

Inputs:  the atlas h5ad
Outputs: <outdir>/hhca_L3_ref_<N>.h5ad, hhca_L3_celltypes.txt, hhca_L3_ref_<N>_counts.csv (cells kept per type)

Usage: build_reference.py --atlas ATLAS.h5ad --outdir DIR [--n-per-type 1000] [--seed 0] [--config CONFIG]
"""

import argparse
import os

import pandas as pd

import deconv_config
import hhca
from hhca.log import get_logger


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="Subsampled HHCA Level-3 reference with raw counts")
    parser.add_argument("--atlas", required=True, help="HHCA atlas h5ad")
    parser.add_argument("--outdir", required=True, help="output directory")
    parser.add_argument("--n-per-type", type=int, default=None, help="cells drawn per cell type (default: config)")
    parser.add_argument("--seed", type=int, default=None, help="seed of the subsampling (default: config)")
    parser.add_argument("--label", default="", help="obs column with the cell-type labels (default: config)")
    return deconv_config.add_argument(parser).parse_args()


def main():
    """Draw the reference cells and write the reference h5ad, the cell-type list and the cells kept per type."""
    args = parse_args()
    log = get_logger()
    settings = deconv_config.load(args.config)["reference"]
    label = args.label or settings["label"]
    n_per_type = args.n_per_type or settings["n_per_type"]
    seed = settings["seed"] if args.seed is None else args.seed
    os.makedirs(args.outdir, exist_ok=True)

    adata = hhca.tl.sample_reference(
        args.atlas, label, n_per_type, seed, settings["exclude"], settings["obs_columns"], log=log
    )
    kept_counts = adata.uns.pop("n_cells")
    log(f"types={len(kept_counts)} cells_kept={adata.n_obs:,} genes={adata.n_vars:,}")
    log("X sum:", float(adata.X.sum()), "max:", float(adata.X.max()), "min:", float(adata.X.min()))
    adata.uns["reference"] = {
        "source": args.atlas,
        "L3_col": label,
        "n_per_type": n_per_type,
        "seed": seed,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
    }
    out = f"{args.outdir}/hhca_L3_ref_{n_per_type}.h5ad"
    adata.write(out)
    log("wrote", out, adata.shape)

    with open(f"{args.outdir}/hhca_L3_celltypes.txt", "w") as handle:
        for cell_type in sorted(set(adata.obs["cell_type_L3"].astype(str))):
            handle.write(cell_type + "\n")
    pd.Series(kept_counts).sort_values(ascending=False).to_csv(
        f"{args.outdir}/hhca_L3_ref_{n_per_type}_counts.csv", header=["n_cells"]
    )
    log("done")


if __name__ == "__main__":
    main()
