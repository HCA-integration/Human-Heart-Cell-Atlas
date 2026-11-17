#!/usr/bin/env python
"""hormone2cell producer and receiver strength per Level-4 cell type of the HHCA atlas (Fig 4 l tables).

hormone2cell 0.1.1 with its Hormone-Receptor Reference Database v2.0.0, run on the per-cell-type pseudobulk (mean
normalised expression, genes x Level-4 types: the L4 atlas profile of the substrate pipeline) and the nuclei per
type (hhca.tl.producer_strength). hormone2cell normally takes single cells and computes mean logExpression and the
share of expressing cells per gene and cell type. The pseudobulk carries the mean but not the detection rate, so:
  - logExpression = log1p(pseudobulk mean), the CP10k / log1p scale
  - TotalCellNumber = nuclei of the type (the nuclei-per-type table)
  - ExpressedCellNumber = TotalCellNumber, Percentage = 100 (every gene counts as detected)
The 5 % detection filter therefore does not act; calls rest on the mean-expression thresholds and the AND / OR
logic over biosynthesis genes. use_precomputed=None: per-gene threshold = 0.999 quantile of logExpression across
cell types x threshold, computed by the package from these data.

Inputs: L4 atlas profile (genes x Level-4 types), nuclei per Level-4 type (factor, n_cells).
Outputs: producer_strength_L4_long.csv, producer_strength_L4_wide.csv, receiver_strength_L4_long.csv,
producer_focus_axes.csv (the hormones of config/atlas_hormones.yaml, producer_focus).

Usage: python hormone2cell_producer_L4.py --profile atlas_L4_pseudobulk.tsv \
           --ncells atlas_L4_ncells.csv --outdir results/atlas/hormones/collab_hormone
"""

import argparse
import os

import pandas as pd

import hhca
from hhca.config import load_config


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--profile", required=True, help="Level-4 pseudobulk table (genes x types)")
    parser.add_argument("--ncells", required=True, help="nuclei per Level-4 type (factor, n_cells)")
    parser.add_argument("--outdir", required=True, help="directory for the tables")
    return parser.parse_args()


def main():
    """Write the producer and receiver strength tables of the Level-4 pseudobulk."""
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cfg = load_config("atlas_hormones")

    profile = pd.read_csv(args.profile, sep="\t", index_col=0)  # genes x cell types
    profile.index.name = "Gene"
    ncells = pd.read_csv(args.ncells)
    nuclei_per_type = dict(zip(ncells["factor"], ncells["n_cells"]))
    print(f"pseudobulk: {profile.shape[0]} genes x {profile.shape[1]} cell types")
    producing, receptors = hhca.tl.hormone_definitions()

    producers = hhca.tl.producer_strength(
        hhca.tl.profile_pseudobulk(profile, nuclei_per_type),
        producing,
        strongest_first=False,
        max_expression_file=os.path.join(args.outdir, "gene_max_expression_producer.csv"),
    )
    producers.to_csv(os.path.join(args.outdir, "producer_strength_L4_long.csv"), index=False)
    print(f"producer pairs (Strength>0): {producers.shape[0]}, hormones produced: {producers['Hormone'].nunique()}")

    # wide producer matrix (hormone x cell type)
    wide = producers.pivot_table(index="Hormone", columns="Celltype", values="Strength", aggfunc="max").fillna(0)
    wide.to_csv(os.path.join(args.outdir, "producer_strength_L4_wide.csv"))

    receivers = hhca.tl.receptor_expression(profile, receptors)
    receivers.to_csv(os.path.join(args.outdir, "receiver_strength_L4_long.csv"), index=False)
    print(f"receiver pairs: {receivers.shape[0]}, hormones with receptors: {receivers['Hormone'].nunique()}")

    focus = hhca.tl.focus_producers(producers, cfg["producer_focus"])
    focus.to_csv(os.path.join(args.outdir, "producer_focus_axes.csv"), index=False)
    print("tables written to", args.outdir)


if __name__ == "__main__":
    main()
