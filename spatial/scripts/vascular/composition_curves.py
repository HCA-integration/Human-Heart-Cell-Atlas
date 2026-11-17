#!/usr/bin/env python
"""Fig 4 g: vascular cell-type composition of vessel structures across calibre (tables).

Composition of a structure = share of its CMFLAG-clean votes per vascular subtype (struct_votes_clean.py);
calibre = lumen-inclusive equivalent diameter (caliber.py). Structures that lose every vascular pixel get no
composition and are left out of the bin means, but stay in the bin counts n (hhca.tl.structure_composition,
hhca.tl.composition_by_calibre). Parameters: `composition` in config/vascular.yaml.
figures/Fig4/panel_g.py draws the panel from the tables written here.

Inputs: caliber_per_structure.csv (caliber.py), vessel_structs_cmflag_clean.csv (struct_votes_clean.py).
Outputs: composition_by_diameter_bin.csv, composition_by_nbins_bin.csv, composition_by_caliber_tier.csv,
         n_structures.txt

Usage: python composition_curves.py --caliber caliber_per_structure.csv --votes vessel_structs_cmflag_clean.csv
           --out-dir DIR
"""

import argparse
import os

from hhca import thread_env

os.environ.update(thread_env(1))

import numpy as np
import pandas as pd

import hhca
from hhca.config import load_config


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--caliber", required=True, help="calibre table of all slides (caliber.py)")
    parser.add_argument("--votes", required=True, help="CMFLAG-clean votes of struct_votes_clean.py")
    parser.add_argument("--out-dir", required=True, help="output directory")
    return parser.parse_args()


def main():
    """Write the composition tables by diameter bin, by size bin and by calibre class."""
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    params = load_config("vascular")["composition"]
    composition = hhca.tl.structure_composition(pd.read_csv(args.caliber), pd.read_csv(args.votes))
    print(f"pooled structures ({composition.slide.nunique()} slides): {len(composition)}")
    print(f"{int(composition['frac_VSMC'].isna().sum()):,} structures without a surviving composition")

    # calibre bins of fixed log width
    tables = {}
    for by, name, bins in (
        ("lumen_equiv_diam_um", "diam_um", params["diameter_bins"]),
        ("n_bins", "n_bins", params["size_bins"]),
    ):
        edges = np.geomspace(bins["low"], bins["high"], bins["n_edges"])
        tables[name] = hhca.tl.composition_by_calibre(composition, by, edges, params["min_structures"], name)
    tables["diam_um"].to_csv(f"{args.out_dir}/composition_by_diameter_bin.csv", index=False)
    tables["n_bins"].to_csv(f"{args.out_dir}/composition_by_nbins_bin.csv", index=False)

    by_class = hhca.tl.composition_by_class(composition)
    by_class.to_csv(f"{args.out_dir}/composition_by_caliber_tier.csv")
    with open(f"{args.out_dir}/n_structures.txt", "w") as handle:
        handle.write(f"{len(composition)}\n")
    print("\nmean composition (%) by calibre class")
    print(by_class.to_string())


if __name__ == "__main__":
    main()
