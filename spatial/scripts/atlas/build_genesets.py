#!/usr/bin/env python
"""Build ALL_genesets.json (the 2,763 gene sets of the per-donor scores) from public sources.

  DRUG::   drug2cell 0.1.2 bundled ChEMBL dictionary (drug-target_dicts.pkl): the ATC-nested dictionary
           flattened with dict(ChainMap(...)), drugs kept if >= 1 target is an atlas gene, gene list = the
           targets present (first-seen order). atc = first ATC category listing the drug; n_cardiac_targets =
           number of the drug's targets among the cardiac-axis genes.
           The >= 3-targets gate is not applied here; it is applied downstream (MIN_TARGETS_DRUG = 3).
  AXIS::   the 12 curated cardiac axes (resources/atlas/cardiac_axes.yaml), genes present in the atlas.
  HPROD::  hormone2cell HRDB v2.0.0 producing definition: for each hormone_short, sorted(hpc_include1-8 minus
           hpc_exclude1-2) of its last definition row (rows processed in file order, a later row
           overwriting an earlier one). For the 29 hormones with several alternative routes this keeps only
           the last route; --union-producers takes the union over all rows instead.
  HRECV::  HRDB v2.0.0 established receptors (receptorgene1/2) restricted to the 14,016-gene FICTURE panel
           of the L4 atlas profile (= receiver_strength_L4_long.csv of hormone2cell_producer_L4.py),
           genes sorted; hormone_display/tier from the hormone2cell L4 producer table when the hormone is
           produced there, otherwise label / "".
The assembly is hhca.tl.gene_set_collection.

Inputs: atlas h5ad (gene names), L4 atlas profile, hormone2cell producer table, cardiac_axes.yaml and the
dictionaries shipped with drug2cell and hormone2cell.
Outputs: ALL_genesets.json ({"groups", "gmeta"}).

Usage: python build_genesets.py --atlas ATLAS.h5ad --panel atlas_L4_pseudobulk.tsv \
           --producer-table producer_strength_L4_long.csv --axes resources/atlas/cardiac_axes.yaml \
           --out results/atlas/resources/ALL_genesets.json
"""

import argparse
import json
import os

import h5py
import pandas as pd
import yaml

import hhca
from hhca.tl._donors import var_names
from hhca.tl._hormones import package_file


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--atlas", required=True, help="atlas h5ad (var names only)")
    parser.add_argument("--panel", required=True, help="L4 atlas profile TSV; its gene index = FICTURE panel")
    parser.add_argument(
        "--producer-table", required=True, help="hormone2cell L4 producer long table (hormone2cell_producer_L4.py)"
    )
    parser.add_argument("--axes", required=True, help="cardiac_axes.yaml")
    parser.add_argument("--out", required=True, help="ALL_genesets.json to write")
    parser.add_argument("--chembl-pkl", default=None, help="default: drug2cell's drug-target_dicts.pkl")
    parser.add_argument("--producing-pkl", default=None, help="default: hormone2cell HRDB v2.0.0 pickle")
    parser.add_argument("--receptor-pkl", default=None, help="default: hormone2cell HRDB v2.0.0 pickle")
    parser.add_argument(
        "--union-producers",
        action="store_true",
        help="HPROD = union over all definition rows of a hormone (default: its last definition row)",
    )
    return parser.parse_args()


def main():
    """Read the four sources and write ALL_genesets.json."""
    args = parse_args()
    with h5py.File(args.atlas, "r") as h5:
        atlas_genes = var_names(h5)
    chembl_nested = pd.read_pickle(args.chembl_pkl or package_file("drug2cell", "drug-target_dicts.pkl"))
    producing, receptors = hhca.tl.hormone_definitions(args.producing_pkl, args.receptor_pkl)
    panel = set(pd.read_csv(args.panel, sep="\t", usecols=[0]).iloc[:, 0].astype(str))
    with open(args.axes) as handle:
        axes = yaml.safe_load(handle)

    out = hhca.tl.gene_set_collection(
        atlas_genes,
        chembl_nested,
        axes,
        producing,
        receptors,
        panel,
        pd.read_csv(args.producer_table),
        args.union_producers,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(out, handle)
    print(f"wrote {args.out}: {len(out['groups'])} gene sets (atlas genes {len(atlas_genes)}, panel {len(panel)})")


if __name__ == "__main__":
    main()
