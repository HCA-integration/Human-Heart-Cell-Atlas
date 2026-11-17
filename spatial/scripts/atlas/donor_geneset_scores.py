#!/usr/bin/env python
"""Per-donor gene-set scores on the HHCA atlas, from one streaming pass over the counts.

  score(set) = mean over the set's genes present in the atlas of log1p CP10k expression
  (the same definition as drug2cell score(method="mean") and the hormone2cell gene-set mean)

Gene-set families (ALL_genesets.json, built by build_genesets.py; one scoring rule, so they are comparable):
  drug            : 2446 ChEMBL drug target groups (drug2cell dictionary, >= 1 target in the atlas)
  cardiac_axis    : 12 curated cardiac drug-mechanism axes
  hormone_producer: 188 hormone biosynthesis gene sets (hormone2cell producing definition v2.0.0,
                    hpc_include minus hpc_exclude), the mean-score counterpart of the package's
                    thresholded "Strength"
  hormone_receiver: 117 established hormone receptor gene sets (hormone2cell receptor definition)

Scores are summed per (donor, cell type) at Final Level 2, 3 and 4 together with the nucleus counts
(hhca.tl.gene_set_scores); donors are the unit of replication (291 atlas donor ids). Nothing is stored per cell.
Annotation levels and block size: config/config.yaml (atlas).

Inputs: atlas h5ad (layers/counts, obs labels), ALL_genesets.json.
Outputs: one .npz with per-(donor, cell type) score sums and nucleus counts at L2-L4 and per-donor metadata.

Usage: python donor_geneset_scores.py --atlas ATLAS.h5ad --genesets resources/atlas/ALL_genesets.json \
           --out results/atlas/per_donor_celltype_scores.npz
"""

import argparse
import json
import os

import hhca
from hhca.config import load_config
from hhca.donors import write_scores
from hhca.log import get_logger


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--atlas", required=True, help="HHCA atlas h5ad (layers/counts, obs labels)")
    parser.add_argument("--genesets", required=True, help="ALL_genesets.json")
    parser.add_argument("--out", required=True, help="output .npz")
    return parser.parse_args()


def main():
    """Stream the atlas and write the per-(donor, cell type) gene-set score sums."""
    args = parse_args()
    log = get_logger()
    cfg = load_config("config")["atlas"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.genesets) as handle:
        gene_sets = json.load(handle)

    donors = hhca.tl.donor_table(args.atlas)
    scores = hhca.tl.gene_set_scores(args.atlas, gene_sets["groups"], cfg["levels"], cfg["stream_chunk"], log)
    write_scores(args.out, donors, scores, gene_sets["gmeta"])
    log(f"wrote {args.out}")
    print(f"donors per sex: {donors['sex'].value_counts().to_dict()}; unknown age: {int(donors['age'].isna().sum())}")
    print(f"disease labels among donors: {donors['disease'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
