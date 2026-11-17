#!/usr/bin/env python
"""Export the hormone2cell HRDB v2.0.0 gene definitions as the two flat tables read by lv_hormone_tensor.py.

Writes hormone_{producer,receptor}_gene_sets.csv (hhca.tl.hormone_gene_sets):
  producer: per hormone_short, each definition row -> AND list of hpc_include1-8 ('+'), alternatives
            joined with '|'; exclude genes (hpc_exclude1-2) per row joined with ',' and rows with '|'
  receptor: per hormone_short, established receptors (receptorgene1/2, union over rows, first-seen order)

Inputs: the definition pickles shipped with hormone2cell.
Outputs: hormone_producer_gene_sets.csv, hormone_receptor_gene_sets.csv.

Usage: python hormone_genesets_export.py --outdir results/atlas/hormones/genesets
       [--producing-pkl P.pkl --receptor-pkl R.pkl]   (default: the pickles shipped with hormone2cell)
"""

import argparse
import os

import hhca


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--producing-pkl", default=None, help="default: hormone2cell HRDB v2.0.0 producing definition")
    parser.add_argument("--receptor-pkl", default=None, help="default: hormone2cell HRDB v2.0.0 receptor definition")
    return parser.parse_args()


def main():
    """Write the producer and receptor gene-set tables."""
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    producing, receptors = hhca.tl.hormone_definitions(args.producing_pkl, args.receptor_pkl)
    producer_sets, receptor_sets = hhca.tl.hormone_gene_sets(producing, receptors)
    producer_sets.to_csv(f"{args.outdir}/hormone_producer_gene_sets.csv", index=False)
    receptor_sets.to_csv(f"{args.outdir}/hormone_receptor_gene_sets.csv", index=False)
    print(
        f"{len(producer_sets)} producer hormones, {len(receptor_sets)} receptor hormones -> {args.outdir}", flush=True
    )


if __name__ == "__main__":
    main()
