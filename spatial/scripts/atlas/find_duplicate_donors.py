#!/usr/bin/env python
"""Find the donors that appear twice in the atlas from the data and compare them with resources/atlas/donor_map.csv.

The map pairs donor ids by the id pattern Reichart_2022_adata_X <-> X (sex agrees for 11/11 pairs, age band
for 10/11). hhca.tl.duplicate_donors tests identity from the nuclei themselves, without using the donor ids:
  1. per donor: dataset, sex, development stage, tissues, nuclei (modes over nuclei);
  2. expression identity: per (donor x Final Level 2 type) pseudobulk of raw counts streamed from the atlas
     -> log1p CP10k; each donor's profile is centred on its dataset mean (removes study batch), sex genes
     are dropped, and donors are correlated over the 2,000 genes with the largest residual variance. This is
     done for the whole donor and within each L2 type (removes composition differences between the two
     samplings); the identity score of a donor pair is the median over L2 types with enough nuclei in both
     donors (whole-donor r reported alongside);
  3. barcode overlap: shared 16-nt cell barcodes between the two donors' nuclei (a re-release of the same
     libraries shares barcodes; independent libraries share ~n1*n2/whitelist by chance).
A pair is called a duplicate when the two donors come from different datasets, are each other's best
cross-dataset match and the score is an outlier among that donor's cross-dataset correlations. The called pairs
are compared with the map (pooled id = bare id). Thresholds and sex genes: config/config.yaml (duplicate_donors).

Inputs: atlas h5ad, resources/atlas/donor_map.csv.
Outputs: donor_metadata.csv, duplicate_candidates_top3.csv, duplicate_pairs_called.csv, map_check.csv.

Usage: python find_duplicate_donors.py --atlas ATLAS.h5ad --map resources/atlas/donor_map.csv \
           --outdir results/atlas/resources/duplicate_donors
"""

import argparse
import os

import hhca
from hhca.config import load_config
from hhca.donors import read_donor_map
from hhca.log import get_logger
from hhca.tl._duplicates import CALLED_COLUMNS


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--atlas", required=True, help="HHCA atlas h5ad")
    parser.add_argument("--map", required=True, help="donor map (columns raw_donor_id, donor_id)")
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def main():
    """Score donor identity from the nuclei, call duplicate pairs and compare them with the donor map."""
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cfg = load_config("config")
    found = hhca.tl.duplicate_donors(
        args.atlas, chunk=cfg["atlas"]["stream_chunk"], log=get_logger(), **cfg["duplicate_donors"]
    )
    found["candidates"].to_csv(f"{args.outdir}/duplicate_candidates_top3.csv", index=False)
    found["donors"].to_csv(f"{args.outdir}/donor_metadata.csv", index=False)
    found["pairs"].to_csv(f"{args.outdir}/duplicate_pairs_called.csv", index=False)

    check = hhca.tl.compare_with_map(found["pairs"], read_donor_map(args.map))
    check.to_csv(f"{args.outdir}/map_check.csv", index=False)
    print(f"\npairs called from the data: {int(check.called.sum())}; pairs in the map: {int(check.in_map.sum())}")
    print(found["pairs"][CALLED_COLUMNS].to_string(index=False))
    print("the map agrees with the data" if (check.called == check.in_map).all() else "the map differs from the data")


if __name__ == "__main__":
    main()
