"""Pool the per-donor gene-set scores from the 291 atlas donor ids to 280 donors.

Eleven donors appear twice in the atlas (Litviňuková and Reichart releases of the same person). The score file
stores per-(donor x cell type) score sums and nucleus counts, so pooling is exact: rows of the same donor are
added (hhca.tl.pool_donors). Per-donor metadata comes from the row whose atlas id equals the pooled id (the bare
Litviňuková id), otherwise the first row; nucleus totals are summed (hhca.tl.pool_donor_table).

Inputs: per_donor_celltype_scores.npz (donor_geneset_scores.py), resources/atlas/donor_map.csv.
Outputs: per_donor_celltype_scores_280donors.npz.

Usage: python pool_duplicate_donors.py --npz per_donor_celltype_scores.npz \
           --map resources/atlas/donor_map.csv --out per_donor_celltype_scores_280donors.npz
"""

import argparse

import hhca
from hhca.donors import read_donor_map, read_scores, write_scores


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--npz", required=True, help="per-donor score file of donor_geneset_scores.py")
    parser.add_argument("--map", required=True, help="donor map (columns raw_donor_id, donor_id)")
    parser.add_argument("--out", required=True, help="pooled score file to write")
    return parser.parse_args()


def main():
    """Pool the score file to one row block per donor."""
    args = parse_args()
    donor_map = read_donor_map(args.map)
    donors, scores, gene_set_meta = read_scores(args.npz)
    pooled_donors = hhca.tl.pool_donor_table(donors, donor_map)
    pooled = {level: hhca.tl.pool_donors(level_scores, donor_map) for level, level_scores in scores.items()}
    write_scores(args.out, pooled_donors, pooled, gene_set_meta)
    print(f"{len(donors)} donor ids -> {len(pooled_donors)} donors -> {args.out}")


if __name__ == "__main__":
    main()
