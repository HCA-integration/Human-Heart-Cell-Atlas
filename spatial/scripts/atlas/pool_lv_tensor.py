"""Pool the LV per-donor hormone-gene tensor (Fig 2 h-i) from the 291 atlas donor ids to 280 donors.

The tensor stores per-(donor x L2 type x gene) means and nucleus counts, so pooling is exact up to the
float32 storage of the means: pooled_mean = sum_k(mean_k * n_k) / sum_k(n_k), pooled_n = sum_k n_k
(hhca.tl.pool_lv_expression). Donor order follows pool_duplicate_donors.py (first-seen pooled id). Dataset and sex
of a pooled donor are those of its row with the most LV nuclei, i.e. the mode over the pooled donor's LV nuclei,
which is what lv_hormone_tensor.py computes per donor (the bare Litviňuková row often has no LV nuclei).

Inputs: lv_donor_L2_gene_meanNorm.npz (lv_hormone_tensor.py), resources/atlas/donor_map.csv.
Outputs: the pooled tensor and, next to it, lv_dataset_sex_composition.csv (LV donors per dataset x sex).

Usage: python pool_lv_tensor.py --npz sexdiff_291/lv_donor_L2_gene_meanNorm.npz \
           --map resources/atlas/donor_map.csv --out sexdiff_280/lv_donor_L2_gene_meanNorm.npz
"""

import argparse
import os

import hhca
from hhca.donors import read_donor_map, read_lv_expression, write_lv_expression


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--npz", required=True, help="LV donor tensor of lv_hormone_tensor.py")
    parser.add_argument("--map", required=True, help="donor map (columns raw_donor_id, donor_id)")
    parser.add_argument("--out", required=True, help="pooled tensor to write")
    return parser.parse_args()


def main():
    """Pool the LV donor tensor and write it with the cohort composition."""
    args = parse_args()
    expression = read_lv_expression(args.npz)
    pooled = hhca.tl.pool_lv_expression(expression, read_donor_map(args.map))
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    write_lv_expression(args.out, pooled)
    hhca.tl.cohort_composition(pooled).to_csv(os.path.join(out_dir, "lv_dataset_sex_composition.csv"))
    n_ids, n_donors = (len(x.obs["donor"].cat.categories) for x in (expression, pooled))
    print(f"{n_ids} donor ids -> {n_donors} donors -> {args.out}")


if __name__ == "__main__":
    main()
