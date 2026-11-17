"""Pool the per-donor cytokine-programme scores to 280 donors and test each programme against age.

The stream stores per-(donor x cell type) mean programme scores and nucleus counts, so duplicate donors are
pooled analytically: pooled_mean = sum(mean_k n_k) / sum(n_k). Per-donor age is the nucleus-weighted median,
sex, tissue and assay the nucleus-weighted mode (hhca.tl.pool_programme_scores). For each immune cell type and
programme, adult donors with at least `cytokine.min_cells` nuclei: OLS score ~ age + C(sex) (age p), Spearman rho
for display, p_sens from score ~ age + C(sex) + C(tissue) + C(assay), BH across all tests (hhca.tl.age_effects).
Cell types and thresholds: config/config.yaml (cytokine, donors).

Inputs: per_donor_cytokine_scores.npz and age_cytokine_effects.csv (donor_cytokine_scores.py),
per_donor_celltype_scores.npz (donor_geneset_scores.py), resources/atlas/donor_map.csv.
Outputs: age_cytokine_effects_280.csv (Fig 5 m), per_donor_cytokine_scores_280.npz (Fig 5 n).

Usage: python cytokine_age_effects.py --cytokine-npz per_donor_cytokine_scores.npz \
           --celltype-npz per_donor_celltype_scores.npz --stream-table age_cytokine_effects.csv \
           --map resources/atlas/donor_map.csv --out-table age_cytokine_effects_280.csv \
           --out-npz per_donor_cytokine_scores_280.npz
"""

import argparse

import numpy as np
import pandas as pd

import hhca
from hhca.config import load_config
from hhca.donors import read_donor_map, read_programme_scores, write_programme_scores


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cytokine-npz", required=True, help="per-donor cytokine scores of donor_cytokine_scores.py")
    parser.add_argument(
        "--celltype-npz",
        required=True,
        help="per-donor gene-set scores of donor_geneset_scores.py (donor tissue, assay, nuclei)",
    )
    parser.add_argument(
        "--stream-table",
        required=True,
        help="age_cytokine_effects.csv of donor_cytokine_scores.py (n_present, myeloid proxy, column order)",
    )
    parser.add_argument("--map", required=True, help="donor map (columns raw_donor_id, donor_id)")
    parser.add_argument("--out-table", required=True, help="age-effect table to write")
    parser.add_argument("--out-npz", required=True, help="pooled per-donor scores to write")
    return parser.parse_args()


def main():
    """Pool the donors, test every programme against age and write the table and the pooled scores."""
    args = parse_args()
    cfg = load_config("config")
    stream = pd.read_csv(args.stream_table)
    key = ["feature", "celltype"]

    donors, scores = read_programme_scores(args.cytokine_npz)
    celltype = np.load(args.celltype_npz, allow_pickle=True)
    for column in ("tissue", "assay", "ncells"):
        donors[column] = celltype[f"donor_{column}"]
    pooled, pooled_donors = hhca.tl.pool_programme_scores(scores, donors, read_donor_map(args.map))
    table = hhca.tl.age_effects(
        pooled,
        pooled_donors,
        cfg["cytokine"]["age_cell_types"],
        cfg["cytokine"]["min_cells"],
        cfg["donors"]["adult_age"],
    )
    table = table.merge(stream[key + ["n_present", "nearest_myeloid_proxy"]], on=key, how="left")
    table[stream.columns.tolist()].to_csv(args.out_table, index=False)
    write_programme_scores(args.out_npz, pooled_donors, pooled, extra=("tissue", "assay"))
    print(f"{len(pooled_donors)} donors; wrote {args.out_table} ({len(table)} rows) and {args.out_npz}")


if __name__ == "__main__":
    main()
