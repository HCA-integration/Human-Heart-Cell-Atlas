"""Donor-level significance of the male-female differences behind the Fig 2 h-i dots.

The donor is the unit of replication, FDR is taken across dots, and a dataset term separates sex differences
from between-study composition. For every shown dot (shown_in_figure in the gate tables of
hormone_relevance_gate.py; panel h = received, panel i = produced) the per-donor system score is
  received: mean normcounts over the hormone's receptor genes, cell-weighted across the system's Final Level 2
            types (the receptor set is an OR set, so the mean, as for the drug-target scores);
  produced: hormone2cell producer logic: max over alternative routes ('|') of the min over co-required
            genes ('+') of the biosynthesis-gene expression, cell-weighted across the system's L2 types.
Scores use LV nuclei only. Donors: adults (>= 20 years) of known sex with >= 10 nuclei of the system.

Per dot: Hedges g (male - female), difference of means, Mann-Whitney U, OLS score ~ C(sex) + age, the same
model with C(dataset), direction agreement across datasets with both sexes, number of single-sex datasets.
BH-FDR on the OLS sex p within each panel, and a robustness tier (hhca.tl.sex_differences; cell systems and
thresholds: config/atlas_hormones.yaml, system_level2 and sexdiff).

Inputs: the pooled LV donor tensor (lv_hormone_tensor.py, pool_lv_tensor.py), its hormone_genesets.json, the
gate tables, donor_meta.csv (donor_meta.py).
Outputs: one row per dot (csv).

Usage: python hormone_sexdiff_significance.py --tensor lv_donor_L2_gene_meanNorm.npz \
           --genesets-json hormone_genesets.json --gated results/atlas/hormones/g7_sexdiff \
           --donor-meta results/atlas/donor_meta.csv --out hormone_sexdiff_significance_ALLDOTS.csv
"""

import argparse
import json
import os

import pandas as pd

import hhca
from hhca.config import load_config
from hhca.donors import read_lv_expression

TIERS = ["A_robust", "B_cohort_ok_weakdir", "C_fdr_only", "D_ns", "NA_underpowered"]


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tensor", required=True, help="pooled lv_donor_L2_gene_meanNorm.npz")
    parser.add_argument("--genesets-json", required=True, help="hormone_genesets.json written by lv_hormone_tensor.py")
    parser.add_argument(
        "--gated", required=True, help="directory with hormone_sexdiff_relevancegated__{received,produced}.csv"
    )
    parser.add_argument("--donor-meta", required=True, help="donor_meta.csv (donor, sex, age, ...)")
    parser.add_argument("--out", required=True, help="table to write, one row per dot")
    return parser.parse_args()


def main():
    """Test every shown dot of both panels and write one row per dot."""
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cfg = load_config("atlas_hormones")
    params = cfg["sexdiff"]
    expression = read_lv_expression(args.tensor)
    with open(args.genesets_json) as handle:
        gene_sets = json.load(handle)["genesets"]
    donors = pd.read_csv(args.donor_meta).set_index("donor")

    results = []
    for panel in ("received", "produced"):
        result = hhca.tl.sex_differences(
            expression,
            donors,
            pd.read_csv(f"{args.gated}/hormone_sexdiff_relevancegated__{panel}.csv"),
            gene_sets[panel],
            cfg["system_level2"],
            panel,
            min_cells=params["min_cells"],
            min_per_group=params["min_per_group"],
            adult_min_age=params["adult_min_age"],
            min_datasets=params["min_datasets"],
        )
        counts = result.tier.value_counts().to_dict()
        print(f"panel {panel}: {len(result)} dots ({int(result.testable.sum())} testable)")
        print("  " + ", ".join(f"{tier}: {counts.get(tier, 0)}" for tier in TIERS))
        results.append(result)
    pd.concat(results, ignore_index=True).to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
