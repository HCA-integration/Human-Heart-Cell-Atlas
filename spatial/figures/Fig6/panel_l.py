"""Fig 6 l: ATG drug-target score vs immune fraction (l1) and vs T-cell fraction (l2), coloured by ISHLT grade.

All 108 biopsies. The ATG score is defined in drug_targets.py (score_genes on the ATG target set, per-core mean).
The correlations are close to definitional (target genes are T-cell genes).

Inputs:
    results/xenium/downstream/stats/fig6_biopsies.csv, fig6_stats.json (fig6_stats.py)

Outputs:
    results/figures/Fig6/panel_l1.{png,pdf}, panel_l2.{png,pdf}

Usage: python figures/Fig6/panel_l.py
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import pandas as pd

import hhca
from hhca.pl import save_panel


def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    stats = hhca.results_dir("xenium", "downstream", "stats")
    biopsies = pd.read_csv(f"{stats}/fig6_biopsies.csv")
    with open(f"{stats}/fig6_stats.json") as handle:
        tests = json.load(handle)["l"]
    for column, xlabel, panel, key in [("immf", "immune fraction", "l1", "l"), ("tf", "T-cell fraction", "l2", "l2")]:
        fig = hhca.pl.atg_score(biopsies, column, xlabel, tests[key]["rho"], tests[key]["p"])
        print(f"6{panel}: n={tests[key]['n']} rho={tests[key]['rho']:.2f} ->", save_panel(fig, "Fig6", panel))


if __name__ == "__main__":
    main()
