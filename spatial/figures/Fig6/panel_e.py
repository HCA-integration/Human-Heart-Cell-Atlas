"""Fig 6 e: fraction of immune cells in DBSCAN foci vs ISHLT cellular grade, per TMA core.

One point per TMA core, coloured by rejection type, with the median per grade and the Spearman correlation.

Inputs:
    results/xenium/downstream/stats/fig6e_cores.csv, fig6e_cores_stats.json (fig6e_cores.py)

Outputs:
    results/figures/Fig6/panel_e.{png,pdf}

Usage: python figures/Fig6/panel_e.py
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
    cores = pd.read_csv(f"{stats}/fig6e_cores.csv")
    with open(f"{stats}/fig6e_cores_stats.json") as handle:
        test = json.load(handle)
    fig = hhca.pl.foci_by_grade(cores, test["rho"], test["p"])
    print(f"cores n={test['n']} rho={test['rho']:.3f} p={test['p']:.3g} ->", save_panel(fig, "Fig6", "e"))


if __name__ == "__main__":
    main()
