"""ED9 b (4 x 2) and c (2 x 4): cytokine-programme scores by rejection type.

Inputs:
    results/xenium/downstream/stats/fig6_cytokine_biopsy_values.csv, fig6_cytokine_panels_stats.csv
    (cytokine_by_rejection.py)
    config/xenium_downstream.yaml: figures.ed9b_programmes, figures.ed9c_programmes

Outputs:
    results/figures/ED9/panel_b.{png,pdf}, panel_c.{png,pdf}

Usage: python figures/ED9/panel_bc.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import pandas as pd

import hhca
from hhca.config import load_config
from hhca.pl import save_panel


def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    figures = load_config("xenium_downstream")["figures"]
    stats = hhca.results_dir("xenium", "downstream", "stats")
    scores = pd.read_csv(f"{stats}/fig6_cytokine_biopsy_values.csv")
    tests = pd.read_csv(f"{stats}/fig6_cytokine_panels_stats.csv").set_index("column")
    for panel, programmes, shape in [
        ("b", figures["ed9b_programmes"], (4, 2)),
        ("c", figures["ed9c_programmes"], (2, 4)),
    ]:
        fig = hhca.pl.programme_scores(scores, tests, programmes, shape)
        print(panel, "->", save_panel(fig, "ED9", panel))


if __name__ == "__main__":
    main()
