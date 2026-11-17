"""Fig 6 i: IL-1beta programme score by rejection type (config figures.fig6i_programme).

Inputs:
    results/xenium/downstream/stats/fig6_cytokine_biopsy_values.csv, fig6_cytokine_panels_stats.csv
    (cytokine_by_rejection.py)
    config/xenium_downstream.yaml: figures.fig6i_programme

Outputs:
    results/figures/Fig6/panel_i.{png,pdf}

Usage: python figures/Fig6/panel_i.py
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
    programme = load_config("xenium_downstream")["figures"]["fig6i_programme"]
    stats = hhca.results_dir("xenium", "downstream", "stats")
    scores = pd.read_csv(f"{stats}/fig6_cytokine_biopsy_values.csv")
    tests = pd.read_csv(f"{stats}/fig6_cytokine_panels_stats.csv").set_index("column")
    fig = hhca.pl.programme_scores(scores, tests, programme)
    kruskal = tests.loc[f"score_{programme}"]
    print(
        f"{programme}: KW p={kruskal.kruskal_p:.3g} q={kruskal.kruskal_q_BH_over_18:.3g} ->",
        save_panel(fig, "Fig6", "i"),
    )


if __name__ == "__main__":
    main()
