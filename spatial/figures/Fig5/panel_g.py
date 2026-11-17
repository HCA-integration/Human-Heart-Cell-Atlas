"""Fig 5 g: aggregate drug2cell score over the peripheral-neurotoxic (CIPN) drugs by nerve myelination.

Myelinated vs non-myelinated nerves; significance stars = one-sided Mann-Whitney. drug2cell scores with the myelin
genes (MBP, PLP1, MOG, MPZ, PMP22) removed from the target sets.

Inputs: results/nerve_macrophage/gh/g_per_nerve.csv, g_summary.json (scripts/nerve_macrophage/cipn_drug2cell.py)
Outputs: results/figures/Fig5/panel_g.{png,pdf}

Usage: python figures/Fig5/panel_g.py
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import pandas as pd

import hhca
from hhca.pl import save_panel


def main():
    """Draw and save the panel."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    per_nerve = pd.read_csv(hhca.results_dir("nerve_macrophage", "gh", "g_per_nerve.csv"), index_col=0)
    with open(hhca.results_dir("nerve_macrophage", "gh", "g_summary.json")) as handle:
        summary = json.load(handle)
    fig = hhca.pl.neurotoxic_score(per_nerve, summary["mwu_p_greater"])
    print(save_panel(fig, "Fig5", "g"))


if __name__ == "__main__":
    main()
