"""Fig 5 h: per-drug drug2cell target score, myelinated minus non-myelinated nerves.

Peripheral-neurotoxic drugs with targets detected in nerves; bars dark when BH-adjusted p < 0.05 (two-sided
Mann-Whitney), stars = BH-adjusted p. drug2cell scores with the myelin genes (MBP, PLP1, MOG, MPZ, PMP22) removed
from the target sets.

Inputs: results/nerve_macrophage/gh/h_stats.csv (scripts/nerve_macrophage/cipn_drug2cell.py)
Outputs: results/figures/Fig5/panel_h.{png,pdf}

Usage: python figures/Fig5/panel_h.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import pandas as pd

import hhca
from hhca.pl import save_panel


def main():
    """Draw and save the panel."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    drugs = pd.read_csv(hhca.results_dir("nerve_macrophage", "gh", "h_stats.csv"))
    fig = hhca.pl.neurotoxic_drugs(drugs)
    print(save_panel(fig, "Fig5", "h"))


if __name__ == "__main__":
    main()
