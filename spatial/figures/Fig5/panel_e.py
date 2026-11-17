"""Fig 5 e: per-nerve myelination score by nerve size class, all slides.

Percentages = myelinating nerves (myelin_total >= 3) per class.

Inputs: results/nerve_macrophage/e/per_nerve.csv (scripts/nerve_macrophage/myelination_size.py)
Outputs: results/figures/Fig5/panel_e.{png,pdf}

Usage: python figures/Fig5/panel_e.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import pandas as pd

import hhca
from hhca.config import load_config
from hhca.pl import save_panel


def main():
    """Draw and save the panel."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    nerve = load_config("nerve_macrophage")["nerve"]
    nerves = pd.read_csv(hhca.results_dir("nerve_macrophage", "e", "per_nerve.csv"))
    fig = hhca.pl.myelination_by_size(nerves, nerve["size_classes"], nerve["myelinated_min"])
    print(save_panel(fig, "Fig5", "e"))


if __name__ == "__main__":
    main()
