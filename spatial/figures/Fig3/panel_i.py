"""Fig 3 i: transmural (epicardium -> endocardium) curves of the data-driven cardiomyocyte gene modules, consensus
of three donors (H0054, H0063: epi, myo, endo; AH10: epi and endo), CMFLAG-clean ventricular-CM 2 um bins.

Per gene, the 45-bin profile of each donor is z-scored; the line is the smoothed donor mean and the band the
spread across donors. Gene sets: the rule of config panels.fig3i_cm_rule (scripts/cardiomyocyte/fig3i_cm_filter.py).

Inputs: results/cardiomyocyte/fig3i/{msearch_*.npz, fig3i_cm_filter_genesets.json}
Outputs: results/figures/Fig3/panel_i.{png,pdf}

Usage: python figures/Fig3/panel_i.py
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import numpy as np

import hhca
from hhca.config import load_config
from hhca.pl import save_panel


def main():
    """Draw the endo-rising and the epi-rising module."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    cfg = load_config("cardiomyocyte")
    tab = hhca.results_dir("cardiomyocyte", "fig3i")
    with open(f"{tab}/fig3i_cm_filter_genesets.json") as handle:
        modules = json.load(handle)[cfg["panels"]["fig3i_cm_rule"]]
    profiles = {
        donor: dict(np.load(f"{tab}/msearch_{donor}.npz", allow_pickle=True)) for donor in cfg["transmural"]["donors"]
    }
    tables = hhca.tl.module_profiles(profiles, modules["endo"] + modules["epi"])
    fig = hhca.pl.transmural_modules(tables, modules, cfg["transmural"]["sigma"])
    print(save_panel(fig, "Fig3", "i"), modules)


if __name__ == "__main__":
    main()
