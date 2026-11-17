"""Fig 5 k: macrophage pseudo-cells with a perineural position and the rest (perivascular and interstitial).

Perineural = centroid within 30 um of a neural bin of a nerve (component of >= 3 neural 2 um bins at 6 um linkage);
two bars, n and % on top.

Inputs: results/nerve_macrophage/kl/k_counts.json (scripts/nerve_macrophage/perineural_de.py)
Outputs: results/figures/Fig5/panel_k.{png,pdf}

Usage: python figures/Fig5/panel_k.py
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")

import hhca
from hhca.pl import save_panel


def main():
    """Draw and save the panel."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    with open(hhca.results_dir("nerve_macrophage", "kl", "k_counts.json")) as handle:
        counts = json.load(handle)
    fig = hhca.pl.perineural_counts(counts["n_perineural"], counts["n_other"])
    print(save_panel(fig, "Fig5", "k"))


if __name__ == "__main__":
    main()
