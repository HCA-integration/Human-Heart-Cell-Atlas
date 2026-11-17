"""Fig 3 f, g, h: NPPA-high vs NPPA-low cardiomyocyte 10 um bins, median distance to the nearest fibroblast-rich
bin (f), vessel-rich bin (g) and endocardium (h, endocardial sections only); stars = BH-adjusted one-sided
Mann-Whitney q.

One output per letter; panel_g.py and panel_h.py call this file with their letter.

Inputs: results/cardiomyocyte/fig3fgh/niche_solidify_pooled.csv (scripts/cardiomyocyte/solidify.py)
Outputs: results/figures/Fig3/panel_{f,g,h}.{png,pdf}

Usage: python figures/Fig3/panel_f.py [--letters f,g,h]
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import pandas as pd

import hhca
from hhca.config import load_config
from hhca.pl import save_panel

METRICS = {"f": "dist_fibroblast_um", "h": "dist_endocardial_um", "g": "dist_vessel_um"}


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--letters", default="f", help="comma-separated panel letters to draw (f, g, h)")
    return parser.parse_args()


def main():
    """Draw the requested panels."""
    args = parse_args()
    cfg = load_config("cardiomyocyte")
    pool = pd.read_csv(hhca.results_dir("cardiomyocyte", "fig3fgh", "niche_solidify_pooled.csv"))
    labels = {
        slide: f"{cfg['sections'][slide]['donor']} {cfg['sections'][slide]['layer']}"
        for slide in dict.fromkeys(pool.slide)
    }
    endo = {slide: labels[slide] for slide in cfg["slides"]["endo_sections"] if slide in labels}
    for letter, metric in METRICS.items():
        if letter not in args.letters.split(","):
            continue
        fig = hhca.pl.niche_distance(pool, metric, endo if letter == "h" else labels)
        print(save_panel(fig, "Fig3", letter))


if __name__ == "__main__":
    main()
