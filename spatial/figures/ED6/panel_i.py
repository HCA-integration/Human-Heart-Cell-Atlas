"""ED6 i: healthy Xenium section (10x Human Non-diseased Heart FFPE) coloured by CellPin + HHCA label, vascular zoom.

Cardiomyocytes are pooled; endothelial subtypes are shown only for EC-marker-positive cells, the rest is drawn
grey as "EC label, marker-negative" (gate in config ed6i).

Inputs:
    results/xenium/downstream/healthy/ed6i_cells.csv (healthy_display_classes.py)
    config/xenium_downstream.yaml: ed6i

Outputs:
    results/figures/ED6/panel_i.{png,pdf}

Usage: python figures/ED6/panel_i.py
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
    cells = pd.read_csv(hhca.results_dir("xenium", "downstream", "healthy", "ed6i_cells.csv"))
    fig = hhca.pl.healthy_section(cells, load_config("xenium_downstream")["ed6i"]["ec_negative_label"])
    print(f"{len(cells):,} cells; classes: {cells.display.value_counts().to_dict()}")
    print(save_panel(fig, "ED6", "i"))


if __name__ == "__main__":
    main()
