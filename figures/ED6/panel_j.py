"""ED6 j: rejection Xenium, immune infiltrate by ISHLT cellular grade on two cores (Patient_12_2 1R, Patient_54_1 3R).

Inputs:
    results/xenium/downstream/tables/spatial_cells.csv.gz (spatial_cells.py; per-core grade from the GEO metadata)
    config/xenium_downstream.yaml: figures.ed6j_cores

Outputs:
    results/figures/ED6/panel_j.{png,pdf}

Usage: python figures/ED6/panel_j.py
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
    cores = load_config("xenium_downstream")["figures"]["ed6j_cores"]
    cells = pd.read_csv(hhca.results_dir("xenium", "downstream", "tables", "spatial_cells.csv.gz"))
    fig = hhca.pl.core_lineages(cells, cores)
    print(cores, "->", save_panel(fig, "ED6", "j"))


if __name__ == "__main__":
    main()
