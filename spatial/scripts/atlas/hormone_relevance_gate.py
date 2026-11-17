#!/usr/bin/env python
"""Relevance gate for the Fig 2 h-i dots: which (hormone, cell system) pairs are shown.

Relevance: for each (hormone, system) and sex, strength = mean Level-4 `Strength` over the system's Level-4
cell types (0 if the hormone is absent for that system); R = max(male strength, female strength).
Threshold, per panel: the 25th percentile of the non-zero R values of the panel's 4 x N matrix. A (hormone,
system) pair passes if R >= threshold and R > 0; it is shown if it passes and its hormone is among the top
hormones by max R (the two oestrogen columns are always kept). The rule is hhca.tl.relevance_gate; cell systems,
quantile and number of hormones: config/atlas_hormones.yaml (cell_systems, relevance_gate).

Inputs: the hormone2cell_stratified.py tables hrc_datadriven_hormones_level_4__LV_{male,female}.csv and
producer_strength_L4_long__LV_{male,female}.csv.
Outputs: hormone_sexdiff_relevancegated__{received,produced}.csv, read by lv_hormone_tensor.py,
hormone_sexdiff_significance.py and figures/Fig2/panel_hi.py.

Usage: python hormone_relevance_gate.py --data results/atlas/hormones/data_stratified \
           --outdir results/atlas/hormones/g7_sexdiff
"""

import argparse
import os

import pandas as pd

import hhca
from hhca.config import load_config

TABLES = {"received": "hrc_datadriven_hormones_level_4__LV_{}.csv", "produced": "producer_strength_L4_long__LV_{}.csv"}


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True, help="directory with the hormone2cell_stratified.py tables")
    parser.add_argument("--outdir", required=True, help="directory for the gate tables")
    return parser.parse_args()


def main():
    """Apply the relevance gate to both panels and write the gate tables."""
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cfg = load_config("atlas_hormones")
    print("relevance gate (R = max over sexes of the mean Level-4 strength)")
    for panel, name in TABLES.items():
        male, female = (pd.read_csv(os.path.join(args.data, name.format(sex))) for sex in ("male", "female"))
        gate = hhca.tl.relevance_gate(male, female, cfg["cell_systems"], **cfg["relevance_gate"])
        gate.to_csv(os.path.join(args.outdir, f"hormone_sexdiff_relevancegated__{panel}.csv"), index=False)
        print(
            f"  {panel}: threshold = {gate.attrs['threshold']:.4f}, {int(gate.relevance_pass.sum())}/{len(gate)} "
            f"(hormone, system) pairs pass, {int(gate.shown_in_figure.sum())} shown"
        )
    print("wrote the gate tables to", args.outdir)


if __name__ == "__main__":
    main()
