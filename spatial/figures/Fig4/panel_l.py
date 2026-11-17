"""Fig 4 l: hormone2cell relay, producing Level-4 states -> hormone -> receiving vascular Level-4 states.

The 12 hormones, the tiers and the number of edges per hormone are set in config/atlas_hormones.yaml (`relay`,
`cell_families`); the edges are chosen by hhca.tl.relay_edges. The tables pool cells over all donors.

Inputs: results/atlas/hormones/data_stratified/hrc_datadriven_hormones_level_4__all.csv   (receiver side)
        results/atlas/hormones/collab_hormone/producer_strength_L4_long.csv              (producer side)
Outputs: results/figures/Fig4/panel_l.{png,pdf} and the drawn edges, panel_l_edges_{producer,receiver}.csv

Usage: python figures/Fig4/panel_l.py
"""

import argparse
import os

from hhca import results_dir, thread_env

os.environ.update(thread_env(1, numba=False, keep_existing=True))

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import hhca
from hhca.config import load_config
from hhca.pl import figure_dir, save_panel


def main():
    """Draw panel l and write the drawn edges."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    cfg = load_config("atlas_hormones")
    relay, families = cfg["relay"], cfg["cell_families"]
    producer_edges, receiver_edges = hhca.tl.relay_edges(
        pd.read_csv(results_dir("atlas", "hormones", "data_stratified", "hrc_datadriven_hormones_level_4__all.csv")),
        pd.read_csv(results_dir("atlas", "hormones", "collab_hormone", "producer_strength_L4_long.csv")),
        relay["hormones"],
        families,
        relay["receiver_families"],
        tiers=relay["tiers"],
        top_n_producers=relay["top_n_producers"],
        top_n_receivers=relay["top_n_receivers"],
        display=relay["display"],
    )
    save_panel(hhca.pl.hormone_relay(producer_edges, receiver_edges, families, relay["hormones"]), "Fig4", "l")
    folder = figure_dir("Fig4")
    producer_edges.to_csv(f"{folder}/panel_l_edges_producer.csv", index=False)
    receiver_edges.to_csv(f"{folder}/panel_l_edges_receiver.csv", index=False)
    print(
        f"hormones {receiver_edges.Hormone.nunique()}, producers {producer_edges.Celltype.nunique()}, "
        f"receivers {receiver_edges.Celltype.nunique()}"
    )


if __name__ == "__main__":
    main()
