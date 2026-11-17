"""ED6 d: marker genes x Level-3 labels, z-scored mean expression (the boxed block diagonal shows that each
label carries its expected markers).

Profiles: counts summed over the CMFLAG-clean 2 um bins of each decoded Level-3 label, 11 slides; log1p of each
label's profile scaled to 1e4, z-scored per gene across labels (hhca.tl.marker_zscores). Marker genes: ed6d in
config/cardiomyocyte.yaml.

Inputs: results/cardiomyocyte/ed6e/labels/*_label_pseudobulk.csv.gz (driver step label_pb)
Outputs: results/figures/ED6/panel_d.{png,pdf} and results/cardiomyocyte/ed6d_markers_z_visiumhd.csv

Usage: python figures/ED6/panel_d.py
"""

import argparse
import glob

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

import hhca
from hhca.config import load_config
from hhca.pl import save_panel


def main():
    """z-score the marker genes across the labels, write the table and draw the heat map."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    cfg = load_config("cardiomyocyte")
    markers = cfg["ed6d"]["markers"]
    genes = [gene for type_genes in markers.values() for gene in type_genes]
    files = sorted(glob.glob(hhca.results_dir("cardiomyocyte", "ed6e", "labels", "*_label_pseudobulk.csv.gz")))
    profiles = hhca.tl.pool_profiles(pd.read_csv(file, index_col=0) for file in files)
    profiles = profiles.rename(columns=cfg["short_names"]).sort_index(axis=1)
    z = hhca.tl.marker_zscores(profiles, genes)
    print(
        f"{len(genes)} markers x {z.shape[1]} types; z in [{np.nanmin(z.values):.2f}, {np.nanmax(z.values):.2f}]; "
        f"missing genes {[gene for gene in genes if gene not in profiles.index]}"
    )
    z.to_csv(hhca.results_dir("cardiomyocyte", "ed6d_markers_z_visiumhd.csv"))
    fig = hhca.pl.label_markers(z, markers)
    print(save_panel(fig, "ED6", "d"))


if __name__ == "__main__":
    main()
