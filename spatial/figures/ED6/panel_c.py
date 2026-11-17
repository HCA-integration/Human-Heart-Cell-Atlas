"""ED6 c, Visium row: 55 um standard-Visium spots of HHH2-LV-endo-1 drawn as pies of the spotiphy proportions.

This is the first row of ED6 c; the Visium HD and Xenium rows are drawn by their own pipelines. The window is
set in config/deconvolution.yaml (figures.ed6_c_fov).

Inputs:  results/deconvolution/figure_inputs/ (per-spot proportions and scale factors of the section, copied
         there by workflow/hpc/deconvolution_downstream.sbatch)
Outputs: results/figures/ED6/panel_c_visium.{png,pdf}

Usage: python figures/ED6/panel_c.py
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import pandas as pd

import hhca
from hhca.config import load_config
from hhca.pl import save_panel

METHOD = "spotiphy"
CELLTYPES = [
    "Adipocytes",
    "Arterial Endothelial cells",
    "Atrial Cardiomyocytes",
    "B cells",
    "Capillary Endothelial cells",
    "Dendritic cells",
    "Endocardial Endothelial Cells",
    "Epicardium",
    "Fibroblasts",
    "ILC",
    "Lymphatic Endothelial Cells",
    "Lymphoid Cells (Proliferating)",
    "Macrophages",
    "Mast cells",
    "Monocytes",
    "NK cells",
    "Neural Cells",
    "Pericytes",
    "Plasma Cells",
    "T cells",
    "Vascular Smooth Muscle cells",
    "Venous Endothelial cells",
    "Ventricular Cardiomyocytes",
]


def main():
    """Draw the spotiphy proportion pies of the spots in the window."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    cfg = load_config("deconvolution")["figures"]
    slide = cfg["ed6_slide"]
    input_dir = hhca.results_dir("deconvolution", "figure_inputs")
    coords = pd.read_csv(f"{input_dir}/precomputed/NNLS/{slide}_L3_deconv.csv.gz", index_col=0)[["x", "y"]]
    proportions = hhca.tl.read_proportions(f"{input_dir}/precomputed/{METHOD}/{slide}.csv.gz", CELLTYPES)
    adata = hhca.tl.collect_proportions({METHOD: proportions.reindex(coords.index)})
    adata.obsm["spatial"] = coords.to_numpy()
    with open(f"{input_dir}/{slide}/spatial/scalefactors_json.json") as handle:
        adata.uns["spatial"] = {slide: {"scalefactors": json.load(handle)}}
    fig = hhca.pl.spot_pies(adata, METHOD, cfg["ed6_c_fov"])
    print(save_panel(fig, "ED6", "c_visium"))


if __name__ == "__main__":
    main()
