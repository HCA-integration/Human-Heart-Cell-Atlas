"""ED6 g: per-spot deconvolution of one standard-Visium section (HHH2-LV-endo-1) by seven methods.

Top row: dominant cell type per spot on the whole section; bottom row: proportion pies in the zoom window set
in config/deconvolution.yaml (figures.ed6_g_fov).

Inputs:  results/deconvolution/figure_inputs/precomputed/<method>/HHH2-LV-endo-1.csv.gz (per-spot proportions) and
         results/deconvolution/figure_inputs/HHH2-LV-endo-1/spatial/ (H&E image and scale factors), copied there
         by workflow/hpc/deconvolution_downstream.sbatch
Outputs: results/figures/ED6/panel_g.{png,pdf}

Usage: python figures/ED6/panel_g.py
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import hhca
from hhca.config import load_config
from hhca.pl import save_panel

METHODS = ["spotiphy", "RCTD", "cell2location", "ucaspatial", "stereoscope", "SPOTlight", "NNLS"]

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
    """Draw the H&E reference column and, per method, the dominant-type map and the proportion pies of the zoom."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    cfg = load_config("deconvolution")["figures"]
    slide = cfg["ed6_slide"]
    input_dir = hhca.results_dir("deconvolution", "figure_inputs")
    nnls_path = f"{input_dir}/precomputed/NNLS/{slide}_L3_deconv.csv.gz"
    paths = {method: f"{input_dir}/precomputed/{method}/{slide}.csv.gz" for method in METHODS}
    paths["NNLS"] = nnls_path
    adata = hhca.tl.collect_proportions(
        {method: hhca.tl.read_proportions(path, CELLTYPES) for method, path in paths.items()}
    )
    adata.obsm["spatial"] = pd.read_csv(nnls_path, index_col=0).loc[adata.obs_names, ["x", "y"]].to_numpy()
    with open(f"{input_dir}/{slide}/spatial/scalefactors_json.json") as handle:
        scalefactors = json.load(handle)
    image = plt.imread(f"{input_dir}/{slide}/spatial/tissue_hires_image.png")
    adata.uns["spatial"] = {slide: {"scalefactors": scalefactors, "images": {"hires": image}}}
    fig = hhca.pl.method_maps(adata, cfg["ed6_g_fov"], METHODS)
    print(save_panel(fig, "ED6", "g"))


if __name__ == "__main__":
    main()
