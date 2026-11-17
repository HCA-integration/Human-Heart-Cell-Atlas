"""Fig 3 i, heavy pass: 45-bin transmural expression profiles of every gene in CMFLAG-clean ventricular
cardiomyocyte bins.

Per donor (H0054 and H0063: epi, myo, endo sections; AH10: epi and endo) and section, each ventricular-CM 2 um bin
gets a within-section transmural coordinate (Laplace solution between the epicardial and endocardial face; the
face orientation comes from marker genes) and the wall is cut into 45 bins, 15 per section; log1p CP10k is summed
per bin for all genes (hhca.tl.transmural_profiles). H0054 and H0063 share one gene space; AH10 has its own, so
each donor keeps its gene names and genes are matched by name later (select_transmural_genes.py). Donors, marker
genes and parameters: `transmural` in config/cardiomyocyte.yaml.

Inputs: the CMFLAG-clean 2 um h5ads of the donors' sections (config paths.h5ad_2um).
Outputs: in --out-dir, msearch_<donor>.npz (genes, bin_sum [45 x G], bin_count [45], nnz [G] = bins with counts
per gene, n_cm) and msearch_phi_diagnostics.csv.

Usage: compute_marker_search.py --out-dir DIR
"""

import os

from hhca import thread_env

os.environ.update(thread_env(1))

import argparse

import anndata as ad
import numpy as np
import pandas as pd

import hhca
from hhca.config import load_config, section_path
from hhca.log import get_logger

log = get_logger("msearch")


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="Fig 3 i heavy pass: 45-bin transmural sums for all genes")
    parser.add_argument("--out-dir", required=True, help="output folder")
    return parser.parse_args()


def main():
    """Compute the profiles of every donor and write the diagnostics table."""
    args = parse_args()
    cfg = load_config("cardiomyocyte")
    params = cfg["transmural"]
    os.makedirs(args.out_dir, exist_ok=True)
    diagnostics = []
    for donor in params["donors"]:
        slides = {section["layer"]: slide for slide, section in cfg["sections"].items() if section["donor"] == donor}
        slides = {layer: slides[layer] for layer in params["layers"] if layer in slides}
        profile = hhca.tl.transmural_profiles(
            {layer: ad.read_h5ad(section_path(cfg, "h5ad_2um", slide), backed="r") for layer, slide in slides.items()},
            params["endo_markers"],
            params["epi_markers"],
            layers=params["layers"],
            bins_per_section=params["bins_per_section"],
            cell_type=cfg["cell_types"]["ventricular"],
            grid_um=params["grid_um"],
            face_fraction=params["face_fraction"],
            log=log,
        )
        for row in profile.pop("diagnostics"):
            diagnostics.append(
                {
                    "donor": donor,
                    "layer": row["layer"],
                    "slide": slides[row["layer"]],
                    "cmflag_applied": False,  # the h5ads are CMFLAG-clean already, no flag table is joined
                    "n_cm": row["n_cm"],
                    "marker_axis_R2": row["marker_axis_R2"],
                    "phi_ok": row["phi_ok"],
                }
            )
        np.savez_compressed(
            f"{args.out_dir}/msearch_{donor}.npz",
            **profile,
            N_BINS=len(params["layers"]) * params["bins_per_section"],
            N_PER_BLOCK=params["bins_per_section"],
        )
        log(f"[{donor}] saved msearch_{donor}.npz  n_cm_tot={profile['n_cm']} G={profile['genes'].size}")
    pd.DataFrame(diagnostics).to_csv(f"{args.out_dir}/msearch_phi_diagnostics.csv", index=False)
    log("done ->", args.out_dir)


if __name__ == "__main__":
    main()
