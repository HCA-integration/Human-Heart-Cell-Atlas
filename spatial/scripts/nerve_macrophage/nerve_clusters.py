#!/usr/bin/env python
"""Distance of each macrophage pseudo-cell to the nearest nerve, for nerves of increasing minimum size.

A nerve is a cluster of neural 2 um bins: per slide, the 'Neural Cells' bins are grouped into connected components
at 6 um and only components with at least k bins are kept, so single stray bins do not count as nerves
(hhca.tl.call_nerves, hhca.tl.nerve_distance). The distance from every pseudo-cell to the nearest kept neural bin
is computed for k = 1, 3, 5, 10, and the association of SPP1, F13A1 and LYVE1 with that distance is fitted for each
k (slide fixed effects and library size; a second model also adjusts for the nearest vessel structure and its
calibre class, and is NaN when those columns are absent, i.e. macrophage_pseudocells.py --caliber none;
hhca.tl.marker_effects). Fig 5 k uses dist_nerve_k3 < 30 um as the perineural rule (perineural_de.py). Sizes,
markers and distance bins: config `macrophage`.

Inputs: the per-cell covariate table (macrophage_covariates.py) and the CMFLAG-clean 2 um h5ads.
Outputs: <outdir>/tables/nerve_cluster_retention.csv, spp1_nerve_by_k.csv, percell_nerve_distances_by_k.csv and
<outdir>/figures/nerve_cluster_size.{png,pdf}.

Usage: python nerve_clusters.py --cov macrophage_percell_covariates.csv --h5ad '<dir>/{s}.h5ad' --outdir DIR
"""

import argparse
import os

from hhca import thread_env

os.environ.update(thread_env(1))

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import hhca
from hhca.config import load_config
from hhca.log import get_logger

log = get_logger()


def add_covariates(table, calibre_rank):
    """Add the z-scored library size, calibre class and distance to the nearest vessel structure."""
    table["logcounts"] = np.log(table.total_counts)
    table["z_lc"] = (table.logcounts - table.logcounts.mean()) / table.logcounts.std()
    rank = table.class_near.map(calibre_rank)
    table["z_cal"] = (rank - rank.mean()) / rank.std()
    table["z_dv"] = (
        table.dist_vessel_struct_um - table.dist_vessel_struct_um.mean()
    ) / table.dist_vessel_struct_um.std()


def fit_markers(table, markers, min_sizes):
    """Slope of each marker on the nerve distance, for each minimum nerve size."""
    adjusted = table["z_dv"].notna().any() and table["z_cal"].notna().any()
    fits = []
    for k in min_sizes:
        distance = table[f"dist_nerve_k{k}"]
        data = table.assign(z_dn=(distance - distance.mean()) / distance.std())
        # nerve distance alone, with slide fixed effects and library size
        alone = hhca.tl.marker_effects(data, markers, "z_dn").set_index("marker")
        fit = pd.DataFrame(
            {
                "marker": markers,
                "k": k,
                "beta_uni": alone.beta.values,
                "p_uni": alone.p.values,
                "ci_lo": alone.ci_lo.values,
                "ci_hi": alone.ci_hi.values,
                "beta_full": np.nan,
                "p_full": np.nan,
                "partial_dR2_full": np.nan,
                "median_dist_um": float(np.nanmedian(distance)),
            }
        )
        if adjusted:
            # mutually adjusted: nerve distance, distance to the nearest vessel structure and its calibre class;
            # partial dR2 = R2 gained by the nerve term
            vessels = ("z_dv", "z_cal", "z_lc")
            full = hhca.tl.marker_effects(data, markers, "z_dn", adjust=vessels)
            without = hhca.tl.marker_effects(data, markers, None, adjust=vessels)
            fit["beta_full"], fit["p_full"] = full.beta.values, full.p.values
            fit["partial_dR2_full"] = full.r2.values - without.r2.values
        fits.append(fit)
    fits = pd.concat(fits)
    # one block per marker, the minimum sizes in order
    return pd.concat([fits[fits.marker == marker] for marker in markers], ignore_index=True)


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cov", required=True, help="per-cell covariate table of macrophage_covariates.py")
    parser.add_argument("--h5ad", required=True, help="CMFLAG-clean 2 um h5ad, path template with {s} for the slide")
    parser.add_argument("--outdir", required=True, help="output folder; tables/ and figures/ are created in it")
    return parser.parse_args()


def main():
    """Measure the nerve distances, fit the markers and write the tables and the figure."""
    args = parse_args()
    cfg = load_config("nerve_macrophage")
    nerve, macrophage = cfg["nerve"], cfg["macrophage"]
    min_sizes, markers, profile = (
        macrophage["nerve_min_sizes"],
        macrophage["distance_markers"],
        macrophage["distance_profile"],
    )
    outdir = args.outdir
    os.makedirs(f"{outdir}/tables", exist_ok=True)
    os.makedirs(f"{outdir}/figures", exist_ok=True)
    table = pd.read_csv(args.cov, index_col=0).reset_index(drop=True)

    neural_bins = {}
    for slide in table.slide.unique():
        adata = ad.read_h5ad(args.h5ad.format(s=slide), backed="r")
        neural_bins[slide] = hhca.tl.call_nerves(adata, nerve["neural_type"], nerve["linkage_um"], nerve["min_bins"])
    retention = hhca.tl.nerve_distance(table, neural_bins, min_sizes)
    retention.to_csv(f"{outdir}/tables/nerve_cluster_retention.csv", index=False)
    print(retention.to_string(index=False))
    print("\nneural bins kept (median % per slide):")
    for k in min_sizes:
        pct = 100 * retention[f"n_bins_k{k}"] / retention["n_neural"]
        print(f"  k>={k:2d}: median {pct.median():.1f}%  range {pct.min():.0f}-{pct.max():.0f}%")

    add_covariates(table, macrophage["calibre_rank"])
    fits = fit_markers(table, markers, min_sizes)
    fits.to_csv(f"{outdir}/tables/spp1_nerve_by_k.csv", index=False)
    print(fits.to_string(index=False))

    profiles = {
        k: hhca.tl.expression_by_distance(table, "SPP1_log1p_cp10k", f"dist_nerve_k{k}", profile["edges_um"])
        for k in min_sizes
    }
    fig = hhca.pl.nerve_size_effect(profiles, fits, profile["centres_um"])
    for extension in ("png", "pdf"):
        fig.savefig(f"{outdir}/figures/nerve_cluster_size.{extension}", dpi=300)
    print(f"\nwrote {outdir}/figures/nerve_cluster_size.png")

    # per-cell distances for every minimum nerve size
    table[
        ["slide", "centroid_x_um", "centroid_y_um", "SPP1_log1p_cp10k", "dist_nerve_um"]
        + [f"dist_nerve_k{k}" for k in min_sizes]
    ].to_csv(f"{outdir}/tables/percell_nerve_distances_by_k.csv", index=False)
    print(f"wrote {outdir}/tables/percell_nerve_distances_by_k.csv")


if __name__ == "__main__":
    main()
