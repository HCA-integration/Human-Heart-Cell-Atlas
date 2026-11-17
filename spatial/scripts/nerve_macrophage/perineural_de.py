#!/usr/bin/env python
"""Fig 5 k, l data: perineural vs other macrophage pseudo-cells.

Perineural = dist_col < radius (config `macrophage`: nerve_clusters.py dist_nerve_k3 < 30 um = centroid within
30 um of a neural bin of a component of >= 3 neural bins at 6 um linkage); the other pseudo-cells are perivascular
or interstitial.

Outputs (<outdir>/):
  k_counts.json      perineural / other counts and percentages (Fig 5 k), per slide
  l_de_markers.csv   markers: log2FC, z score, p, BH-adjusted p, pct in each group (scanpy rank_genes_groups,
                     Wilcoxon, normal approximation, pts=True, BH over all genes, on normalize_total 1e4 + log1p;
                     perineural vs rest)
  l_de_all.csv.gz    the same for every gene
  l_regression.csv   per marker: log1p CP10k ~ z(nerve distance) + z(log library) + slide fixed effects, OLS with
                     HC3 errors (hhca.tl.marker_effects; library = obs total_counts, marker = log1p CP10k over the
                     genes of the pseudo-cell h5ad), and the same with the binary perineural indicator (Fig 5 l)
  l_per_slide.csv    per slide and marker: n, mean log1p CP10k in each group (hhca.tl.marker_means)

Inputs: the pooled pseudo-cell h5ad (macrophage_pseudocells.py) and the per-cell nerve distances
(nerve_clusters.py).

Usage: python perineural_de.py --cache Macrophage_pseudocells_allslides.h5ad
           --distances percell_nerve_distances_by_k.csv --outdir DIR
"""

import argparse
import json
import os

from hhca import thread_env

os.environ.update(thread_env(1, numba=False))

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

import hhca
from hhca.config import load_config


def write_counts(cells, perineural, rule, path):
    """Fig 5 k: number of perineural and other pseudo-cells, in total and per slide."""
    n_perineural, n_other = int(perineural.sum()), int((~perineural).sum())
    print(f"perineural ({rule}): {n_perineural} / other {n_other} of {cells.n_obs}")
    per_slide = (
        cells.obs.assign(peri=perineural)
        .groupby("slide", observed=True)
        .agg(n=("peri", "size"), n_perineural=("peri", "sum"))
        .reset_index()
    )
    counts = {
        "rule": rule,
        "n_total": int(cells.n_obs),
        "n_perineural": n_perineural,
        "n_other": n_other,
        "pct_perineural": 100 * n_perineural / cells.n_obs,
        "pct_other": 100 * n_other / cells.n_obs,
        "per_slide": per_slide.astype({"n": int, "n_perineural": int}).to_dict("records"),
    }
    with open(path, "w") as handle:
        json.dump(counts, handle, indent=2)


def wilcoxon_de(cells):
    """Wilcoxon test of every gene, perineural vs rest, on log1p CP10k (obs column niche)."""
    normalized = cells.copy()
    sc.pp.normalize_total(normalized, target_sum=1e4)
    sc.pp.log1p(normalized)
    sc.tl.rank_genes_groups(
        normalized,
        "niche",
        groups=["perineural"],
        reference="rest",
        method="wilcoxon",
        pts=True,
        corr_method="benjamini-hochberg",
    )
    de = sc.get.rank_genes_groups_df(normalized, group="perineural", pval_cutoff=None)
    pts = normalized.uns["rank_genes_groups"]["pts"]
    pts_rest = normalized.uns["rank_genes_groups"].get("pts_rest")
    de = de.rename(
        columns={"names": "gene", "logfoldchanges": "log2fc", "pvals": "pval", "pvals_adj": "padj", "scores": "score"}
    )
    de["pct_perineural"] = pts.loc[de.gene, "perineural"].values
    de["pct_rest"] = (
        pts_rest.loc[de.gene, "perineural"].values
        if pts_rest is not None and "perineural" in pts_rest
        else pts.loc[de.gene, "rest"].values
        if "rest" in pts
        else np.nan
    )
    return de[["gene", "log2fc", "pval", "padj", "score", "pct_perineural", "pct_rest"]]


def regression(cells, distance, perineural, markers):
    """Slide-adjusted regression of each marker on nerve distance and on the perineural indicator.

    Returns the per-marker table and the per-slide group means.
    """
    design = pd.concat([cells.obs[["slide"]], hhca.tl.marker_expression(cells, markers)], axis=1)
    design["logcounts"] = np.log(cells.obs["total_counts"].to_numpy(np.float64))
    design["z_lc"] = (design.logcounts - design.logcounts.mean()) / design.logcounts.std()
    design["z_dn"] = (distance - np.nanmean(distance)) / np.nanstd(distance, ddof=1)
    design["peri"] = perineural.astype(int)
    by_distance = hhca.tl.marker_effects(design, markers, "z_dn")
    by_niche = hhca.tl.marker_effects(design, markers, "peri")
    table = pd.DataFrame(
        {
            "marker": markers,
            "beta_per_sd_distance": by_distance.beta,
            "ci_lo": by_distance.ci_lo,
            "ci_hi": by_distance.ci_hi,
            "p_distance": by_distance.p,
            "beta_perineural": by_niche.beta,
            "p_perineural": by_niche.p,
            "n": by_distance.n,
            "sd_distance_um": float(np.nanstd(distance, ddof=1)),
            "padj_distance": by_distance.padj,
            "padj_perineural": by_niche.padj,
        }
    )
    return table, hhca.tl.marker_means(design, markers)


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", required=True, help="pooled macrophage pseudo-cell h5ad")
    parser.add_argument("--distances", required=True, help="per-cell nerve distances of nerve_clusters.py")
    parser.add_argument("--outdir", required=True, help="output folder")
    return parser.parse_args()


def main():
    """Split the pseudo-cells into perineural and rest and write the tables of Fig 5 k and l."""
    args = parse_args()
    macrophage = load_config("nerve_macrophage")["macrophage"]
    dist_col, radius = macrophage["dist_col"], macrophage["perineural_um"]
    os.makedirs(args.outdir, exist_ok=True)

    cells = ad.read_h5ad(args.cache)
    distances = pd.read_csv(args.distances)
    assert len(distances) == cells.n_obs and (distances.slide.values == cells.obs.slide.values).all()
    assert np.allclose(distances.centroid_x_um.values, cells.obs.centroid_x_um.values)
    distance = distances[dist_col].to_numpy(float)
    perineural = distance < radius
    cells.obs["niche"] = pd.Categorical(np.where(perineural, "perineural", "rest"), categories=["perineural", "rest"])
    write_counts(cells, perineural, f"{dist_col} < {radius:g} um", f"{args.outdir}/k_counts.json")

    de = wilcoxon_de(cells)
    de.to_csv(f"{args.outdir}/l_de_all.csv.gz", index=False)
    markers = de.set_index("gene").loc[[g for g in macrophage["markers"] if g in set(de.gene)]].reset_index()
    markers.to_csv(f"{args.outdir}/l_de_markers.csv", index=False)
    print(markers.to_string(index=False))

    table, per_slide = regression(cells, distance, perineural, macrophage["regression_markers"])
    table.to_csv(f"{args.outdir}/l_regression.csv", index=False)
    per_slide.to_csv(f"{args.outdir}/l_per_slide.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
