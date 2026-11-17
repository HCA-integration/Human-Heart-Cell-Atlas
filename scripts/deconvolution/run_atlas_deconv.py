#!/usr/bin/env python
"""NNLS deconvolution of the Noseda standard-Visium slides against the atlas Level-3 profile.

Per spot, linear CP10k expression is fitted as a non-negative mixture of the Level-3 signatures
(hhca.tl.nnls_deconvolution). Signatures = the atlas Level-3 profile of the substrate pipeline (23 cell types, the
same types as the Visium HD and Xenium annotations); genes = the 2,000 with the largest coefficient of variation
across types plus cardiac markers, intersected with the genes of the slide. The reported proportion is the mRNA
fraction (weight x signature total, normalised per spot). Slides, filters, number of genes and markers: config
slides, qc, methods.NNLS and markers.

Inputs:  <data>/<slide>/filtered_feature_bc_matrix.h5 (and spatial/ where present), the Level-3 profile
Outputs: <out>/tables/<slide>_L3_deconv.csv.gz    barcode, x, y, 23 proportions, dominant_L3, deconv_maxprop,
                                                  total_counts
         <out>/h5ad/<slide>_L3_deconv.h5ad        the same in obs, counts in X, obsm['spatial'] where present
         <out>/figures/<slide>_L3_spatial.png     dominant type and compartment proportions
         <out>/tables/cohort_L3_mean_proportion.csv, cohort_L3_argmax_composition.csv, cohort_summary.{csv,json}

Usage: run_atlas_deconv.py --data NOSEDA_DIR --pseudobulk atlas_L3_pseudobulk.tsv --out DIR [--config CONFIG]
           [slide ...]
"""

import argparse
import json
import os
import traceback

from hhca import thread_env

os.environ.update(thread_env(1, numba=False))

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import deconv_config
import hhca
from hhca.log import get_logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# compartments shown in the per-slide figure
FIGURE_GROUPS = ["CM", "Fibroblast", "Endothelial", "Pericyte", "Myeloid"]


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="NNLS deconvolution of the Noseda standard-Visium slides")
    parser.add_argument(
        "--data", required=True, help="directory with <slide>/filtered_feature_bc_matrix.h5 and spatial/"
    )
    parser.add_argument("--pseudobulk", required=True, help="atlas Level-3 profile (genes x cell types, tsv)")
    parser.add_argument("--out", required=True, help="output root; writes tables/, h5ad/ and figures/")
    deconv_config.add_argument(parser)
    parser.add_argument("slides", nargs="*", help="slide names (default: all 16)")
    return parser.parse_args()


def write_spot_table(path, prop_df, dominant, max_prop, adata, has_spatial):
    """Per-spot table: coordinates where present, proportions, dominant type, its proportion and total counts."""
    table = prop_df.copy()
    if has_spatial:
        table.insert(0, "y", adata.obsm["spatial"][:, 1])
        table.insert(0, "x", adata.obsm["spatial"][:, 0])
    table["dominant_L3"] = dominant
    table["deconv_maxprop"] = max_prop.astype(np.float32)
    table["total_counts"] = adata.obs["total_counts"].values.astype(np.float32)
    table.index.name = "barcode"
    table.to_csv(path, compression="gzip")


def write_h5ad(path, sample, prop, prop_df, dominant, max_prop, adata, has_spatial, cell_types, n_genes, rmse):
    """The slide with the proportions in obs and obsm['L3_proportions'], counts in X."""
    obs = prop_df.copy()
    obs["dominant_L3"] = pd.Categorical(dominant)
    obs["deconv_maxprop"] = max_prop.astype(np.float32)
    obs["total_counts"] = adata.obs["total_counts"].values.astype(np.float32)
    result = ad.AnnData(X=csr_matrix(adata.X), obs=obs, var=pd.DataFrame(index=adata.var_names))
    result.obsm["L3_proportions"] = prop.astype(np.float32)
    result.uns["L3_cell_types"] = list(cell_types)
    result.uns["deconv"] = {
        "method": "atlas_L3 NNLS (mRNA-fraction)",
        "n_genes_used": n_genes,
        "nnls_rmse": rmse,
        "has_spatial": bool(has_spatial),
        "sample": sample,
    }
    if has_spatial:
        result.obsm["spatial"] = adata.obsm["spatial"]
        result.obs["array_row"] = adata.obs["array_row"].values
        result.obs["array_col"] = adata.obs["array_col"].values
    result.write(path)


def deconv_slide(sample, signatures, cfg, data_dir, out_dir, log):
    """Deconvolve one slide and write its table, h5ad and figure.

    Returns the mean proportions, the argmax composition and a summary dict.
    """
    adata = hhca.tl.visium_spots(
        f"{data_dir}/{sample}", cfg["qc"]["min_cells_per_gene"], cfg["qc"]["min_counts_per_spot"]
    )
    has_spatial = "spatial" in adata.obsm
    n_spots = adata.n_obs
    log(f"  {sample}: {n_spots} spots x {adata.n_vars} genes | spatial={has_spatial}")
    hhca.tl.nnls_deconvolution(adata, signatures, cfg["methods"]["NNLS"]["n_top_genes"], cfg["markers"]["cardiac"])
    prop_df = adata.obsm["NNLS"]
    prop, cell_types = prop_df.to_numpy(), list(prop_df.columns)
    genes, rmse = adata.uns["NNLS"]["genes"], adata.uns["NNLS"]["rmse"]
    dominant = np.array(cell_types)[prop.argmax(1)]
    max_prop = prop.max(1)

    write_spot_table(f"{out_dir}/tables/{sample}_L3_deconv.csv.gz", prop_df, dominant, max_prop, adata, has_spatial)
    write_h5ad(
        f"{out_dir}/h5ad/{sample}_L3_deconv.h5ad",
        sample,
        prop,
        prop_df,
        dominant,
        max_prop,
        adata,
        has_spatial,
        cell_types,
        len(genes),
        rmse,
    )
    groups = {group: cfg["comparison"]["compartments"][group] for group in FIGURE_GROUPS}
    make_figure(f"{out_dir}/figures/{sample}_L3_spatial.png", sample, prop_df, dominant, adata, has_spatial, groups)

    mean_prop = prop_df.mean(0)
    composition = (pd.Series(dominant).value_counts() / n_spots).reindex(cell_types, fill_value=0.0)
    summary = {
        "sample": sample,
        "n_spots": int(n_spots),
        "n_genes_used": len(genes),
        "nnls_rmse": rmse,
        "median_counts": float(np.median(adata.obs["total_counts"].values)),
        "has_spatial": bool(has_spatial),
        "top_mean_type": mean_prop.idxmax(),
        "top_mean_prop": float(mean_prop.max()),
    }
    log(
        f"    -> rmse={rmse:.3f} top_mean={summary['top_mean_type']} ({summary['top_mean_prop']:.3f}) "
        f"table, h5ad and figure written"
    )
    return mean_prop, composition, summary


def make_figure(path, sample, prop_df, dominant, adata, has_spatial, groups):
    """Dominant type and compartment proportions in space, or composition bar plots without spot positions."""
    group_props = {}
    for group, members in groups.items():
        present = [member for member in members if member in prop_df.columns]
        group_props[group] = prop_df[present].sum(1).values if present else np.zeros(len(prop_df))
    if has_spatial:
        xy = adata.obsm["spatial"]
        x, y = xy[:, 0], xy[:, 1]
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        categories = pd.Categorical(dominant)
        codes = categories.codes
        labels = list(categories.categories)
        cmap = plt.get_cmap("tab20", max(len(labels), 1))
        ax = axes[0, 0]
        ax.scatter(x, y, c=codes, cmap=cmap, s=6, vmin=-0.5, vmax=len(labels) - 0.5)
        ax.set_title(f"{sample}\ndominant L3 (argmax)")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
        handles = [
            plt.Line2D([0], [0], marker="o", ls="", mfc=cmap(i), mec="none", label=labels[i])
            for i in range(len(labels))
        ]
        ax.legend(handles=handles, fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, ncol=1)
        for i, group in enumerate(groups):
            ax = axes.flat[i + 1]
            values = group_props[group]
            points = ax.scatter(x, y, c=values, cmap="viridis", s=6, vmin=0, vmax=max(values.max(), 1e-6))
            ax.set_title(f"{group} proportion")
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.axis("off")
            fig.colorbar(points, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
    else:
        # no spot positions: bar plots of the composition
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        mean_prop = prop_df.mean(0).sort_values(ascending=False)
        mean_prop.plot.bar(ax=axes[0], color="steelblue")
        axes[0].set_title(f"{sample} (no spatial)\nmean per-spot L3 proportion")
        axes[0].set_ylabel("mean proportion")
        axes[0].tick_params(axis="x", labelsize=7, rotation=90)
        composition = (pd.Series(dominant).value_counts() / len(dominant)).reindex(mean_prop.index, fill_value=0.0)
        composition.plot.bar(ax=axes[1], color="indianred")
        axes[1].set_title("argmax dominant-type fraction")
        axes[1].set_ylabel("fraction of spots")
        axes[1].tick_params(axis="x", labelsize=7, rotation=90)
        plt.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    """Deconvolve every slide and write the cohort tables."""
    args = parse_args()
    log = get_logger()
    out_dir = args.out
    for folder in ("tables", "h5ad", "figures"):
        os.makedirs(f"{out_dir}/{folder}", exist_ok=True)

    log("atlas Level-3 signatures:", args.pseudobulk)
    signatures = pd.read_csv(args.pseudobulk, sep="\t", index_col=0)
    signatures.index = signatures.index.astype(str)
    signatures = signatures[~signatures.index.duplicated()]
    cell_types = list(signatures.columns)
    log("L3 types:", len(cell_types))

    cfg = deconv_config.load(args.config)
    only = args.slides or None
    all_slides = cfg["slides"]["benchmark"] + cfg["slides"]["nnls_only"]
    slides = [slide for slide in all_slides if (only is None or slide in only)]
    # slides whose count matrix is present
    slides = [slide for slide in slides if os.path.exists(f"{args.data}/{slide}/filtered_feature_bc_matrix.h5")]
    log("slides to deconvolve:", len(slides), slides)

    mean_rows, composition_rows, summaries = {}, {}, []
    for slide in slides:
        # a slide that fails is reported and left out of the cohort tables; the others are still written
        try:
            mean_prop, composition, summary = deconv_slide(slide, signatures, cfg, args.data, out_dir, log)
            mean_rows[slide] = mean_prop
            composition_rows[slide] = composition
            summaries.append(summary)
        except (OSError, KeyError, ValueError, RuntimeError) as error:
            log(f"  {slide} failed: {type(error).__name__}: {error}")
            traceback.print_exc()

    if mean_rows:
        mean_df = pd.DataFrame(mean_rows).reindex(cell_types)
        composition_df = pd.DataFrame(composition_rows).reindex(cell_types)
        mean_df.to_csv(f"{out_dir}/tables/cohort_L3_mean_proportion.csv")
        composition_df.to_csv(f"{out_dir}/tables/cohort_L3_argmax_composition.csv")
        pd.DataFrame(summaries).to_csv(f"{out_dir}/tables/cohort_summary.csv", index=False)
        with open(f"{out_dir}/tables/cohort_summary.json", "w") as handle:
            json.dump(summaries, handle, indent=2)
        log("cohort mean per-spot proportion:\n" + mean_df.round(3).to_string())
        log("wrote the cohort tables to", f"{out_dir}/tables/")
    log("done:", len(summaries), "slides")


if __name__ == "__main__":
    main()
