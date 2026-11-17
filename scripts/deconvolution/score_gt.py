#!/usr/bin/env python
"""Score every method's per-spot proportions on the simulated slides against the known composition.

  * leaderboard per method: overall RMSE, MAE, Jensen-Shannon divergence, mean Pearson / Spearman, ranks
  * RMSE and Pearson r per method and cell type; best method per cell type
  * accuracy by region (epi / myo / endo)
  * known vs estimated proportion for six cell types and the four best methods

The scores are computed by hhca.tl.score_methods, hhca.tl.leaderboard, hhca.tl.celltype_scores and
hhca.tl.region_scores; slides, methods and the cell types of the scatter figure: config slides, comparison, gtsim.

Inputs:  <outputs>/<ref>/<method>/<slide>.csv.gz, <truth>/<slide>/truth_proportions.csv.gz (the same spots)
Outputs: tables in --tables, figures in --figures

Usage: score_gt.py --outputs DIR --truth DIR --tables DIR --figures DIR --factor-order JSON [--refs hhca]
           [--config CONFIG]
"""

import argparse
import json
import os

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import deconv_config
import hhca

matplotlib.use("Agg")
import matplotlib.pyplot as plt

POINT_COLOUR = "#888888"


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--outputs", required=True, help="directory with <ref>/<method>/<slide>.csv.gz")
    parser.add_argument("--truth", required=True, help="directory with <slide>/truth_proportions.csv.gz")
    parser.add_argument("--tables", required=True, help="output directory of the tables")
    parser.add_argument("--figures", required=True, help="output directory of the figures")
    parser.add_argument("--factor-order", required=True, help="json list of the Level-3 cell types")
    parser.add_argument("--refs", default="hhca", help="comma-separated reference names (sub-folders of --outputs)")
    return deconv_config.add_argument(parser).parse_args()


def load_sections(ref, slides, methods, cell_types, outputs_dir, truth_dir):
    """Simulated slides with the known composition in obsm["truth"] and each method's proportions in obsm."""
    sections = {}
    for slide in slides:
        truth_path = f"{truth_dir}/{slide}/truth_proportions.csv.gz"
        if not os.path.exists(truth_path):
            continue
        tables = {"truth": hhca.tl.read_proportions(truth_path, cell_types, normalise=False)}
        for method in methods:
            path = f"{outputs_dir}/{ref}/{method}/{slide}.csv.gz"
            if os.path.exists(path):
                tables[method] = hhca.tl.read_proportions(path, cell_types, normalise=False)
        adata = hhca.tl.collect_proportions(tables)
        sections[slide] = adata
    return sections


def write_leaderboards(overall_df, refs, table_dir, figure_dir):
    """Leaderboard per reference: metrics averaged over slides, ranked by the mean of three ranks."""
    for ref in refs:
        sub = overall_df[overall_df.ref == ref]
        if sub.empty:
            continue
        leaderboard = hhca.tl.leaderboard(sub)
        leaderboard.to_csv(f"{table_dir}/leaderboard_{ref}.csv", index=False)
        print(f"\nleaderboard ({ref}):\n", leaderboard.round(4).to_string(index=False))
        if ref == "hhca":
            leaderboard_figure(leaderboard, figure_dir)


def write_celltype_tables(long_df, refs, cell_types, table_dir, figure_dir):
    """RMSE and Pearson r per method and cell type (mean over slides); best method per cell type for hhca."""
    for ref in refs:
        sub = long_df[long_df.ref == ref]
        if sub.empty:
            continue
        scores = hhca.tl.celltype_scores(sub, cell_types)
        scores["rmse"].to_csv(f"{table_dir}/celltype_rmse_{ref}.csv")
        scores["pearson"].to_csv(f"{table_dir}/celltype_pearson_{ref}.csv")
        if ref == "hhca":
            scores["best"].to_csv(f"{table_dir}/best_method_per_celltype_{ref}.csv")
            print(f"\nbest method per cell type ({ref}):\n", scores["best"].round(4).to_string())
            celltype_heatmaps(scores["rmse"], scores["pearson"], ref, figure_dir)


def celltype_heatmaps(rmse_pivot, corr_pivot, ref, figure_dir):
    """Heatmaps of the per-cell-type RMSE and Pearson r (cell types x methods)."""
    for pivot, name, cmap, limits, title in [
        (rmse_pivot, "celltype_rmse_heatmap", "magma_r", (0, None), "Per-spot RMSE (lower=better)"),
        (corr_pivot, "celltype_pearson_heatmap", "RdBu_r", (-1, 1), "Per-spot Pearson r (higher=better)"),
    ]:
        methods = list(pivot.columns)
        fig, ax = plt.subplots(figsize=(1.1 * len(methods) + 3, 9))
        vmin, vmax = limits if limits[1] is not None else (0, float(np.nanpercentile(pivot.values, 98)))
        im = ax.imshow(pivot.values.astype(float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        for (row, col), value in np.ndenumerate(pivot.values.astype(float)):
            if np.isfinite(value):
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if (cmap == "magma_r" and value > (vmax * 0.5)) else "black",
                )
        ax.set_title(f"{title}\nmethod x L3 cell type ({ref} reference)")
        fig.colorbar(im, ax=ax, fraction=0.025)
        plt.tight_layout()
        plt.savefig(f"{figure_dir}/{name}_{ref}.png", dpi=300)
        plt.close()


def leaderboard_figure(leaderboard, figure_dir):
    """Bar charts of the overall RMSE, the mean JSD and the mean Pearson r per method."""
    leaderboard = leaderboard.sort_values("rmse")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, metric, title, ascending in [
        (axes[0], "rmse", "Overall RMSE (lower=better)", True),
        (axes[1], "jsd", "Mean per-spot JSD (lower=better)", True),
        (axes[2], "pearson", "Mean per-celltype Pearson r (higher=better)", False),
    ]:
        ranked = leaderboard.sort_values(metric, ascending=ascending)
        ax.barh(range(len(ranked)), ranked[metric].values, color="#4C72B0")
        ax.set_yticks(range(len(ranked)))
        ax.set_yticklabels(ranked["method"], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        for i, value in enumerate(ranked[metric].values):
            ax.text(value, i, f" {value:.3f}", va="center", fontsize=7)
    plt.suptitle("Deconvolution accuracy LEADERBOARD vs HD-derived truth (HHCA reference)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{figure_dir}/leaderboard.png", dpi=300)
    plt.close()


def region_figure(overall_df, figure_dir):
    """Overall RMSE per method on the epi, myo and endo sections (hhca reference)."""
    sub = overall_df[overall_df.ref == "hhca"]
    if sub.empty:
        return
    methods = sorted(sub.method.unique())
    regions = ["epi", "myo", "endo"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    width = 0.25
    colours = {"epi": "#66c2a5", "myo": "#fc8d62", "endo": "#8da0cb"}
    for i, region in enumerate(regions):
        values = [sub[(sub.region == region) & (sub.method == method)]["rmse"].mean() for method in methods]
        ax.bar(x + (i - 1) * width, values, width, label=region, color=colours[region])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("overall RMSE")
    ax.set_title("Region stratification (within-ventricle, H0054LV): accuracy by region")
    ax.legend(title="region", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{figure_dir}/region_effect.png", dpi=300)
    plt.close()


def scatter_figure(sections, scatter_types, table_dir, figure_dir):
    """Known vs estimated proportion of six cell types for the four best methods, all slides pooled."""
    # the four best methods of the leaderboard
    if os.path.exists(f"{table_dir}/leaderboard_hhca.csv"):
        leaderboard = pd.read_csv(f"{table_dir}/leaderboard_hhca.csv")
        methods = list(leaderboard.sort_values("score")["method"])[:4]
    else:
        methods = sorted({key for adata in sections.values() for key in adata.obsm if key != "truth"})[:4]
    fig, axes = plt.subplots(len(methods), len(scatter_types), figsize=(3 * len(scatter_types), 3 * len(methods)))
    if len(methods) == 1:
        axes = axes[None, :]
    for row, method in enumerate(methods):
        scored = [adata for adata in sections.values() if method in adata.obsm]
        if not scored:
            continue
        truth_all = pd.concat([adata.obsm["truth"] for adata in scored])
        pred_all = pd.concat([adata.obsm[method] for adata in scored])
        for col, cell_type in enumerate(scatter_types):
            ax = axes[row, col]
            x, y = truth_all[cell_type].values, pred_all[cell_type].values
            ax.scatter(x, y, s=3, alpha=0.25, color=POINT_COLOUR, rasterized=True)
            lim = max(x.max(), y.max(), 0.05)
            ax.plot([0, lim], [0, lim], "k--", lw=0.7)
            pearson = pearsonr(x, y)[0] if x.std() > 1e-9 and y.std() > 1e-9 else np.nan
            rmse = np.sqrt(np.mean((x - y) ** 2))
            ax.set_title(f"{cell_type}\nr={pearson:.2f} rmse={rmse:.3f}", fontsize=7)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            if col == 0:
                ax.set_ylabel(f"{method}\npredicted", fontsize=8)
            if row == len(methods) - 1:
                ax.set_xlabel("truth (HD FICTURE)", fontsize=8)
            ax.tick_params(labelsize=6)
    plt.suptitle("Truth vs predicted proportion (HHCA ref, all sim slides pooled)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{figure_dir}/truth_vs_pred_scatter.png", dpi=200)
    plt.close()


def main():
    """Score all outputs and write the leaderboard, per-cell-type and region tables with their figures."""
    args = parse_args()
    cfg = deconv_config.load(args.config)
    slides, regions = cfg["slides"]["gtsim_hd"], cfg["slides"]["gtsim_region"]
    methods = cfg["comparison"]["methods"]
    table_dir = args.tables
    figure_dir = args.figures
    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    with open(args.factor_order) as handle:
        cell_types = list(json.load(handle))
    refs = [ref for ref in args.refs.split(",") if ref]

    sections, overall, long = {}, [], []
    for ref in refs:
        sections[ref] = load_sections(ref, slides, methods, cell_types, args.outputs, args.truth)
        overall_ref, long_ref = hhca.tl.score_methods(sections[ref], methods, regions)
        if len(overall_ref):
            overall.append(overall_ref.assign(ref=ref))
            long.append(long_ref.assign(ref=ref))
    if not overall:
        print("no method outputs to score")
        return
    overall_df, long_df = pd.concat(overall, ignore_index=True), pd.concat(long, ignore_index=True)
    overall_df = overall_df[["ref"] + [column for column in overall_df.columns if column != "ref"]]
    long_df = long_df[["ref"] + [column for column in long_df.columns if column != "ref"]]
    long_df.to_csv(f"{table_dir}/per_celltype_scores_long.csv", index=False)
    overall_df.to_csv(f"{table_dir}/overall_scores_long.csv", index=False)
    available = overall_df.groupby(["ref", "method"]).size().rename("n_slides").reset_index()
    print("available (ref, method, n_slides):\n", available.to_string(index=False))

    write_leaderboards(overall_df, refs, table_dir, figure_dir)
    write_celltype_tables(long_df, refs, cell_types, table_dir, figure_dir)

    # accuracy by region
    region_effect = hhca.tl.region_scores(overall_df[overall_df.ref == "hhca"])
    region_effect.to_csv(f"{table_dir}/region_effect_hhca.csv", index=False)
    region_figure(overall_df, figure_dir)
    scatter_figure(sections.get("hhca", {}), cfg["gtsim"]["scatter_types"], table_dir, figure_dir)

    summary = {
        "n_scored": len(overall_df),
        "available": available.to_dict(orient="records"),
        "leaderboard_hhca": pd.read_csv(f"{table_dir}/leaderboard_hhca.csv").to_dict(orient="records")
        if os.path.exists(f"{table_dir}/leaderboard_hhca.csv")
        else None,
    }
    with open(f"{table_dir}/score_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print("\nwrote tables to", table_dir, "and figures to", figure_dir)


if __name__ == "__main__":
    main()
