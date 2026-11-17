#!/usr/bin/env python
"""Compare the deconvolution methods on the per-spot proportion tables of the benchmark slides.

  (a) cross-method concordance: method x method Pearson r and Jensen-Shannon divergence, and per cell type
  (b) uniform background ("floor") of immune and lymphatic-EC proportions per method, atrial vs ventricular
  (c) compartment composition per method (composition_mean_by_method.csv is the table behind ED6 h)
  (d) runtime per method
  (e) marker support: Spearman correlation of proportion and marker score per method and cell type

The tables are computed by hhca.tl (method_concordance, background_floor, compartment_composition, marker_support)
from the proportions of every slide, held as AnnData with one obsm table per method; slides, methods,
compartments and thresholds: config slides and comparison.

Inputs:  <outputs>/<method>/<slide>.csv.gz and .json, <nnls-dir>/<slide>_L3_deconv.csv.gz,
         marker scores <marker-scores>/<slide>.csv.gz (when given)
Outputs: the tables and benchmark_summary.json in --tables, overview figures in --figures

Usage: compare_methods.py --outputs DIR --nnls-dir DIR --tables DIR --figures DIR [--marker-scores DIR]
           [--config CONFIG]
"""

import argparse
import json
import os

import matplotlib
import numpy as np
import pandas as pd

import deconv_config
import hhca

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CTS = [
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
# compartments of the stacked composition bars, bottom to top
COMPART_STACK_ORDER = [
    "CM",
    "Fibroblast",
    "Endothelial",
    "Pericyte",
    "SMC",
    "Myeloid",
    "Lymphoid",
    "Mast",
    "Adipocyte",
    "Neural",
    "Epicardium",
]


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="Cross-method comparison tables")
    parser.add_argument("--outputs", required=True, help="directory with <method>/<slide>.csv.gz")
    parser.add_argument("--nnls-dir", required=True, help="directory with the NNLS tables <slide>_L3_deconv.csv.gz")
    parser.add_argument("--tables", required=True, help="output directory of the tables")
    parser.add_argument("--figures", required=True, help="output directory of the figures")
    parser.add_argument("--marker-scores", default="", help="directory with <slide>.csv.gz of marker_scores.py")
    return deconv_config.add_argument(parser).parse_args()


def load_sections(slides, methods, outputs_dir, nnls_dir, marker_dir):
    """One AnnData per slide with the proportions of every method (and the marker scores) as obsm tables."""
    sections = {}
    for slide in slides:
        tables = {}
        for method in methods:
            path = (
                f"{nnls_dir}/{slide}_L3_deconv.csv.gz" if method == "NNLS" else f"{outputs_dir}/{method}/{slide}.csv.gz"
            )
            if os.path.exists(path):
                tables[method] = hhca.tl.read_proportions(path, CTS)
        adata = hhca.tl.collect_proportions(tables)
        score_path = f"{marker_dir}/{slide}.csv.gz"
        if marker_dir and tables and os.path.exists(score_path):
            adata.obsm["marker_scores"] = pd.read_csv(score_path, index_col=0).reindex(adata.obs_names)
        sections[slide] = adata
    return sections


def runtimes(outputs_dir, methods, slides):
    """Runtime, genes used and spots per method and slide, from the run summaries of the methods."""
    rows = []
    for method in methods:
        for slide in slides:
            path = f"{outputs_dir}/{method}/{slide}.json"
            if os.path.exists(path):
                with open(path) as handle:
                    run = json.load(handle)
                rows.append(
                    {
                        "method": method,
                        "slide": slide,
                        "runtime_sec": run.get("runtime_sec", np.nan),
                        "n_genes_used": run.get("n_genes_used", np.nan),
                        "n_spots": run.get("n_spots", np.nan),
                    }
                )
    return pd.DataFrame(rows)


def concordance(sections, methods, table_dir, figure_dir):
    """(a) Method x method Pearson r and Jensen-Shannon divergence, and the mean cross-method r per cell type."""
    result = hhca.tl.method_concordance(sections, methods)
    pearson_mat, jsd_mat = result["pearson"], result["jsd"]
    pearson_mat.to_csv(f"{table_dir}/concordance_pearson_matrix.csv")
    jsd_mat.to_csv(f"{table_dir}/concordance_jsd_matrix.csv")
    result["per_type"].sort_values().to_csv(
        f"{table_dir}/concordance_per_celltype.csv", header=["mean_cross_method_pearson"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, mat, title in [
        (axes[0], pearson_mat, "Cross-method Pearson r\n(per-spot, avg over cell types & slides)"),
        (axes[1], 1 - jsd_mat, "Cross-method similarity (1 - JSD)"),
    ]:
        im = ax.imshow(mat.values.astype(float), vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods, fontsize=8)
        for (row, col), value in np.ndenumerate(mat.values.astype(float)):
            if np.isfinite(value):
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value < 0.6 else "black",
                    fontsize=7,
                )
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(f"{figure_dir}/concordance_heatmaps.png", dpi=150)
    plt.close()
    return pearson_mat


def uniform_floor(sections, methods, chambers, settings, table_dir, figure_dir):
    """(b) Mean proportion of the floor types on atrial and ventricular slides per method; returns the summary."""
    floor_types = settings["cell_types"]
    floor_df, floor_summary = hhca.tl.background_floor(
        sections, methods, chambers, floor_types, settings["min_ventricular"], settings["min_ratio"], settings["max_cv"]
    )
    floor_df.to_csv(f"{table_dir}/floor_by_method.csv", index=False)
    floor_summary.to_csv(f"{table_dir}/floor_summary_by_method.csv")

    fig, axes = plt.subplots(1, len(floor_types), figsize=(4 * len(floor_types), 4.5), sharey=False)
    for ax, cell_type in zip(axes, floor_types):
        sub = floor_df[floor_df.cell_type == cell_type].set_index("method").reindex(methods)
        x = np.arange(len(methods))
        width = 0.4
        ax.bar(x - width / 2, sub["atrial_mean"], width, label="atrial", color="#4C72B0")
        ax.bar(x + width / 2, sub["ventricular_mean"], width, label="ventricular", color="#C44E52")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax.set_title(cell_type, fontsize=10)
        ax.set_ylabel("mean per-spot proportion")
    axes[0].legend(fontsize=8)
    plt.suptitle("The immune / Lymphatic-EC FLOOR per method (atrial vs ventricular)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{figure_dir}/floor_by_method.png", dpi=150)
    plt.close()
    return floor_summary


def compartment_composition(sections, methods, compartments, chambers, table_dir, figure_dir):
    """(c) Mean compartment proportion per method and slide; returns the mean per method."""
    comp_df = hhca.tl.compartment_composition(sections, methods, compartments, chambers)
    comp_df.to_csv(f"{table_dir}/composition_by_method_slide.csv", index=False)
    comp_mean = comp_df.groupby("method")[list(compartments)].mean()
    comp_mean.to_csv(f"{table_dir}/composition_mean_by_method.csv")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottom = np.zeros(len(methods))
    palette = plt.get_cmap("tab20")
    comp_by_method = comp_mean.reindex(methods)
    for i, compartment in enumerate(COMPART_STACK_ORDER):
        values = comp_by_method[compartment].values
        ax.bar(range(len(methods)), values, bottom=bottom, label=compartment, color=palette(i))
        bottom += np.nan_to_num(values)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("cohort-mean proportion")
    ax.set_title("Compartment composition per method (cohort mean)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{figure_dir}/composition_by_method.png", dpi=150)
    plt.close()
    return comp_mean


def runtime_summary(outputs_dir, methods, slides, table_dir, figure_dir):
    """(d) Runtime per method and slide (empty frame when no run summaries exist)."""
    runtime = runtimes(outputs_dir, methods, slides)
    if not runtime.empty:
        runtime.to_csv(f"{table_dir}/runtime_by_method_slide.csv", index=False)
        by_method = runtime.groupby("method")["runtime_sec"].agg(["mean", "median", "min", "max", "count"])
        by_method.to_csv(f"{table_dir}/runtime_summary.csv")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        order = by_method["median"].sort_values().index
        ax.bar(range(len(order)), by_method.loc[order, "median"] / 60.0, color="#55A868")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha="right")
        ax.set_ylabel("median runtime / slide (min)")
        ax.set_title("Per-slide runtime per method")
        plt.tight_layout()
        plt.savefig(f"{figure_dir}/runtime_by_method.png", dpi=150)
        plt.close()
    return runtime


def marker_support(sections, methods, cell_types, table_dir, figure_dir):
    """(e) Median Spearman correlation of proportion and marker score per method and cell type.

    Returns an empty frame when no marker scores were given.
    """
    if not any("marker_scores" in adata.obsm for adata in sections.values()):
        return pd.DataFrame()
    support = hhca.tl.marker_support(sections, methods, cell_types)
    support.to_csv(f"{table_dir}/marker_support_by_method.csv", index=False)
    pivot = support.pivot(index="cell_type", columns="method", values="median_spearman").reindex(cell_types)
    fig, ax = plt.subplots(figsize=(1.2 * len(methods) + 3, 9))
    im = ax.imshow(pivot.values.astype(float), vmin=-0.3, vmax=0.8, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for (row, col), value in np.ndenumerate(pivot.values.astype(float)):
        if np.isfinite(value):
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title("Marker support: Spearman(proportion, marker score)\nhigh=trustworthy, ~0=artifact/floor")
    fig.colorbar(im, ax=ax, fraction=0.025)
    plt.tight_layout()
    plt.savefig(f"{figure_dir}/marker_support.png", dpi=150)
    plt.close()
    return support


def main():
    """Run the five comparisons and write benchmark_summary.json."""
    args = parse_args()
    cfg = deconv_config.load(args.config)
    settings, slides = cfg["comparison"], cfg["slides"]["benchmark"]
    cardiomyocytes = dict(zip(("atrial_type", "ventricular_type"), cfg["downstream"]["fig3c_cm"]))
    table_dir = args.tables
    figure_dir = args.figures
    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    sections = load_sections(slides, settings["methods"], args.outputs, args.nnls_dir, args.marker_scores)
    available = {
        method: sorted(slide for slide, adata in sections.items() if method in adata.obsm)
        for method in settings["methods"]
    }
    methods = [method for method in settings["methods"] if available[method]]
    print("methods with outputs:", {method: len(available[method]) for method in methods})
    chambers = hhca.tl.section_chambers(sections, **cardiomyocytes, atrial_sections=settings["atrial_slides"])
    with open(f"{table_dir}/_availability.json", "w") as handle:
        json.dump(
            {"available": {method: available[method] for method in methods}, "chamber": chambers}, handle, indent=2
        )

    pearson_mat = concordance(sections, methods, table_dir, figure_dir)
    floor_summary = uniform_floor(sections, methods, chambers, settings["floor"], table_dir, figure_dir)
    comp_mean = compartment_composition(sections, methods, settings["compartments"], chambers, table_dir, figure_dir)
    runtime = runtime_summary(args.outputs, settings["methods"], slides, table_dir, figure_dir)
    support = marker_support(sections, methods, CTS, table_dir, figure_dir)

    summary = {
        "methods_with_outputs": {method: len(available[method]) for method in methods},
        "chamber_map": chambers,
        "floor_n_types_by_method": floor_summary.to_dict(),
        "concordance_mean_offdiag": {
            method: float(np.nanmean([pearson_mat.loc[method, other] for other in methods if other != method]))
            for method in methods
        },
        "composition_mean_by_method": comp_mean.round(4).to_dict(orient="index"),
    }
    if not runtime.empty:
        summary["runtime_median_sec"] = runtime.groupby("method")["runtime_sec"].median().round(1).to_dict()
    if not support.empty:
        floor_types = settings["floor"]["cell_types"]
        floor_support = support[support.cell_type.isin(floor_types)].groupby("method")["median_spearman"].mean()
        summary["floor_marker_support_mean"] = floor_support.round(3).to_dict()
    with open(f"{table_dir}/benchmark_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    print("\nwrote tables to", table_dir, "and figures to", figure_dir)


if __name__ == "__main__":
    main()
