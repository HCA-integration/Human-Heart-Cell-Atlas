#!/usr/bin/env python
"""Fig 3 c: cardiomyocyte chamber-marker panel on cardiomyocyte-dominant Visium spots.

The snRNA-derived chamber panel (atrial and ventricular cardiomyocyte genes) is averaged per chamber region
(LA, RA, LV, RV) over the Noseda and Kanemaru Visium sections, once over all spots and once over the spots
whose top deconvolved cell type is a cardiomyocyte. Restricting to cardiomyocyte-dominant spots separates
expression differences within cardiomyocytes from differences in how much myocardium each chamber section holds.

  Noseda    obs['dominant_L3'] of the NNLS h5ads is Atrial or Ventricular Cardiomyocytes
  Kanemaru  obsm['prop'] (72 cell2location types, stored in the objects) summed to coarse lineages; the top
            lineage is aCM1-5 (atrial) or vCM1-5 (ventricular). Pacemaker / conduction cells and the
            CMC-like pericyte and endothelial states do not count as cardiomyocytes.

Both sums are taken in one pass per slide (hhca.tl.cardiomyocyte_dominant, hhca.tl.chamber_sums), with one slide
in memory at a time, and pooled per region (hhca.tl.region_means). The six Noseda
ventricular slides without an LV / RV assignment are not used.

Inputs:  the chamber gene panel (concordance_per_gene.csv), <noseda-h5ad>/<slide>_L3_deconv.h5ad,
         <kanemaru>/visium-OCT_<region>_lognormalised.h5ad
Outputs: <out>/tables/ (region means and z-scores for all spots and CM-dominant spots, their difference,
         CMdominant_spots.csv.gz, spot fractions by slide and region) and <out>/figures/

Usage: build_cmdominant_panel.py --conc concordance_per_gene.csv --noseda-h5ad DIR --kanemaru DIR --out DIR
           [--config CONFIG]
"""

import argparse
import json
import os
import sys
import warnings

from hhca import thread_env

os.environ.update(thread_env(1, numba=False))
# silence the import-time warnings of the libraries below, unless warnings were asked for on the command line
if not sys.warnoptions:
    warnings.filterwarnings("ignore")

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc

import deconv_config
import hhca
from hhca.log import get_logger
from hhca.pl import style

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ATRIA_COLOUR, VENTRICLE_COLOUR = style.CHAMBER_GENES["atria"], style.CHAMBER_GENES["ventricle"]
DIRECTION_COLOURS = style.CHAMBER_GENES
COHORTS = {"kan": "kanemaru", "nos": "noseda"}
REGIONS = ["LA", "RA", "LV", "RV"]  # chamber regions of the figures


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--conc", required=True, help="chamber gene panel (concordance_per_gene.csv)")
    parser.add_argument("--noseda-h5ad", required=True, help="directory with the NNLS h5ads <slide>_L3_deconv.h5ad")
    parser.add_argument("--kanemaru", required=True, help="directory with the Kanemaru Visium h5ad files")
    parser.add_argument("--out", required=True, help="output directory (tables/ and figures/)")
    return deconv_config.add_argument(parser).parse_args()


def slide_sums(adata, genes, cohort, region, label, log):
    """Panel sums of one slide (log-normalised expression in X), its spots and its row of the slide table."""
    sums = hhca.tl.chamber_sums(adata, genes)
    sums.insert(0, "slide", label)
    sums.insert(0, "region", region)
    sums.insert(0, "cohort", cohort)
    cm_mask = adata.obs["cm_dominant"].to_numpy()
    n_total, n_cm = int(cm_mask.shape[0]), int(cm_mask.sum())
    spots = pd.DataFrame({"cohort": cohort, "slide": label, "barcode": adata.obs_names, "cm_dominant": cm_mask})
    row = {
        "cohort": cohort,
        "slide": label,
        "region": region,
        "n_total": n_total,
        "n_cm_dominant": n_cm,
        "cm_dominant_frac": (n_cm / n_total if n_total else np.nan),
    }
    if n_cm:
        row["median_CMprop_in_kept"] = float(np.nanmedian(adata.obs["cm_proportion"].to_numpy()[cm_mask]))
    log(f"  {cohort} {label} [{region}]: {n_cm}/{n_total} CM-dominant ({100 * n_cm / max(n_total, 1):.1f}%)")
    return sums, spots, row


def kanemaru_slide(kanemaru_dir, region, cardiomyocytes):
    """Kanemaru section of one region (X is log-normalised); CM-dominant by the top coarse lineage."""
    adata = ad.read_h5ad(f"{kanemaru_dir}/visium-OCT_{region}_lognormalised.h5ad")
    adata.var_names_make_unique()
    assert set(adata.obs["region"].astype(str)) == {region}
    lineages = hhca.tl.lineage_proportions(adata.obsm["prop"])  # 72 proportions per spot, rows sum to 1
    hhca.tl.cardiomyocyte_dominant(adata, cardiomyocytes, proportions=lineages)
    return adata


def noseda_slide(noseda_dir, slide, cardiomyocytes):
    """Noseda slide (raw counts, log-normalised here); CM-dominant by the dominant Level-3 type of the NNLS."""
    adata = ad.read_h5ad(f"{noseda_dir}/{slide}_L3_deconv.h5ad")
    adata.var_names_make_unique()
    hhca.tl.cardiomyocyte_dominant(adata, cardiomyocytes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def write_run_info(path, region_counts):
    """Definition of the CM-dominant spots and the combined fraction per region, as json."""
    by_region = region_counts.set_index("region")
    run_info = {
        "noseda_props": None,
        "kanemaru_props": None,
        "definition": "CM-dominant = argmax deconvolved cell type is Atrial-CM or Ventricular-CM",
        "noseda_criterion": "obs['dominant_L3'] in {Atrial Cardiomyocytes, Ventricular Cardiomyocytes}",
        "kanemaru_criterion": (
            "argmax over coarse lineages aggregated from 72 cell2location "
            "obsm['prop'] fine types; CM = aCM1-5 (Atrial-CM), vCM1-5 (Ventricular-CM); "
            "SAN/AVN pacemaker + *CMC-like NOT counted as CM"
        ),
        "both_cohorts_have_deconv": True,
        "regions": REGIONS,
        "combined_frac": {
            region: (float(by_region.loc[region, "combined_cm"]) / float(by_region.loc[region, "combined_total"]))
            for region in REGIONS
        },
    }
    with open(path, "w") as handle:
        json.dump(run_info, handle, indent=2)


def style_axes(ax, panel, title, region_n=None):
    """Region and gene tick labels (genes coloured by chamber), separators between gene groups and chambers."""
    gene_order, direction, n_atria, extra_gene = panel
    ax.set_xticks(range(4))
    if region_n is not None:
        ax.set_xticklabels(
            [f"{r}\n{region_n[r] // 1000}k" if region_n[r] >= 1000 else f"{r}\n{region_n[r]}" for r in REGIONS]
        )
    else:
        ax.set_xticklabels(REGIONS)
    ax.set_yticks(range(len(gene_order)))
    ax.set_yticklabels(gene_order)
    for tick_label, gene in zip(ax.get_yticklabels(), gene_order):
        tick_label.set_color(DIRECTION_COLOURS[direction[gene]])
        if gene == extra_gene:
            tick_label.set_fontstyle("italic")
            tick_label.set_fontweight("bold")
    ax.axhline(n_atria - 0.5, color="black", lw=1.2)
    ax.axvline(1.5, color="black", lw=1.2)
    ax.set_title(title, fontsize=9.5, pad=8)
    ax.tick_params(length=2)


def hatch_nan(ax, zscores):
    """Hatch the cells without a value (gene not measured in the region)."""
    n_genes, n_regions = zscores.shape
    for gene_row in range(n_genes):
        for region_col in range(n_regions):
            if not np.isfinite(zscores.values[gene_row, region_col]):
                ax.add_patch(
                    plt.Rectangle(
                        (region_col - 0.5, gene_row - 0.5), 1, 1, facecolor="0.85", edgecolor="white", hatch="///", lw=0
                    )
                )


def draw_heatmap(ax, zscores, panel, title, region_n=None, cmap="RdBu_r", vmax=None):
    """Genes x regions heatmap on a symmetric colour scale; returns the image."""
    if vmax is None:
        vmax = float(np.nanmax(np.abs(zscores.values)))
    im = ax.imshow(zscores.values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    style_axes(ax, panel, title, region_n)
    hatch_nan(ax, zscores)
    return im


def label_gene_groups(ax, x, panel):
    """Vertical 'atria genes' / 'ventricle genes' labels left of the gene names."""
    gene_order, _, n_atria, _ = panel
    for y, text, colour in [
        ((n_atria - 1) / 2, "atria genes", ATRIA_COLOUR),
        ((n_atria + len(gene_order) - 1) / 2, "ventricle genes", VENTRICLE_COLOUR),
    ]:
        ax.text(
            x,
            y,
            text,
            rotation=90,
            va="center",
            ha="center",
            color=colour,
            fontsize=8.5,
            transform=ax.get_yaxis_transform(which="grid"),
        )


def save_figure(fig, stem):
    """Write a figure as png and pdf."""
    for extension in ("png", "pdf"):
        fig.savefig(f"{stem}.{extension}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_cm_dominant(path, z_cm, panel, cm_counts):
    """CM-dominant spots, both cohorts pooled."""
    fig, ax = plt.subplots(figsize=(4.3, 6.6))
    im = draw_heatmap(
        ax,
        z_cm,
        panel,
        "snRNA-derived cardiomyocyte chamber panel\n"
        "across chamber Visium slides, CM-dominant spots only\n"
        "(Noseda + Kanemaru; argmax deconvolved type = CM)",
        region_n=cm_counts,
    )
    colorbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    colorbar.set_label("z-scored mean log1p expression\n(across the four regions)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7.5)
    label_gene_groups(ax, -0.78, panel)
    ax.text(
        0.5,
        -0.11,
        "x-axis labels show CM-dominant spot count per region",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
        color="0.35",
    )
    save_figure(fig, path)


def figure_side_by_side(path, z_all, z_cm, panel, all_counts, cm_counts):
    """All spots next to CM-dominant spots, same gene order and colour scale."""
    vmax_shared = float(np.nanmax(np.abs(np.concatenate([z_all.values.ravel(), z_cm.values.ravel()]))))
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 6.6), sharey=True)
    draw_heatmap(axes[0], z_all, panel, "All spots", region_n=all_counts, vmax=vmax_shared)
    im = draw_heatmap(axes[1], z_cm, panel, "CM-dominant spots", region_n=cm_counts, vmax=vmax_shared)
    axes[0].tick_params(axis="y", length=0, labelleft=True)
    axes[1].tick_params(axis="y", length=0, labelleft=False)
    label_gene_groups(axes[0], -0.95, panel)
    colorbar = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.03)
    colorbar.set_label("z-scored mean log1p expression\n(across the four regions)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7.5)
    fig.suptitle(
        "CM chamber panel: all spots vs CM-dominant spots (shared gene order and scale)\n"
        "x-axis labels show spot count per region",
        fontsize=9.5,
        y=0.99,
    )
    save_figure(fig, path)


def figure_difference(path, z_all, z_cm, panel):
    """Difference z(CM-dominant) - z(all spots)."""
    fig, ax = plt.subplots(figsize=(4.3, 6.6))
    im = draw_heatmap(
        ax,
        z_cm - z_all,
        panel,
        "Change from restricting to CM-dominant spots\nz(CM-dominant) - z(all spots)",
        region_n=None,
        cmap="PuOr_r",
    )
    colorbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    colorbar.set_label("difference in z-scored expression", fontsize=8)
    colorbar.ax.tick_params(labelsize=7.5)
    label_gene_groups(ax, -0.78, panel)
    save_figure(fig, path)


def figure_fractions(path, region_counts):
    """Fraction of CM-dominant spots per region and cohort."""
    by_region = region_counts.set_index("region")
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    x = np.arange(len(REGIONS))
    width = 0.38
    kanemaru = [by_region.loc[region, "kanemaru_frac"] for region in REGIONS]
    noseda = [by_region.loc[region, "noseda_frac"] for region in REGIONS]
    kanemaru_bars = ax.bar(x - width / 2, kanemaru, width, label="Kanemaru", color="#009E73")
    noseda_bars = ax.bar(x + width / 2, noseda, width, label="Noseda", color="#CC79A7")
    ax.set_xticks(x)
    ax.set_xticklabels(REGIONS)
    ax.set_ylabel("fraction of spots that are CM-dominant")
    ax.set_ylim(0, 1.14)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title("CM-dominant spot fraction per chamber region", pad=10)
    for bars, fractions in [(kanemaru_bars, kanemaru), (noseda_bars, noseda)]:
        for bar, value in zip(bars, fractions):
            ax.text(
                bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=7
            )
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, path)


def main():
    """Accumulate the panel per region over both cohorts, then write the tables and figures."""
    args = parse_args()
    log = get_logger()
    cfg = deconv_config.load(args.config)
    regions = cfg["slides"]["kanemaru_regions"]
    assert regions == REGIONS, regions
    extra_gene = cfg["downstream"]["fig3c_extra_gene"]
    table_dir = f"{args.out}/tables"
    figure_dir = f"{args.out}/figures"
    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    direction = hhca.tl.chamber_gene_panel(pd.read_csv(args.conc), extra=[extra_gene])
    gene_order, n_atria = list(direction.index), int((direction == "atria").sum())
    assert n_atria == 12 and len(gene_order) == 17, (n_atria, len(gene_order))
    panel = (gene_order, direction.to_dict(), n_atria, extra_gene)
    log(f"panel: {n_atria - 1} atria + {extra_gene} + {len(gene_order) - n_atria} ventricle = {len(gene_order)} genes")

    results = []
    for region in regions:
        adata = kanemaru_slide(args.kanemaru, region, cfg["downstream"]["fig3c_cm_kanemaru"])
        results.append(slide_sums(adata, gene_order, "kan", region, f"visium-OCT_{region}", log))
    for slide, region in cfg["slides"]["fig3c_region"].items():
        adata = noseda_slide(args.noseda_h5ad, slide, cfg["downstream"]["fig3c_cm"])
        results.append(slide_sums(adata, gene_order, "nos", region, slide, log))
    sums = pd.concat([result[0] for result in results], ignore_index=True)

    means_all, means_cm = hhca.tl.region_means(sums, regions, "all"), hhca.tl.region_means(sums, regions, "cm")
    z_all, z_cm = hhca.tl.region_zscores(means_all), hhca.tl.region_zscores(means_cm)

    means_cm.to_csv(f"{table_dir}/CMdominant_region_mean_log1p.csv")
    z_cm.to_csv(f"{table_dir}/CMdominant_region_zscore.csv")
    (z_cm - z_all).to_csv(f"{table_dir}/CMdominant_minus_allspots_zscore.csv")
    pd.concat([result[1] for result in results]).to_csv(
        f"{table_dir}/CMdominant_spots.csv.gz", index=False, compression="gzip"
    )
    means_all.to_csv(f"{table_dir}/allspots_region_mean_log1p.csv")
    z_all.to_csv(f"{table_dir}/allspots_region_zscore.csv")
    pd.DataFrame([result[2] for result in results]).to_csv(
        f"{table_dir}/CMdominant_spot_fractions_by_slide.csv", index=False
    )

    region_counts = hhca.tl.region_spot_counts(sums, regions, COHORTS)
    region_counts.to_csv(f"{table_dir}/CMdominant_spot_fractions_by_region.csv", index=False)
    log(
        "CM-dominant spot fractions per region (combined):\n"
        + region_counts[["region", "combined_total", "combined_cm", "combined_frac"]].to_string(index=False)
    )
    write_run_info(f"{table_dir}/run_info.json", region_counts)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "figure.dpi": 300,
        }
    )
    by_region = region_counts.set_index("region")
    cm_counts = {region: int(by_region.loc[region, "combined_cm"]) for region in regions}
    all_counts = {region: int(by_region.loc[region, "combined_total"]) for region in regions}

    figure_cm_dominant(f"{figure_dir}/combined_heatmap_panel_by_region_CMdominant", z_cm, panel, cm_counts)
    log("wrote combined_heatmap_panel_by_region_CMdominant")
    figure_side_by_side(f"{figure_dir}/sidebyside_allspots_vs_CMdominant", z_all, z_cm, panel, all_counts, cm_counts)
    log("wrote sidebyside_allspots_vs_CMdominant")
    figure_difference(f"{figure_dir}/difference_CMdominant_minus_allspots", z_all, z_cm, panel)
    log("wrote difference_CMdominant_minus_allspots")
    figure_fractions(f"{figure_dir}/CMdominant_fraction_per_region", region_counts)
    log("wrote CMdominant_fraction_per_region")
    log("done")


if __name__ == "__main__":
    main()
