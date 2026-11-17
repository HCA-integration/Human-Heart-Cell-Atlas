"""Fig 3 c: snRNA-derived cardiomyocyte chamber panel on cardiomyocyte-dominant Visium spots.

Row z-score of the pooled mean log1p expression per chamber region (Noseda and Kanemaru sections); a spot is
cardiomyocyte-dominant when its top deconvolved cell type is a cardiomyocyte (Noseda: NNLS; Kanemaru: the
cell2location result stored in the objects).

Inputs:  results/deconvolution/downstream/fig3c/tables/CMdominant_{region_zscore,spot_fractions_by_region}.csv
         resources/deconvolution/concordance_per_gene.csv (atrial / ventricular direction of each gene)
Outputs: results/figures/Fig3/panel_c.{png,pdf}

Usage: python figures/Fig3/panel_c.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import pandas as pd

import hhca
from hhca.pl import save_panel

CONCORDANCE = str(hhca.REPO / "resources" / "deconvolution" / "concordance_per_gene.csv")
REGIONS = ["LA", "RA", "LV", "RV"]
EXTRA_GENE = "POTEJ"  # atrial gene outside the panel, shown with it


def main():
    """Draw the genes x regions heatmap of the CM-dominant spots."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    table_dir = hhca.results_dir("deconvolution", "downstream", "fig3c", "tables")
    zscores = pd.read_csv(f"{table_dir}/CMdominant_region_zscore.csv", index_col=0)[REGIONS]
    region_counts = pd.read_csv(f"{table_dir}/CMdominant_spot_fractions_by_region.csv").set_index("region")
    cm_counts = {region: int(region_counts.loc[region, "combined_cm"]) for region in REGIONS}
    direction = pd.read_csv(CONCORDANCE).drop_duplicates("gene").set_index("gene")["snuc_dir"].reindex(zscores.index)
    fig = hhca.pl.chamber_markers(zscores, direction, cm_counts, highlight=[EXTRA_GENE])
    print(save_panel(fig, "Fig3", "c"), "| CM-dominant spots:", cm_counts)


if __name__ == "__main__":
    main()
