"""Fig 3 l: transmural transcript slopes of ventricular cardiomyocytes and their concordance with the DVP
protein slopes.

Transcript slope per gene = OLS of log1p CP10k (cardiomyocyte pseudobulk per section) on position (epi 0, myo 1,
endo 2) over all sections, with per-donor slopes and their sign consistency; when every donor has three sections
the table also carries the layer means and their monotonicity (hhca.tl.transmural_slopes). Protein slope = mean
of the two DVP cohorts (hhca.tl.protein_slopes). The curated gene panels of the concordance statistics are set in
config/cardiomyocyte.yaml (fig3l.gene_panels).

Inputs: cm_pseudobulk_counts.csv of transmural_pseudobulk.py (--tab) and the DVP trend table (--dvp, xlsx).
Outputs: in --out-dir, transmural_slopes_CM.csv, concordance_CM_protein_vs_transcript.csv, concordance_stats.csv
(Spearman rho and sign agreement genome-wide and for the gene panels).

Usage: transmural_slopes.py --tab DIR --dvp trend_analysis_all.xlsx --out-dir DIR [--donors H0054,H0063]
"""

import os

from hhca import thread_env

os.environ.update(thread_env(1, numba=False))

import argparse

import pandas as pd

import hhca
from hhca.config import load_config


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="Transmural transcript slopes and DVP protein concordance (Fig 3 l)")
    parser.add_argument("--tab", required=True, help="folder with cm_pseudobulk_counts.csv")
    parser.add_argument("--dvp", required=True, help="DVP trend table (trend_analysis_all.xlsx)")
    parser.add_argument("--out-dir", required=True, help="output folder")
    parser.add_argument(
        "--donors", default="H0054,H0063", help="comma-separated donors (sample columns <donor>_<layer>)"
    )
    return parser.parse_args()


def main():
    """Fit the transcript slopes, join the protein slopes and write the concordance tables."""
    args = parse_args()
    cfg = load_config("cardiomyocyte")
    os.makedirs(args.out_dir, exist_ok=True)
    donors = args.donors.split(",")
    layers = cfg["transmural"]["layers"]

    counts = pd.read_csv(f"{args.tab}/cm_pseudobulk_counts.csv", index_col=0)
    transcript = hhca.tl.transmural_slopes(hhca.tl.log_cp10k_pseudobulk(counts, donors, layers), donors, layers)
    transcript.to_csv(f"{args.out_dir}/transmural_slopes_CM.csv")

    protein = hhca.tl.protein_slopes(pd.read_excel(args.dvp))
    joined = transcript.join(
        protein[
            [
                "protein_CM_consensus",
                "cohort1_CM_slope",
                "cohort2_CM_slope",
                "protein_CM_sign_consistent",
                "protein_CM_qmax",
            ]
        ],
        how="inner",
    )
    joined = joined.rename(columns={"slope": "transcript_CM_slope"})
    joined.index.name = "gene"
    joined.to_csv(f"{args.out_dir}/concordance_CM_protein_vs_transcript.csv")

    panels = {"genome_wide": joined.index, "cm_structural_axis": cfg["fig3l"]["structural_axis"]}
    panels.update(cfg["fig3l"]["gene_panels"])
    statistics = []
    for panel, genes in panels.items():
        of_panel = joined[joined.index.isin(genes)]
        statistics.append(
            {
                "set": panel,
                **hhca.tl.slope_concordance(of_panel["transcript_CM_slope"], of_panel["protein_CM_consensus"]),
            }
        )
    statistics = pd.DataFrame(statistics)
    statistics.to_csv(f"{args.out_dir}/concordance_stats.csv", index=False)
    print(statistics.to_string())


if __name__ == "__main__":
    main()
