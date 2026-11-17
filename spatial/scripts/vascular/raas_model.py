#!/usr/bin/env python
"""Fig 4 i, j: richness-adjusted RAAS-enzyme levels across vessel types.

Raw per-vessel-type means are confounded by transcript richness (library size), so enzyme levels are compared on
marginal means adjusted for richness and slide.

Model (one row per structure and gene):
  y = log1p(CP10k), CP10k = 1e4 * count / lib
  y ~ C(gene):C(vessel_type)              # 25 cell means
    + C(gene):log_lib_centred             # gene-specific richness slope
    + C(gene):C(slide, Sum)               # gene-specific slide effects (deviation coded)
  All genes are fitted jointly so that contrasts between enzymes share one covariance; standard errors are
  cluster-robust with the structure as cluster (each structure contributes all five genes).

  Slide is sum-to-zero coded and log-lib is centred, so each cell-mean coefficient is the estimated marginal
  mean (EMM) at the mean richness, balanced over slides, and any contrast is a difference of two coefficients
  with its variance read from the covariance matrix.

The model is fitted by hhca.tl.fit_raas; genes, vessel types and thresholds: `raas` in config/vascular.yaml.

Input: the pooled per-structure table of raas_substrate.py.
Outputs (in --out-dir):
  raas_emm_by_gene_vessel.csv                   adjusted EMMs (log1p scale + adj CP10k) + CI (Fig 4 i, j)
  raas_pairwise_significant_orderings.csv       per vessel type: X>Y orderings + FDR + CI + slide robustness
  raas_across_vessel_pairs.csv                  per enzyme: across-vessel contrasts
  raas_across_vessel_by_enzyme.csv              per enzyme: where highest, top vs second
  raas_slide_pseudobulk_rates.csv               per (slide, vessel type) pseudobulk CP10k rates
  _fit.npz                                      coefficients, covariance, compact letter displays

Usage: python raas_model.py --parquet pooled_struct_uniform.parquet --out-dir DIR
"""

from __future__ import annotations

import argparse
import os

from hhca import thread_env

os.environ.update(thread_env(1, numba=False, keep_existing=True))

import numpy as np
import pandas as pd

import hhca
from hhca.config import load_config


def print_summary(fit, genes, vessel_types):
    """Print the EMM table with letters and the significant orderings per vessel type."""
    params, cell_idx, orderings = fit["params"], fit["cell_index"], fit["orderings"]
    cld, cld_rob = fit["letters"], fit["letters_robust"]
    print("\nrichness-adjusted EMM, log1p(CP10k), with letter display")
    print("vessel_type".ljust(24) + "".join(gene.rjust(16) for gene in genes))
    for vessel in vessel_types:
        cells = [f"{params[cell_idx[(gene, vessel)]]:.3f}/{cld[vessel][gene]}".rjust(16) for gene in genes]
        print(vessel.ljust(24) + "".join(cells))
    print("\n(enzymes sharing a letter are not significantly different, BH-FDR)")

    print("\norderings per vessel type (letters: FDR < 0.05 and robust across slides)")
    for vessel in vessel_types:
        chain = "  ".join(
            f"{gene}[{cld_rob[vessel][gene]}]"
            for gene in sorted(genes, key=lambda gene: -params[cell_idx[(gene, vessel)]])
        )
        print(f"  {vessel}: {chain}")
        significant = orderings[(orderings.vessel_type == vessel) & (orderings.significant)]
        listed = ", ".join(
            f"{row.higher_enzyme}>{row.lower_enzyme}" + ("" if row.robust_across_slides else "*")
            for row in significant.itertuples()
        )
        print(f"      significant X>Y: {listed if listed else '(none)'}")
    print("  (* = FDR-significant across structures, but fewer than 70 % of the slides agree)")


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--parquet", required=True, help="pooled per-structure table (raas_substrate.py pool)")
    parser.add_argument("--out-dir", required=True, help="output directory")
    return parser.parse_args()


def main():
    """Fit the joint model and write the EMM, contrast and ordering tables."""
    args = parse_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    raas = load_config("vascular")["raas"]
    genes, vessel_types = raas["model_genes"], raas["vessel_types"]

    fit = hhca.tl.fit_raas(
        pd.read_parquet(args.parquet), genes, vessel_types, raas["alpha"], raas["min_bins"], raas["robust_fraction"]
    )
    structs = fit["structures"]
    print(f"[data] structures={structs.shape[0]:,}  slides={structs['slide'].nunique()}")
    print("[data] structures per vessel type:")
    print(structs["vessel_type"].value_counts().reindex(vessel_types).to_string())
    print(f"[model] design {fit['design_shape']}  full column rank: {fit['full_rank']}")

    fit["emm"].to_csv(f"{out_dir}/raas_emm_by_gene_vessel.csv", index=False)
    fit["rates"].to_csv(f"{out_dir}/raas_slide_pseudobulk_rates.csv", index=False)
    fit["orderings"].to_csv(f"{out_dir}/raas_pairwise_significant_orderings.csv", index=False)
    fit["across"].to_csv(f"{out_dir}/raas_across_vessel_pairs.csv", index=False)
    fit["highest"].to_csv(f"{out_dir}/raas_across_vessel_by_enzyme.csv", index=False)
    print_summary(fit, genes, vessel_types)

    cld, cld_rob, n_struct = fit["letters"], fit["letters_robust"], fit["n_structures"]
    np.savez(
        f"{out_dir}/_fit.npz",
        params=fit["params"],
        cov=fit["cov"],
        names=np.array(fit["names"], dtype=object),
        genes=np.array(genes, dtype=object),
        vt=np.array(vessel_types, dtype=object),
        cld=np.array([[cld[vessel][gene] for gene in genes] for vessel in vessel_types], dtype=object),
        cld_rob=np.array([[cld_rob[vessel][gene] for gene in genes] for vessel in vessel_types], dtype=object),
        nstruct=np.array([n_struct.get(vessel, 0) for vessel in vessel_types]),
    )
    print(f"\nwrote the tables and _fit.npz to {out_dir}")


if __name__ == "__main__":
    main()
