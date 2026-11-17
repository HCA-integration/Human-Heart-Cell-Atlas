#!/usr/bin/env python
"""Fig 5 g, h data: drug2cell scores of myelinated vs non-myelinated nerves from the per-nerve pseudobulks.

  pool: concat of the per-slide nerve pseudobulks (counts layer), join = inner over the genes of the inputs;
        group = myelinated (myelin_total >= 3) / non_myelinated (== 0), intermediate dropped
        (hhca.tl.myelination_groups); n_bins >= 15 and total_counts >= 80; normalize_total(1e4) + log1p
  score: drug2cell 0.1.2 `score` (bundled ChEMBL dictionary; uns['drug2cell'].X, method "mean") with MBP, PLP1,
        MOG, MPZ, PMP22 (the genes that define the groups) removed from every target set
        (hhca.tl.score_drug_targets); g_summary.json also gives the largest difference between this score and the
        mean of the log1p CP10k values over each drug's target genes
  g: per-nerve aggregate = mean over the 7 CIPN drugs' scores (hhca.tl.mean_drug_score); one-sided Mann-Whitney
     (myelinated greater)
  h: per drug: delta mean score (myelinated - non), two-sided Mann-Whitney, BH over drugs with a p value;
     targets detected = in >= 10% of the nerves (hhca.tl.drug_differences)

Inputs: the per-nerve pseudobulk h5ads of nerves.py; drug lists and thresholds of config/nerve_macrophage.yaml.
Outputs: g_per_nerve.csv, g_summary.json, h_stats.csv, universe.json.

Usage: python cipn_drug2cell.py --pseudobulk P1.h5ad ... --outdir results/nerve_macrophage/gh
"""

import argparse
import json
import os
import sys
import warnings

from hhca import config_path, thread_env

os.environ.update(thread_env(1, numba=False))
if not sys.warnoptions:
    warnings.filterwarnings("ignore")

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

import hhca
from hhca.config import load_config


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pseudobulk", nargs="+", required=True, help="per-nerve pseudobulk h5ads, one per slide")
    parser.add_argument("--outdir", required=True, help="output folder")
    parser.add_argument(
        "--config",
        help="pipeline config with the drug lists and thresholds",
        default=config_path("nerve_macrophage"),
    )
    return parser.parse_args()


def main():
    """Score the pooled nerves and write the tables of Fig 5 g and h."""
    args = parse_args()
    cfg = load_config(args.config)
    drugs, myelin_genes = cfg["drugs"], cfg["nerve"]["myelin_genes"]
    os.makedirs(args.outdir, exist_ok=True)

    nerves = ad.concat([ad.read_h5ad(path) for path in args.pseudobulk], join="inner", index_unique=None)
    nerves.obs_names_make_unique()
    hhca.tl.myelination_groups(nerves, cfg["nerve"]["myelinated_min"])
    keep = (
        (nerves.obs.n_bins >= drugs["min_bins"])
        & (nerves.obs.total_counts >= drugs["min_counts"])
        & (nerves.obs.group != "intermediate")
    )
    nerves = nerves[keep].copy()
    sc.pp.normalize_total(nerves, target_sum=1e4)
    sc.pp.log1p(nerves)
    myelinated = (nerves.obs.group == "myelinated").to_numpy()
    non_myelinated = (nerves.obs.group == "non_myelinated").to_numpy()
    print(f"nerves: {nerves.n_obs}  myel={myelinated.sum()} non={non_myelinated.sum()}  genes={nerves.n_vars}")
    hhca.tl.score_drug_targets(nerves, exclude=myelin_genes)

    # g: mean score over the neurotoxic drugs, myelinated against non-myelinated nerves
    score = hhca.tl.mean_drug_score(nerves, drugs["g_chembl"])
    U, p = stats.mannwhitneyu(score[myelinated], score[non_myelinated], alternative="greater")
    print(
        f"g: myel n={myelinated.sum()} med={np.median(score[myelinated]):.2f} | non n={non_myelinated.sum()} "
        f"med={np.median(score[non_myelinated]):.2f} | MWU p={p:.2e}"
    )
    per_nerve = nerves.obs[["slide", "n_bins", "myelin_total", "total_counts", "group"]].copy()
    per_nerve["cipn_score"] = score
    per_nerve.to_csv(f"{args.outdir}/g_per_nerve.csv")
    summary = {
        "score": "drug2cell",
        "exclude_myelin_targets": True,
        "n_myelinated": int(myelinated.sum()),
        "n_non_myelinated": int(non_myelinated.sum()),
        "median_myelinated": float(np.median(score[myelinated])),
        "median_non_myelinated": float(np.median(score[non_myelinated])),
        "mwu_U": float(U),
        "mwu_p_greater": float(p),
        "n_drugs_scored": int(hhca.tl.drug_scores(nerves, drugs["g_chembl"]).shape[1]),
        "max_abs_diff_manual_vs_drug2cell": hhca.tl.target_mean_difference(nerves, drugs["g_chembl"]),
    }
    with open(f"{args.outdir}/g_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    # h: every drug on its own
    per_drug = hhca.tl.drug_differences(
        nerves, drugs["h_cipn"], exclude=myelin_genes, detected_frac=drugs["detected_frac"]
    )
    per_drug.to_csv(f"{args.outdir}/h_stats.csv", index=False)
    pd.set_option("display.width", 200)
    print(
        per_drug[["drug", "cls", "in_d2c", "n_targets", "n_detected", "diff", "p", "fdr", "detected_genes"]].to_string(
            index=False
        )
    )
    coverage = hhca.tl.drug_target_coverage(nerves, drugs["h_cipn"], myelin_genes, drugs["check_targets"])
    with open(f"{args.outdir}/universe.json", "w") as handle:
        json.dump(coverage, handle, indent=2)


if __name__ == "__main__":
    main()
