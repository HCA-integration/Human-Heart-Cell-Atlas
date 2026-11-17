"""Per-donor LV expression of the hormone genes behind Fig 2 h-i.

Panel h = received (receptor genes), panel i = produced (biosynthesis genes); the shown dots come from
hormone_relevance_gate.py and their gene sets from hhca.tl.shown_gene_sets.

Left-ventricle nuclei only (tissue_general == "Left Ventricle"). layers/normcounts is read in row blocks and
averaged per (donor, Final Level 2 type, gene); the donor is the unit of replication (hhca.tl.lv_expression;
settings: config/atlas_hormones.yaml, lv_expression).

Inputs: atlas h5ad, the gate tables, the gene-set tables of hormone_genesets_export.py.
Outputs, in --outdir:
  lv_donor_L2_gene_meanNorm.npz
    means  (nD, nCT, nG) float32   mean normcounts over the LV nuclei of (donor, L2 type)
    ncells (nD, nCT)     int32     LV nuclei per (donor, L2 type)
    donors, celltypes (L2), genes, donor_dataset, donor_sex
  hormone_genesets.json            gene set of every shown dot
  lv_dataset_sex_composition.csv   LV donors per dataset x sex
The tensor has one row per atlas donor id (291); pool_lv_tensor.py pools it to the 280 donors.

Usage: python lv_hormone_tensor.py --atlas ATLAS.h5ad --gated results/atlas/hormones/g7_sexdiff \
           --genesets results/atlas/hormones/genesets --outdir results/atlas/hormones/sexdiff_291
"""

import argparse
import json
import os

import pandas as pd

import hhca
from hhca.config import load_config
from hhca.donors import write_lv_expression
from hhca.log import get_logger


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--atlas", required=True, help="HHCA atlas h5ad (layers/normcounts, obs)")
    parser.add_argument(
        "--gated", required=True, help="directory with hormone_sexdiff_relevancegated__{received,produced}.csv"
    )
    parser.add_argument("--genesets", required=True, help="directory with hormone_{receptor,producer}_gene_sets.csv")
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def main():
    """Stream the LV nuclei and write the donor tensor, the gene sets and the cohort composition."""
    args = parse_args()
    log = get_logger()
    os.makedirs(args.outdir, exist_ok=True)
    params = load_config("atlas_hormones")["lv_expression"]

    gates = {
        panel: pd.read_csv(f"{args.gated}/hormone_sexdiff_relevancegated__{panel}.csv")
        for panel in ("received", "produced")
    }
    gene_sets, genes = hhca.tl.shown_gene_sets(
        gates,
        pd.read_csv(f"{args.genesets}/hormone_producer_gene_sets.csv"),
        pd.read_csv(f"{args.genesets}/hormone_receptor_gene_sets.csv"),
    )
    print(
        f"hormones: {len(gene_sets['received'])} received, {len(gene_sets['produced'])} produced; "
        f"{len(genes)} genes in total"
    )

    expression = hhca.tl.lv_expression(args.atlas, genes, log=log, **params)
    missing = list(expression.uns["genes_missing"])
    print(f"genes in the atlas: {expression.n_vars}; missing {len(missing)}: {missing}")
    write_lv_expression(f"{args.outdir}/lv_donor_L2_gene_meanNorm.npz", expression)
    with open(f"{args.outdir}/hormone_genesets.json", "w") as handle:
        json.dump(
            {"genesets": gene_sets, "genes_present": list(expression.var_names), "genes_missing": missing},
            handle,
            indent=2,
        )

    # LV cohort composition (donors with LV nuclei)
    composition = hhca.tl.cohort_composition(expression)
    composition.to_csv(f"{args.outdir}/lv_dataset_sex_composition.csv")
    print("\nLV donors per dataset x sex")
    print(composition.to_string())


if __name__ == "__main__":
    main()
