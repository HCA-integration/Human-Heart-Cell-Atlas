"""Immunosuppressant drug-target scores: Fig 6 k matrix and the per-core ATG score of Fig 6 l.

Fig 6 k: curated drug -> target-gene sets (params.drugs; not drug2cell / ChEMBL), per-cell sc.tl.score_genes
(CP10k, log1p, ctrl_size 50, random_state 0; hhca.tl.score_programmes), mean per cell-type group (lymphoid
Level 4, Level 3 for the rest, cardiomyocytes pooled; config classes.drug_targets), z-scored per drug across the
groups (hhca.tl.group_zscores); drawn by figures/Fig6/panel_k.py.

Fig 6 l: the per-core ATG score is the per-core mean over all cells of the same per-cell ATG score_genes value used
in Fig 6 k (params.atg_drug), so 6 k and 6 l share one definition.

Inputs:
    labels/xenium_atlas_labels.csv, labels/xenium_lymphoid8_annotation.csv (label_tables.py)
    counts.mtx, genes.txt, barcodes.txt, obs.csv, cells_geom.csv of the extraction folder (config inputs.extracted)
    GEO clinical metadata table (config inputs.metadata)

Outputs (<out>/):
    drug_target_matrix_z.csv, drug_target_matrix_mean.csv, drug_target_cells_per_group.csv, drug_target_genes.csv,
    atg_drug_target_percore.csv

Usage: python scripts/xenium/downstream/drug_targets.py
"""

import argparse

import pandas as pd
import scanpy as sc

import downstream_common as common
import hhca


def parse_args(cfg):
    """Command-line arguments; the defaults come from config/xenium_downstream.yaml."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    common.add_label_arguments(parser)
    parser.add_argument("--meta", default=cfg["inputs"]["metadata"], help="GEO clinical metadata table")
    parser.add_argument("--out", help="output folder (default: under config outputs.root)")
    return parser.parse_args()


def main():
    cfg = common.load_config()
    args = parse_args(cfg)
    params = cfg["params"]
    out = args.out or common.out_dir(cfg, "tables")

    adata = common.load_cohort(cfg, args, counts=True)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print("adata", adata.shape)

    drugs = {drug: [gene for gene in targets if gene in adata.var_names] for drug, targets in params["drugs"].items()}
    common.classify(adata, cfg, "drug_targets", key_added="ct_group")
    groups = cfg["classes"]["drug_target_order"]
    keys = hhca.tl.score_programmes(
        adata, drugs, "d_", ctrl_size=params["cytokine"]["ctrl_size"], random_state=params["cytokine"]["random_state"]
    )
    means, zscores = hhca.tl.group_zscores(adata, dict(zip(drugs, keys)), "ct_group", groups)
    cells_per_group = adata.obs.ct_group.value_counts().reindex(groups)
    print("cells per column:", cells_per_group.to_dict())
    print("z range", zscores.values.min().round(2), zscores.values.max().round(2))
    zscores.rename_axis("drug").to_csv(f"{out}/drug_target_matrix_z.csv")
    means.rename_axis("drug").to_csv(f"{out}/drug_target_matrix_mean.csv")
    cells_per_group.rename("n_cells").rename_axis("ct_group").to_csv(f"{out}/drug_target_cells_per_group.csv")
    pd.Series(drugs).map("|".join).rename("genes_on_panel").rename_axis("drug").to_csv(f"{out}/drug_target_genes.csv")

    # Fig 6 l: per-core ATG score
    atg = adata.obs.groupby("Sample", observed=True)[f"d_{params['atg_drug']}"].mean().rename("drug_target_score")
    md = common.read_meta(args.meta).set_index("Sample")
    per_core = atg.to_frame().join(md[["biopsy_cellular_grading"]])
    per_core.reset_index().to_csv(f"{out}/atg_drug_target_percore.csv", index=False)
    print(f"wrote {out}/drug_target_matrix_z.csv, atg_drug_target_percore.csv ({len(per_core)} cores)")


if __name__ == "__main__":
    main()
