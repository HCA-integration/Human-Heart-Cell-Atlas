"""Per-core composition, DBSCAN foci and huCIRA cytokine-programme scores over immune cells.

Immune = atlas_immune; frac_NK = Level-4 NK fraction (NK +/- proliferating). Foci: DBSCAN eps 30 um, min_samples 5
on immune centroids per core (hhca.tl.immune_foci). Cytokine programmes = program_panel_genes.csv, per-cell
sc.tl.score_genes (CP10k, log1p, ctrl_size 50, random_state 0; hhca.tl.score_programmes) on all cells, averaged
over immune cells per core.

Inputs:
    labels/xenium_atlas_labels.csv, labels/xenium_lymphoid8_annotation.csv (label_tables.py)
    counts.mtx, genes.txt, barcodes.txt, obs.csv, cells_geom.csv of the extraction folder (config inputs.extracted)
    shared/program_panel_genes.csv (program_genes.py)

Outputs (<out>/):
    amr_axis_features_ATLAS.csv (Fig 6 d / ED9 a bars)
    per_biopsy_scores_immune_ATLAS.csv (Fig 6 h, i, ED9 b-c)

Usage: python scripts/xenium/downstream/core_features.py
"""

import argparse
import os

import pandas as pd
import scanpy as sc

import downstream_common as common
import hhca


def parse_args(cfg):
    """Command-line arguments; the defaults come from config/xenium_downstream.yaml."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    common.add_label_arguments(parser)
    parser.add_argument(
        "--programs",
        default=os.path.join(common.out_dir(cfg, "shared"), "program_panel_genes.csv"),
        help="programme gene table of program_genes.py",
    )
    parser.add_argument("--out", help="output folder (default: under config outputs.root)")
    return parser.parse_args()


def immune_programme_scores(adata, programmes, params):
    """Per-core mean over immune cells of the per-cell programme scores (scored on all cells)."""
    cytokines = programmes.cytokine.tolist()
    keys = hhca.tl.score_programmes(
        adata, hhca.tl.gene_sets(programmes), "c_", ctrl_size=params["ctrl_size"], random_state=params["random_state"]
    )
    immune = adata.obs[adata.obs.atlas_immune].copy()
    scores = immune.groupby("Sample", observed=True)[keys].mean()
    scores.columns = [f"score_{c}" for c in cytokines]
    scores["n_cells"] = immune.groupby("Sample", observed=True).size()
    meta = adata.obs.groupby("Sample", observed=True).agg(
        patient_id=("Sample", "first"),
        biopsy_rejection_type=("rej", "first"),
        biopsy_cellular_grading=("biopsy_cellular_grading", "first"),
        biopsy_antibody_grading=("biopsy_antibody_grading", "first"),
    )
    return meta.join(scores).reset_index(), cytokines


def main():
    cfg = common.load_config()
    args = parse_args(cfg)
    out = args.out or common.out_dir(cfg, "tables")
    os.makedirs(out, exist_ok=True)

    print("loading counts ...")
    adata = common.load_cohort(cfg, args, counts=True)
    common.classify(adata, cfg, "immune")
    adata.obs["rej"] = adata.obs.biopsy_rejection_type.astype(str).str.strip()
    print("cells", adata.n_obs, "| atlas immune", int(adata.obs.atlas_immune.sum()))

    dbscan = cfg["params"]["dbscan"]
    foci = hhca.tl.immune_foci(adata, "atlas_immune", eps=dbscan["eps_um"], min_samples=dbscan["min_samples"])
    composition = hhca.tl.core_features(adata, foci, cfg["labels"]["nk"])  # NK: Level 4, +/- proliferating
    composition.to_csv(f"{out}/amr_axis_features_ATLAS.csv", index=False)
    print("wrote amr_axis_features_ATLAS.csv", composition.shape)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    scores, cytokines = immune_programme_scores(adata, pd.read_csv(args.programs), cfg["params"]["cytokine"])
    scores.to_csv(f"{out}/per_biopsy_scores_immune_ATLAS.csv", index=False)
    print("wrote per_biopsy_scores_immune_ATLAS.csv", scores.shape, "| cytokines", len(cytokines))


if __name__ == "__main__":
    main()
