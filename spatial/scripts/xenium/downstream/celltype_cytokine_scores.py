"""Per-core x immune-cell-type cytokine-programme scores (input of the Fig 6 j tests).

  * counts CP10k + log1p (float32),
  * per cell: mean log1p expression over the programme's panel genes (program_panel_genes.csv; no control-gene
    subtraction, so the scores are all positive and not on the sc.tl.score_genes scale),
  * per core x cell type: mean over cells; groups with >= 10 cells kept (params.celltype_cytokine.min_cells),
  * programmes with >= 5 panel genes (17 of 18; IL-10 has 4) (params.celltype_cytokine.min_panel_genes),
  * cell types: lymphoid Level-4 classes + Macrophages, Monocytes, Dendritic cells, B cells,
    Plasma cells, Mast cells (12 classes, no Treg; config classes.immune),
  * row order = first occurrence of (core, cell type) in cell order.
The scoring is hhca.tl.programme_means.

Inputs:
    labels/xenium_atlas_labels.csv, labels/xenium_lymphoid8_annotation.csv (label_tables.py)
    counts.mtx, genes.txt, barcodes.txt, obs.csv, cells_geom.csv of the extraction folder (config inputs.extracted)
    shared/program_panel_genes.csv (program_genes.py)

Outputs:
    <out>/cytokine_celltype_scores_percore.csv (programmes, Sample, ct, n_cells), read by fig6_stats.py and
    fig6j_all_tests.py

Usage: python scripts/xenium/downstream/celltype_cytokine_scores.py
"""

import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc

import downstream_common as common
import hhca


def parse_args(cfg):
    """Command-line arguments; the defaults come from config/xenium_downstream.yaml."""
    params = cfg["params"]["celltype_cytokine"]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    common.add_label_arguments(parser)
    parser.add_argument(
        "--programs",
        default=os.path.join(common.out_dir(cfg, "shared"), "program_panel_genes.csv"),
        help="programme gene table of program_genes.py",
    )
    parser.add_argument(
        "--min-cells", type=int, default=params["min_cells"], help="smallest core x cell-type group kept"
    )
    parser.add_argument(
        "--min-genes", type=int, default=params["min_panel_genes"], help="fewest panel genes per programme"
    )
    parser.add_argument("--out", help="output folder (default: under config outputs.root)")
    return parser.parse_args()


def main():
    cfg = common.load_config()
    args = parse_args(cfg)
    out = args.out or common.out_dir(cfg, "tables")

    adata = common.load_cohort(cfg, args, counts=True)
    common.classify(adata, cfg, "immune")
    adata.X = adata.X.astype(np.float32)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    programmes = pd.read_csv(args.programs)
    programmes = programmes[programmes.n_on_panel >= args.min_genes]
    table = hhca.tl.programme_means(adata, hhca.tl.gene_sets(programmes), ("Sample", "ct"), args.min_cells)
    table.to_csv(os.path.join(out, "cytokine_celltype_scores_percore.csv"), index=False)
    print(
        f"programmes {len(programmes)} ({', '.join(programmes.cytokine)}) | groups >= {args.min_cells} cells: "
        f"{len(table)} | cell types {table.ct.nunique()}"
    )
    print(table.ct.value_counts().to_string())
    print("wrote", os.path.join(out, "cytokine_celltype_scores_percore.csv"))


if __name__ == "__main__":
    main()
