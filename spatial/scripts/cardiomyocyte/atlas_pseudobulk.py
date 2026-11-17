"""Atlas Level-2 / Level-3 pseudobulk: mean expression and detection fraction per cell type.

The `normcounts` layer of the HHCA atlas h5ad is read in row blocks (the atlas is 130 GB; nothing per cell is
kept). Mean = sum of normcounts over the type's nuclei / number of nuclei; detection = fraction of the type's
nuclei with normcounts > 0 (hhca.tl.type_means).

Inputs: atlas h5ad (1,459,368 nuclei x 36,601 genes; obs `Final Level {2,3}_harmonized`).
Outputs: in --out-dir, {mean_normcounts,detection}_L{2,3}.parquet (genes in atlas order x types sorted by name)
and ncells_L{2,3}.csv.

Usage: atlas_pseudobulk.py --atlas final_cleaned_hhca_annotated.h5ad --out-dir DIR [--layer normcounts]
       [--chunk 20000]
"""

import argparse
import os

import hhca
from hhca.config import load_config
from hhca.log import get_logger


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="Per-type mean and detection fraction from the atlas h5ad")
    parser.add_argument("--atlas", required=True, help="HHCA atlas h5ad")
    parser.add_argument("--out-dir", required=True, help="output folder")
    parser.add_argument("--layer", default="normcounts", help="layer to average")
    parser.add_argument("--chunk", type=int, default=20000, help="rows read per chunk")
    return parser.parse_args()


def main():
    """Average the layer per cell type and write the tables."""
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log = get_logger()
    levels = load_config("cardiomyocyte")["atlas"]["levels"]
    for level, tables in hhca.tl.type_means(args.atlas, levels, args.layer, args.chunk, log).items():
        tag = "L" + level.split("Level ")[1][0]
        tables["mean"].to_parquet(f"{args.out_dir}/mean_{args.layer}_{tag}.parquet")
        tables["detection"].to_parquet(f"{args.out_dir}/detection_{tag}.parquet")
        tables["n_cells"].to_csv(f"{args.out_dir}/ncells_{tag}.csv")
        log(f"wrote {tag}: {len(tables['n_cells'])} types; ncells {tables['n_cells'].to_dict()}")
    log("done", args.out_dir)


if __name__ == "__main__":
    main()
