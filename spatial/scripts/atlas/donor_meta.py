"""Per-donor metadata table (donor, sex, age, tissue, ...) from the per-donor gene-set score file.

The table supplies donor sex and age to hormone_sexdiff_significance.py (Fig 2 h-i). Sex = mode over the
donor's nuclei, age = median of the development-stage midpoints (as stored by donor_geneset_scores.py).

Inputs: per_donor_celltype_scores.npz.
Outputs: donor_meta.csv.

Usage: python donor_meta.py --npz results/atlas/per_donor_celltype_scores.npz --out results/atlas/donor_meta.csv
"""

import argparse
import os

import numpy as np
import pandas as pd


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--npz", required=True, help="per-donor score file of donor_geneset_scores.py")
    parser.add_argument("--out", required=True, help="metadata csv to write")
    return parser.parse_args()


def main():
    """Write the per-donor metadata table."""
    args = parse_args()
    scores = np.load(args.npz, allow_pickle=True)
    table = pd.DataFrame(
        {
            "donor": list(scores["donors"]),
            "sex": [str(x) for x in scores["donor_sex"]],
            "age": [float(x) for x in scores["donor_age"]],
            "tissue": [str(x) for x in scores["donor_tissue"]],
            "tissue_frac": [float(x) for x in scores["donor_tissue_frac"]],
            "assay": [str(x) for x in scores["donor_assay"]],
            "suspension": [str(x) for x in scores["donor_suspension"]],
            "disease": [str(x) for x in scores["donor_disease"]],
            "ncells": [int(x) for x in scores["donor_ncells"]],
        }
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"{len(table)} donors -> {args.out}")


if __name__ == "__main__":
    main()
