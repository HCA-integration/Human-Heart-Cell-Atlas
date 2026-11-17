#!/usr/bin/env python
"""Per-donor huCIRA cytokine-response programme scores on the immune cell types of the HHCA atlas.

Programmes are the cell-type-matched huCIRA response gene sets of cytokine_programs.json (the dictionary is
PBMC-derived, so only immune types are scored). Scoring follows donor_geneset_scores.py: per donor and cell type
the mean log1p CP10k of each gene, then the mean over the programme genes present in the atlas. Only the union
of the programme genes is streamed (hhca.tl.donor_pseudobulk, hhca.tl.programme_scores). Two contrasts on adult
donors (hhca.tl.programme_effects):
  sex: OLS score ~ sex + age (p_sens adds tissue and assay), Hedges g, Mann-Whitney U, BH-FDR
  age: OLS score ~ age + sex (p_sens adds tissue and assay), Spearman rho, BH-FDR
Cell types and thresholds: config/config.yaml (cytokine, donors).

Inputs: atlas h5ad (layers/counts), cytokine_programs.json.
Outputs: per_donor_cytokine_scores.npz (donor x cell type x cytokine), sex_cytokine_effects.csv,
age_cytokine_effects.csv.

Usage: python donor_cytokine_scores.py --atlas ATLAS.h5ad --programs resources/atlas/cytokine_programs.json \
           --outdir results/atlas/cytokine
"""

import argparse
import json
import os

import hhca
from hhca.config import load_config
from hhca.donors import write_programme_scores
from hhca.log import get_logger


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--atlas", required=True, help="HHCA atlas h5ad")
    parser.add_argument("--programs", required=True, help="cytokine_programs.json")
    parser.add_argument("--outdir", required=True, help="output directory")
    return parser.parse_args()


def main():
    """Stream the atlas, write the per-donor programme scores and the sex and age effect tables."""
    args = parse_args()
    log = get_logger()
    cfg = load_config("config")
    params = cfg["cytokine"]
    os.makedirs(args.outdir, exist_ok=True)
    with open(args.programs) as handle:
        resource = json.load(handle)
    programmes = resource["programs"]
    genes = sorted({g for ct in programmes for cy in programmes[ct] for g in programmes[ct][cy]})

    donors = hhca.tl.donor_table(args.atlas)
    pseudobulk = hhca.tl.donor_pseudobulk(args.atlas, genes, params["level"], cfg["atlas"]["stream_chunk"], log)
    scores = hhca.tl.programme_scores(pseudobulk, programmes, params["cell_types"], params["min_programme_genes"])
    write_programme_scores(f"{args.outdir}/per_donor_cytokine_scores.npz", donors, scores)
    log("saved per_donor_cytokine_scores.npz")

    sex_table, age_table = hhca.tl.programme_effects(
        pseudobulk,
        donors,
        programmes,
        params["cell_types"],
        nearest_myeloid=resource["nearest_myeloid"],
        min_cells=params["min_cells"],
        min_donors=params["min_donors"],
        min_genes=params["min_programme_genes"],
        adult_age=cfg["donors"]["adult_age"],
    )
    sex_table.to_csv(f"{args.outdir}/sex_cytokine_effects.csv", index=False)
    age_table.to_csv(f"{args.outdir}/age_cytokine_effects.csv", index=False)
    for name, table in (("sex", sex_table), ("age", age_table)):
        log(
            f"{name}: {len(table)} tests, {(table.q < 0.05).sum()} with q < 0.05, "
            f"{((table.q < 0.05) & (table.p_sens < 0.05)).sum()} also with p_sens < 0.05"
        )
    print("\nsex, strongest effects")
    print(sex_table.head(12).to_string(index=False))
    print("\nage, strongest effects")
    print(age_table.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
