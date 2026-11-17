#!/usr/bin/env python
"""Build resources/atlas/cytokine_programs.json and cytokine_program_meta.csv from the huCIRA dictionary.

Rule (264 programmes; hhca.tl.cytokine_programmes, settings: config/atlas_hormones.yaml, cytokine_programs):
  1. curated dictionary = public huCIRA human cytokine dictionary rows of the 24 cytokines in
     hucira_celltype_match.yaml with adj_p_value < 0.05 and |log_fc| >= 0.25, including rows flagged
     well_biased (hucira 0.1.3's loader drops them, and also drops TGF-beta1 / IL-18 / C3a; the curated
     table is made from the raw download without either step: 553,312 rows);
  2. per atlas immune type and cytokine: rows of the matched huCIRA cell types with log_fc > 0.5 and
     adj_p_value < 0.05 ("up");
  3. per gene: n = number of matched cell types in which it is up, mean = mean up log_fc;
     rank by n, then mean (both descending, stable), keep the top 50.
Programmes with no up gene are written as empty lists in the JSON and omitted from the meta table.

Inputs: either the full dictionary as saved by hucira 0.1.3 load_human_cytokine_dict() (download of
https://cdn.parsebiosciences.com/gigalab/10m/DEGs.csv, index reset; 3.46 GB) with --dict, or an already
curated table with --curated; hucira_celltype_match.yaml.
Outputs: cytokine_programs.json, cytokine_program_meta.csv and, with --dict, human_cytokine_dict_curated.csv.

Usage: python build_cytokine_programs.py --dict human_cytokine_dict.csv \
           --match resources/atlas/hucira_celltype_match.yaml --outdir results/atlas/resources/cytokine
"""

import argparse
import json
import os

import pandas as pd
import yaml

import hhca
from hhca.config import load_config

COLUMNS = ["gene", "log_fc", "adj_p_value", "celltype", "cytokine"]


def curated_from_full(path, cytokines, adj_p_lt, abs_log_fc_ge, chunksize=2_000_000):
    """Significant rows of the chosen cytokines, read from the full dictionary in blocks."""
    parts = []
    for chunk in pd.read_csv(path, usecols=COLUMNS + ["well_biased"], chunksize=chunksize):
        keep = chunk.cytokine.isin(cytokines) & (chunk.adj_p_value < adj_p_lt) & (chunk.log_fc.abs() >= abs_log_fc_ge)
        parts.append(chunk.loc[keep])
    return pd.concat(parts, ignore_index=True)


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dict", help="full huCIRA dictionary CSV (as saved by hucira)")
    source.add_argument("--curated", help="curated dictionary CSV (gene, log_fc, adj_p_value, celltype, cytokine)")
    parser.add_argument("--match", required=True, help="hucira_celltype_match.yaml")
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def main():
    """Build the cytokine programmes and write them with their description table."""
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    params = load_config("atlas_hormones")["cytokine_programs"]
    with open(args.match) as handle:
        match = yaml.safe_load(handle)

    if args.dict:
        curated = curated_from_full(
            args.dict,
            match["cytokines"],
            params["curated_filter"]["adj_p_lt"],
            params["curated_filter"]["abs_log_fc_ge"],
        )
        curated[COLUMNS].to_csv(f"{args.outdir}/human_cytokine_dict_curated.csv", index=False)
        print(
            f"curated subset: {len(curated):,} rows, {curated.cytokine.nunique()} cytokines, "
            f"{int(curated.well_biased.sum())} well_biased rows kept"
        )
    else:
        curated = pd.read_csv(args.curated)

    out, meta = hhca.tl.cytokine_programmes(curated, match, params["up_log_fc"], params["up_adj_p"], params["top_n"])
    with open(f"{args.outdir}/cytokine_programs.json", "w") as handle:
        json.dump(out, handle)
    meta.to_csv(f"{args.outdir}/cytokine_program_meta.csv", index=False)
    n_programmes = sum(len(by_cytokine) for by_cytokine in out["programs"].values())
    n_nonempty = sum(1 for by_cytokine in out["programs"].values() for genes in by_cytokine.values() if genes)
    print(f"programmes: {n_nonempty} non-empty of {n_programmes} -> {args.outdir}")


if __name__ == "__main__":
    main()
