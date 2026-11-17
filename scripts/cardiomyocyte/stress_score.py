"""Fig 3 d: ventricular-cardiomyocyte stress score across the wall of donor H0054 (epi, myo, endo sections).

Per CMFLAG-clean vCM 2 um bin, mean over the gene set of log1p CP10k z-scored against the pooled H0054
reference (all three sections; hhca.tl.log_cp10k, hhca.tl.score_gene_sets). Each section is cut into 650 um tiles
ranked by their mean score (hhca.tl.rank_tiles; tiles with >= 4000 vCM bins) and one field per section is shown on
a 5 um grid (hhca.tl.field_grid). Two gene sets: `stress18` (the 18-gene stress score of the panel) and `stress6`.

Fields: --fields FILE.json ({"epi": [xlo, xhi, ylo, yhi], ...}) fixes the fields for every gene set (pipeline:
config fig3d.fields, the fields the rule picks with stress18); without it each gene set takes the tile at the
percentile fig3d.pick of its ranking. Sections and parameters: fig3d in config/cardiomyocyte.yaml.

Inputs: the 2 um h5ads of the three sections (config paths.h5ad_2um); the gene sets (json).
Outputs: OUT.npz (grids, fields), OUT_tiles.csv (tile ranking), OUT_fields.json (fields, section means, genes).

Usage: stress_score.py --genesets GENESETS.json --out OUT.npz [--fields FIELDS.json]
"""

import argparse
import json
import os

import anndata as ad
import numpy as np
import pandas as pd

import hhca
from hhca.config import load_config, section_path


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ventricular-cardiomyocyte stress score of the H0054 sections (Fig 3 d)"
    )
    parser.add_argument("--genesets", required=True, help='json {"stress18": [...], "stress6": [...]}')
    parser.add_argument("--out", required=True, help="output npz")
    parser.add_argument(
        "--fields", default="", help='json {"epi": [xlo, xhi, ylo, yhi], ...}; default: chosen by the tile rule'
    )
    return parser.parse_args()


def main():
    """Score the three sections, pick the fields and write the grids, the tile ranking and the summary."""
    args = parse_args()
    cfg = load_config("cardiomyocyte")
    params = cfg["fig3d"]
    field_um, grid_um = params["field_um"], params["grid_um"]
    with open(args.genesets) as handle:
        gene_sets = json.load(handle)
    score_genes = list(dict.fromkeys(gene for genes in gene_sets.values() for gene in genes))
    preset_fields = None
    if args.fields:
        with open(args.fields) as handle:
            preset_fields = json.load(handle)

    sections = {
        layer: ad.read_h5ad(section_path(cfg, "h5ad_2um", slide), backed="r")
        for layer, slide in params["sections"].items()
    }
    cm = hhca.tl.log_cp10k(sections, score_genes, cfg["cell_types"]["ventricular"], chunk=params["chunk"])
    hhca.tl.score_gene_sets(cm, gene_sets)

    arrays, tile_rows, fields, summary = {}, [], {}, {}
    for name in gene_sets:
        fields[name] = {}
        for layer in sections:
            section = cm[cm.obs["section"] == layer]
            ranking = hhca.tl.rank_tiles(section, name, field_um, params["min_tile_bins"])
            for _, tile in ranking.iterrows():
                tile_rows.append(
                    {
                        "gene_set": name,
                        "layer": layer,
                        "xlo": tile.xlo,
                        "ylo": tile.ylo,
                        "mean": tile["mean"],
                        "count": int(tile["count"]),
                    }
                )
            rule_field = hhca.tl.tile_at(ranking, params["pick"][layer], field_um)
            field = preset_fields[layer] if preset_fields else rule_field
            fields[name][layer] = {"used": field, "rule": rule_field}
            grid = hhca.tl.field_grid(section, name, field, field_um, grid_um)
            arrays[f"{name}_{layer}_grid"] = grid
            arrays[f"{name}_{layer}_ext"] = np.array(field)
            summary[f"{name}_{layer}"] = {
                "section_mean": float(section.obs[name].to_numpy().mean()),
                "n_vcm_clean": int(section.n_obs),
                "field_mean": float(np.nanmean(grid)),
            }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    base = args.out[:-4] if args.out.endswith(".npz") else args.out
    pd.DataFrame(tile_rows).to_csv(f"{base}_tiles.csv", index=False)
    present = list(cm.var_names)
    with open(f"{base}_fields.json", "w") as handle:
        json.dump(
            {
                "fields": fields,
                "summary": summary,
                "genes_present": present,
                "genes_missing": [gene for gene in score_genes if gene not in present],
                "gene_sets": gene_sets,
            },
            handle,
            indent=2,
        )
    print(json.dumps(summary, indent=1))
    print(json.dumps(fields, indent=1))


if __name__ == "__main__":
    main()
