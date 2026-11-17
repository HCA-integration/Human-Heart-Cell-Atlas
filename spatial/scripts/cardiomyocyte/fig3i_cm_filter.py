"""Fig 3 i: cardiomyocyte-specificity filters for the transmural gene modules, as explicit atlas rules.

`cm_safe` (cm_specificity.py) = ventricular-CM mean >= 0.05 and a cardiomyocyte type is the top Level-3 expresser.
Each rule below is applied to the 3-donor selection of select_transmural_genes.py (hhca.tl.module_rules); the
panel draws the rule named in config panels.fig3i_cm_rule.
  hand_drop            selected top 6 / 4 without the stromal genes MMP2 and SPARC (config transmural.hand_drop)
  cm_safe_post         selected top 6 / 4, genes with cm_safe kept            (filter after selection)
  cm_ratio_post        selected top 6 / 4, genes with CM_over_topNonCM >= 1 kept (no expression floor)
  cm_safe_pre          top 6 / 4 among in-universe, sign-consistent, cm_safe genes (filter before selection)
  selected_unfiltered  selected top 6 / 4 as they are

Inputs: module_marker_search_withAH10.csv and markersearch_genesets_withAH10.json in --tab, and the specificity
table.
Outputs: PREFIX_genes.csv (one row per rule and gene) and PREFIX_genesets.json.

Usage: fig3i_cm_filter.py --tab DIR --specificity snrna_cm_specificity_allgenes.csv --out-prefix PREFIX
"""

import argparse
import json

import pandas as pd

import hhca
from hhca.config import load_config


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="Cardiomyocyte-specificity rules for the Fig 3 i gene modules")
    parser.add_argument(
        "--tab",
        required=True,
        help="folder with module_marker_search_withAH10.csv and markersearch_genesets_withAH10.json",
    )
    parser.add_argument("--specificity", required=True, help="snrna_cm_specificity_allgenes.csv of cm_specificity.py")
    parser.add_argument("--out-prefix", required=True, help="prefix of the two output files")
    return parser.parse_args()


def gene_rows(sets, ranking):
    """One row per rule, direction and gene with its rank and specificity values."""
    by_gene = ranking.set_index("gene")
    rows = []
    for rule, modules in sets.items():
        for direction, genes in modules.items():
            for gene in genes:
                r = by_gene.loc[gene]
                rows.append(
                    {
                        "rule": rule,
                        "direction": direction,
                        "gene": gene,
                        "rank": int(r.rank_endo_to_epi_3donor),
                        "consensus_rho_3donor": r.consensus_rho_3donor,
                        "VentCM": r.VentCM,
                        "top_nonCM": r.top_nonCM,
                        "CM_over_topNonCM": r.CM_over_topNonCM,
                        "cm_safe": bool(r.cm_safe),
                        "cm_call": r.cm_call,
                    }
                )
    return pd.DataFrame(rows)


def main():
    """Join the specificity calls to the ranked genes, apply the rules and write the gene sets."""
    args = parse_args()
    with open(f"{args.tab}/markersearch_genesets_withAH10.json") as handle:
        selection = json.load(handle)
    ranking = pd.read_csv(f"{args.tab}/module_marker_search_withAH10.csv")
    specificity = pd.read_csv(args.specificity).drop_duplicates("gene").set_index("gene")

    ranking = ranking.merge(
        specificity[["VentCM", "CM_max", "top_nonCM", "CM_over_topNonCM", "cm_safe", "cm_call"]],
        left_on="gene",
        right_index=True,
        how="left",
    )
    ranking["cm_safe"] = ranking["cm_safe"].fillna(False).astype(bool)

    sets = hhca.tl.module_rules(ranking, selection, load_config("cardiomyocyte")["transmural"]["hand_drop"])
    gene_rows(sets, ranking).to_csv(f"{args.out_prefix}_genes.csv", index=False)
    with open(f"{args.out_prefix}_genesets.json", "w") as handle:
        json.dump(sets, handle, indent=2)
    for rule, modules in sets.items():
        print(f"{rule:20s} endo {modules['endo']}  epi {modules['epi']}")
    eligible = ranking.in_universe & ranking.sign_consistent_3donor & ranking.cm_safe
    print("eligible genes under cm_safe_pre:", int(eligible.sum()), "of", int(ranking.in_universe.sum()), "in-universe")


if __name__ == "__main__":
    main()
