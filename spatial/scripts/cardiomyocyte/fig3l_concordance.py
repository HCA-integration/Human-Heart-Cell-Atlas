"""Fig 3 l: transcript-protein transmural concordance for explicit gene-selection rules.

The panel draws the combination named in config panels.fig3l_substrate and panels.fig3l_gene_set.
Gene sets (hhca.tl.dvp_gene_sets)
  curated17        the 22 hand-curated structural cardiomyocyte genes (config fig3l.structural_axis) with snRNA
                   CM_over_topNonCM >= 1 (17 genes)
  dvp_rule         DVP candidates as in Methods: raw p < 0.05 in cohort 1 and cohort 2 with the same slope sign
                   (cardiomyocytes), computed from trend_analysis_all.xlsx
  dvp_rule_cmexpr  dvp_rule genes expressed in snRNA ventricular cardiomyocytes (VentCM >= 0.05)
  dvp_rule_cmratio dvp_rule genes with CM_over_topNonCM >= 1
  dvp_rule_cmsafe  dvp_rule genes with cm_safe (both conditions)
Statistics per substrate and gene set (hhca.tl.slope_concordance): Spearman rho and p between transcript and
protein slope, sign agreement, n; genes with a zero or missing slope are dropped.

Inputs: the two DVP tables (xlsx), the specificity table of cm_specificity.py and, per substrate, the
concordance table of transmural_slopes.py with the names of its transcript and protein slope columns.
Outputs: PREFIX_stats.csv, PREFIX_genes.csv (slopes per substrate, set and gene) and PREFIX_genesets.json (the
sets, and how dvp_rule compares with the candidate sheets of trend_analysis_top_candidates.xlsx).

Usage: fig3l_concordance.py --dvp-all trend_analysis_all.xlsx --dvp-cand trend_analysis_top_candidates.xlsx
         --specificity snrna_cm_specificity_allgenes.csv --out-prefix PREFIX
         --substrate LABEL=TABLE:TRANSCRIPT_COL:PROTEIN_COL [--substrate ...]
"""

import argparse
import json

import numpy as np
import pandas as pd

import hhca
from hhca.config import load_config


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="Transcript-protein transmural concordance per gene set (Fig 3 l)")
    parser.add_argument("--dvp-all", required=True, help="DVP trend table of all proteins (xlsx)")
    parser.add_argument("--dvp-cand", required=True, help="DVP top-candidate sheets (xlsx)")
    parser.add_argument("--specificity", required=True, help="snrna_cm_specificity_allgenes.csv of cm_specificity.py")
    parser.add_argument(
        "--substrate",
        action="append",
        required=True,
        help="LABEL=TABLE:TRANSCRIPT_COL:PROTEIN_COL; may be given several times",
    )
    parser.add_argument("--out-prefix", required=True, help="prefix of the three output files")
    return parser.parse_args()


def main():
    """Build the gene sets, compute the concordance per substrate and set and write the tables."""
    args = parse_args()
    params = load_config("cardiomyocyte")["fig3l"]
    specificity = pd.read_csv(args.specificity).drop_duplicates("gene").set_index("gene")
    sets = hhca.tl.dvp_gene_sets(
        pd.read_excel(args.dvp_all, engine="openpyxl"), specificity, params["structural_axis"], params["p_max"]
    )
    workbook = pd.ExcelFile(args.dvp_cand, engine="openpyxl")
    candidates = set()
    for sheet in params["candidate_sheets"]:
        candidates |= set(workbook.parse(sheet)["Genes"].astype(str).str.split(";").str[0].str.upper())
    notes = {
        "sets": dict(sets),
        "curated17_matches_ratio_rule": sorted(sets["curated17"]) == sorted(params["axis17"]),
        "dvp_rule_n": len(sets["dvp_rule"]),
        "dvp_rule_equals_candidate_sheets": set(sets["dvp_rule"]) == candidates,
        "dvp_candidate_sheets_n": len(candidates),
        "curated17_failing_dvp_rule": [gene for gene in params["axis17"] if gene not in set(sets["dvp_rule"])],
    }

    stat_rows, gene_rows = [], []
    for substrate in args.substrate:
        label, rest = substrate.split("=", 1)
        table_path, transcript_col, protein_col = rest.rsplit(":", 2)
        slopes = pd.read_csv(table_path)
        slopes["gene"] = slopes["gene"].astype(str).str.upper()
        slopes = (
            slopes.drop_duplicates("gene")
            .set_index("gene")[[transcript_col, protein_col]]
            .rename(columns={transcript_col: "t", protein_col: "p"})
        )
        for name, genes in sets.items():
            in_table = slopes.reindex([gene for gene in genes if gene in slopes.index])
            concordance = hhca.tl.slope_concordance(in_table["t"], in_table["p"])
            stat_rows.append(dict(substrate=label, gene_set=name, n_set=len(genes), **concordance))
            for gene in genes:
                gene_rows.append(
                    {
                        "substrate": label,
                        "gene_set": name,
                        "gene": gene,
                        "transcript_slope": slopes["t"].get(gene, np.nan),
                        "protein_slope": slopes["p"].get(gene, np.nan),
                    }
                )
    statistics = pd.DataFrame(stat_rows)
    statistics.to_csv(f"{args.out_prefix}_stats.csv", index=False)
    pd.DataFrame(gene_rows).to_csv(f"{args.out_prefix}_genes.csv", index=False)
    with open(f"{args.out_prefix}_genesets.json", "w") as handle:
        json.dump(notes, handle, indent=2)
    pd.set_option("display.width", 200)
    print(statistics.to_string(index=False))
    print(json.dumps({key: value for key, value in notes.items() if key != "sets"}, indent=1))


if __name__ == "__main__":
    main()
