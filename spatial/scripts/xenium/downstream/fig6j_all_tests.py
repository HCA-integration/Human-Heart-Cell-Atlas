"""Every cell type x programme test of resolved vs persistent rejection (the family of the 6 j test).

The 6 j test of fig6_stats.py is one pair (IL-4 in pooled CD4 T cells) out of the per-core x cell-type table.
This runs the same test for every cell type of the table (and the pooled CD4 compartment of 6 j) x every programme,
on the 6 j cohort (pre-treatment biopsies of patients with a rejection episode and a recorded resolution_broad), and
corrects over the whole family (BH); hhca.tl.outcome_tests.
Per cell type: per-core value -> biopsy mean over its cores; pooled CD4: cell-weighted per core, as 6 j.
Test: two-sided Mann-Whitney persistence vs resolved; AUC = U / (n1 n2); tests need >= 3 biopsies per group
(config params.outcome_min_biopsies).

Inputs:
    GEO clinical metadata table (config inputs.metadata)
    tables/cytokine_celltype_scores_percore.csv (celltype_cytokine_scores.py)

Outputs (<out>/):
    fig6j_all_tests.csv, fig6j_all_tests_summary.json

Usage: python scripts/xenium/downstream/fig6j_all_tests.py [--celltype-scores CSV]
"""

import argparse
import json

import pandas as pd

import downstream_common as common
import hhca

POOLED_NAME = "CD4 T (pooled, 6 j)"


def parse_args(cfg):
    """Command-line arguments; the defaults come from config/xenium_downstream.yaml."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--meta", default=cfg["inputs"]["metadata"], help="GEO clinical metadata table")
    parser.add_argument("--celltype-scores", help="tables/cytokine_celltype_scores_percore.csv")
    parser.add_argument(
        "--min-n",
        type=int,
        default=cfg["params"]["outcome_min_biopsies"],
        help="fewest biopsies per outcome group for a test",
    )
    parser.add_argument("--out", help="output folder (default: under config outputs.root)")
    return parser.parse_args()


def main():
    cfg = common.load_config()
    args = parse_args(cfg)
    tables = common.out_dir(cfg, "tables")
    out = args.out or common.out_dir(cfg, "stats")

    md = common.read_meta(args.meta).set_index("Sample")
    biopsies = hhca.tl.outcome_cohort(md)

    celltype_scores = pd.read_csv(args.celltype_scores or f"{tables}/cytokine_celltype_scores_percore.csv")
    programmes = [column for column in celltype_scores.columns if column not in ("Sample", "ct", "n_cells")]
    pooled = hhca.tl.pool_cell_types(celltype_scores, cfg["classes"]["cd4_pool"], POOLED_NAME)
    celltype_scores["bxid"] = celltype_scores.Sample.map(md.bxid)
    pooled["bxid"] = pooled.Sample.map(md.bxid)
    scores = pd.concat([celltype_scores[["Sample", "ct", "bxid"] + programmes], pooled], ignore_index=True)

    tests = hhca.tl.outcome_tests(scores, biopsies, args.min_n, pooled=POOLED_NAME)
    tests.to_csv(f"{out}/fig6j_all_tests.csv", index=False)
    per_celltype = tests[tests.ct != POOLED_NAME]
    il4_pooled = tests[(tests.ct == POOLED_NAME) & (tests.programme == "IL-4")]
    summary = {
        "n_tests": len(tests),
        "n_celltypes": int(tests.ct.nunique()),
        "n_programmes": len(programmes),
        "n_tests_per_celltype_family": len(per_celltype),
        "n_p_lt_05": int((tests.p < 0.05).sum()),
        "n_q_lt_05": int((tests.q_BH_all < 0.05).sum()),
        "expected_false_p_lt_05": 0.05 * len(tests),
        "min_q": float(tests.q_BH_all.min()),
        "il4_cd4_pooled": il4_pooled.iloc[0].to_dict() if len(il4_pooled) else None,
        "top5": tests.head(5)[["ct", "programme", "p", "auc", "q_BH_all"]].to_dict("records"),
    }
    with open(f"{out}/fig6j_all_tests_summary.json", "w") as handle:
        json.dump(summary, handle, indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
