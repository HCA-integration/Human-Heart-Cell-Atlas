"""Fig 6 biopsy cohort and statistics: 6 e (biopsy level), f, g, h, i (IL-1beta), j (IL-4 in CD4 T cells), l.

The panels are drawn by figures/Fig6/panel_*.py from the tables written here (6 j has no panel); the biopsy-level
6 d counts are logged. The cohort and the tests are functions of hhca.tl (src/hhca/tl/_rejection.py).

Cohort: biopsy = patient + biopsy day; value = mean of its 1-2 TMA cores. Grade = GEO clinical biopsy grade
(patient_cellular_grading; config labels.grade_rule). Rejection type = per-core biopsy_rejection_type combined over
cores: "mixed" when the cores show cellular and antibody rejection (config labels.rejection_rule). Core-negative
biopsies (no_rejection but pre-treatment) are excluded from the rejection-labelled panels.

Inputs:
    GEO clinical metadata table (config inputs.metadata)
    tables/geom_features_L4.csv (core_geometry.py), tables/per_biopsy_scores_immune_ATLAS.csv (core_features.py),
    tables/atg_drug_target_percore.csv (drug_targets.py),
    tables/cytokine_celltype_scores_percore.csv (celltype_cytokine_scores.py)

Outputs (<out>/):
    fig6_biopsies.csv, fig6_fg_pairs.csv, fig6_h_matrix.csv, fig6_h_tests.csv, fig6_i_il1b.csv, fig6_j_values.csv,
    fig6_stats.json, fig6_run_info.txt

Usage: python scripts/xenium/downstream/fig6_stats.py [--geom CSV --scores CSV --atg CSV --celltype-scores CSV]
           [--perm N] [--out DIR]
"""

import argparse
import json
import os

import pandas as pd

import downstream_common as common
import hhca
from hhca.stats import stars
from hhca.tl._rejection import REJECTING, REJECTION_TYPES

RUN_INFO = []


def note(*message):
    """Print a line and keep it for fig6_run_info.txt."""
    line = " ".join(str(part) for part in message)
    RUN_INFO.append(line)
    print(line, flush=True)


def parse_args(cfg):
    """Command-line arguments; the defaults come from config/xenium_downstream.yaml."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--meta", default=cfg["inputs"]["metadata"], help="GEO clinical metadata table")
    parser.add_argument("--geom", help="tables/geom_features_L4.csv")
    parser.add_argument("--scores", help="tables/per_biopsy_scores_immune_ATLAS.csv")
    parser.add_argument("--atg", help="tables/atg_drug_target_percore.csv")
    parser.add_argument("--celltype-scores", help="tables/cytokine_celltype_scores_percore.csv")
    parser.add_argument(
        "--perm", type=int, default=cfg["params"]["permutations"], help="number of patient-level permutations"
    )
    parser.add_argument("--out", help="output folder (default: under config outputs.root)")
    return parser.parse_args()


def describe_cohort(cores, biopsies, rejection_rule):
    """Log the cohort and return its counts, including how the grade and rejection rules differ."""
    note(
        f"COHORT: {len(biopsies)} biopsies / {biopsies.patient.nunique()} patients "
        f"({int((biopsies.ncore == 2).sum())} with 2 cores, {int((biopsies.ncore == 1).sum())} with 1; "
        f"{int(biopsies.ncore.sum())} cores total)"
    )
    note(
        f"  core-negative excluded from rejection-labelled panels: {int(biopsies.core_neg.sum())} "
        f"(all pre-treatment, episode = {sorted(set(biopsies.loc[biopsies.core_neg, 'epi']))})"
    )
    note(
        "  completeness: "
        + ", ".join(
            f"{f}={int(biopsies[f].notna().sum())}/{len(biopsies)}"
            for f in ["grade", "pamr", "rej", "res", "immf", "ff", "lf", "tf", "atg"]
        )
    )
    summary = hhca.tl.cohort_summary(cores, biopsies)
    note(
        f"  rejection rule = {rejection_rule}; first vs combine differ in "
        f"{len(summary['rejection_rule_first_vs_combine'])} biopsies: {summary['rejection_rule_first_vs_combine']}"
    )
    note(
        f"  grade rule clinical vs max per-core grade: differ in {summary['grade_rule_differs']} biopsies "
        f"(clinical higher {summary['grade_clinical_gt_max_core']}, "
        f"lower {summary['grade_clinical_lt_max_core']}; "
        f"no per-core grade {summary['grade_max_core_missing']}); "
        f"biopsies with two different rejecting core types (worst() order-dependent): "
        f"{len(summary['biopsies_two_rejecting_types'])}"
    )
    return summary


def describe_pretreatment(biopsies):
    """Log the biopsy-level counts behind 6 d (the 6 d panel itself is per core, fig6d_overview.py)."""
    pre = biopsies[(biopsies.epi.isin(REJECTING)) & (biopsies.timing == "pre")]
    note(
        f"\n6d [pre-treatment biopsies, grouped by the treated rejection episode]: n={len(pre)} biopsies "
        f"{pre.epi.value_counts().to_dict()} | resolution {pre.res.value_counts(dropna=False).to_dict()}"
    )


def main():
    cfg = common.load_config()
    args = parse_args(cfg)
    grade_rule, rejection_rule = cfg["labels"]["grade_rule"], cfg["labels"]["rejection_rule"]
    tables = common.out_dir(cfg, "tables")
    geom_path = args.geom or f"{tables}/geom_features_L4.csv"
    scores_path = args.scores or f"{tables}/per_biopsy_scores_immune_ATLAS.csv"
    atg_path = args.atg or f"{tables}/atg_drug_target_percore.csv"
    celltype_scores_path = args.celltype_scores or f"{tables}/cytokine_celltype_scores_percore.csv"
    out = args.out or common.out_dir(cfg, "stats")
    os.makedirs(out, exist_ok=True)

    results = {
        "grade_rule": grade_rule,
        "rejection_rule": rejection_rule,
        "inputs": {
            "geom": geom_path,
            "scores": scores_path,
            "atg": atg_path,
            "celltype_scores": celltype_scores_path,
        },
    }

    geometry = pd.read_csv(geom_path).set_index("Sample")
    atg = pd.read_csv(atg_path).set_index("Sample")
    core_scores = pd.read_csv(scores_path).set_index("Sample")
    cores, biopsies, cyto = hhca.tl.biopsy_cohort(
        common.read_meta(args.meta), core_scores, geometry, atg, grade_rule, rejection_rule
    )

    results["cohort"] = describe_cohort(cores, biopsies, rejection_rule)
    labelled = biopsies[~biopsies.core_neg]
    describe_pretreatment(biopsies)

    # 6 e: grade against foci
    test = hhca.tl.foci_by_grade(labelled, args.perm)
    note(
        f"\n6e [all biopsies, need grade + foci]: n={test['n']} biopsies / {test['patients']} patients "
        f"| rho={test['rho']:.2f} p={test['p']:.3g} | patient-clustered permutation p={test['p_perm']:.4f}"
    )
    results["e_biopsy"] = test

    # 6 f / 6 g: before and after treatment
    pairs, tests = hhca.tl.pre_post_pairs(biopsies)
    note(f"\n6f/6g [patients with a pre AND a post biopsy]: n={len(pairs)} patients")
    pairs.to_csv(f"{out}/fig6_fg_pairs.csv")
    for key, test in tests.items():
        tag, group = key.split("_")
        note(
            f"   6{tag} {group}: n={test['n']}, median {test['median_pre']:.3f} -> {test['median_post']:.3f}, "
            f"p={test['p']:.3g}"
        )
    results["fg"] = tests

    # 6 h: programmes of each rejection type against no rejection
    scored = labelled.dropna(subset=["rej"] + cyto)
    note(
        f"\n6h/6i [all biopsies, need rejection type + cytokine scores]: n={len(scored)} "
        f"{scored.rej.value_counts().to_dict()}"
    )
    zscores, tests = hhca.tl.programmes_vs_no_rejection(scored, cyto)
    note(f"   6h: {int((tests.q < 0.05).sum())}/{len(tests)} cells significant vs no rejection")
    zscores.to_csv(f"{out}/fig6_h_matrix.csv")
    tests.to_csv(f"{out}/fig6_h_tests.csv", index=False)
    results["h"] = {
        "n": len(scored),
        "groups": {g: int((scored.rej == g).sum()) for g in REJECTION_TYPES},
        "tests": len(tests),
        "significant": int((tests.q < 0.05).sum()),
        "significant_by_group": {
            g: sorted(tests[(tests.group == g) & (tests.q < 0.05)].programme.str.replace("score_", "").tolist())
            for g in REJECTING
        },
    }
    for g in REJECTING:
        note(f"   6h {g}: {results['h']['significant_by_group'][g]}")

    # 6 i: one programme across the rejection types
    test = hhca.tl.kruskal_permutation(scored, f"score_{cfg['figures']['fig6i_programme']}", args.perm)
    note(
        f"   6i: n {test['n']} / {scored.patient.nunique()} patients "
        f"| Kruskal p={test['kruskal_p']:.3g} | patient-clustered permutation p={test['p_perm']:.4f}"
    )
    for pair, q in test["pairs"].items():
        if q < 0.05:
            note(f"      {pair.replace('|', ' vs ')}: q={q:.3g} " + stars(q))
    scored[["patient", "rej"] + cyto].to_csv(f"{out}/fig6_i_il1b.csv")
    results["i_il1b"] = test

    # 6 j: IL-4 programme in the pooled CD4 T compartment, persistence against resolved
    celltype_scores = pd.read_csv(celltype_scores_path)
    pool = cfg["classes"]["cd4_pool"]
    n_resting, n_prolif = (int((celltype_scores.ct == name).sum()) for name in pool)
    note(
        f"\n6j: CD4 compartment pooled (resting + proliferating), cell-weighted; "
        f"cores contributing: {n_resting} resting, {n_prolif} proliferating"
    )
    pooled = hhca.tl.pool_cell_types(celltype_scores, pool, "CD4 T (pooled)")
    tested, test = hhca.tl.outcome_test(biopsies, cores, pooled, "IL-4")
    note(
        f"\n6j [pre-treatment + rejection diagnosis + CD4 T IL-4 score]: n={len(tested)} "
        f"{tested.res.value_counts().to_dict()}"
    )
    note(f"   MWU p={test['p']:.3g} AUC={test['auc']:.2f}")
    tested[["patient", "res", "epi", "il4"]].to_csv(f"{out}/fig6_j_values.csv")
    results["j"] = {**test, "cores_resting": n_resting, "cores_prolif": n_prolif}

    # 6 l: ATG drug-target score against the immune and the T-cell fraction
    tests = hhca.tl.score_correlations(biopsies, "atg", ("immf", "tf"))
    results["l"] = {}
    for column, tag in [("immf", "l"), ("tf", "l2")]:
        test = tests[column]
        note(f"6{tag} [all biopsies]: n={test['n']} rho={test['rho']:.2f} p={test['p']:.3g}")
        results["l"][tag] = test

    biopsies.to_csv(f"{out}/fig6_biopsies.csv")
    with open(f"{out}/fig6_stats.json", "w") as handle:
        json.dump(results, handle, indent=1, default=float)
    with open(f"{out}/fig6_run_info.txt", "w") as handle:
        handle.write("\n".join(RUN_INFO))
    note(f"\nwrote {out}/fig6_*")


if __name__ == "__main__":
    main()
