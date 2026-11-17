"""Cytokine-programme scores by rejection type, every programme (Fig 6 i, ED9 b-c).

The statistics are over all 18 programmes; --drop only removes programmes from the `selected` column. Per-biopsy
rejection type: config labels.rejection_rule (combine; 37/39/12/7 biopsies).
Statistics (hhca.tl.programmes_by_rejection): per programme Kruskal-Wallis over the 4 groups, BH over the 18
programmes; within a programme all 6 pairwise Mann-Whitney tests, BH over those 6. Drawn by figures/Fig6/panel_i.py
and figures/ED9/panel_bc.py.

Inputs:
    GEO clinical metadata table (config inputs.metadata)
    tables/per_biopsy_scores_immune_ATLAS.csv (core_features.py)

Outputs (<out>/):
    fig6_cytokine_panels_stats.csv, fig6_cytokine_biopsy_values.csv, fig6_cytokine_panels_run_info.txt

Usage: python scripts/xenium/downstream/cytokine_by_rejection.py [--scores CSV] [--out DIR]
"""

import argparse
import os

import pandas as pd

import downstream_common as common
import hhca
from hhca.tl._rejection import PAIRS, REJECTION_TYPES, SHORT_TYPES, programme_name

ALPHA = 0.05
RUN_INFO = []


def note(*message):
    """Print a line and keep it for fig6_cytokine_panels_run_info.txt."""
    line = " ".join(str(part) for part in message)
    RUN_INFO.append(line)
    print(line, flush=True)


def parse_args(cfg):
    """Command-line arguments; the defaults come from config/xenium_downstream.yaml."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--meta", default=cfg["inputs"]["metadata"], help="GEO clinical metadata table")
    parser.add_argument("--scores", help="tables/per_biopsy_scores_immune_ATLAS.csv")
    parser.add_argument(
        "--drop", default="IL-1-beta", help="programmes (comma-separated) left out of the `selected` column"
    )
    parser.add_argument("--out", help="output folder (default: under config outputs.root)")
    return parser.parse_args()


def biopsy_scores(meta_path, scores_path, rejection_rule):
    """Biopsies with a rejection type and all programme scores (mean over cores), and the score columns."""
    core_scores = pd.read_csv(scores_path).set_index("Sample")
    _, biopsies, cyto = hhca.tl.biopsy_cohort(common.read_meta(meta_path), core_scores, rejection_rule=rejection_rule)
    note(
        f"COHORT: {len(biopsies)} biopsies / {biopsies.patient.nunique()} patients "
        f"({int((biopsies.ncore == 2).sum())} with 2 cores, {int((biopsies.ncore == 1).sum())} with 1; "
        f"{int(biopsies.ncore.sum())} cores total)"
    )
    note(f"  core-negative biopsies excluded: {int(biopsies.core_neg.sum())}")
    labelled = biopsies[~biopsies.core_neg].dropna(subset=["rej"] + cyto)
    sizes = {g: int((labelled.rej == g).sum()) for g in REJECTION_TYPES}
    note(f"  n={len(labelled)} biopsies / {labelled.patient.nunique()} patients | {sizes}")
    note(f"  rejection rule: {rejection_rule}")
    note(f"  {len(cyto)} cytokine programmes tested")
    return labelled, cyto


def select_programmes(tests, drop):
    """Mark the programmes that are not dropped (`selected`) and log their tests, most significant first."""
    by_column = tests.set_index("column")
    kp, kq = by_column.kruskal_p.to_dict(), by_column.kruskal_q_BH_over_18.to_dict()
    selected = sorted(
        [k for k in tests.column if programme_name(k) not in drop and k.replace("score_", "") not in drop],
        key=lambda k: (kp[k], kq[k]),
    )
    note(
        f"\nplotting all {len(selected)}/{len(tests)} programmes -> "
        + ", ".join(f"{programme_name(k)} (q={kq[k]:.3g})" for k in selected)
    )
    for k in selected:
        pair_q = [by_column.loc[k, f"q_{SHORT_TYPES[i]}_vs_{SHORT_TYPES[j]}"] for i, j in PAIRS]
        hits = [f"{SHORT_TYPES[i]} vs {SHORT_TYPES[j]} q={q:.3g}" for (i, j), q in zip(PAIRS, pair_q) if q < ALPHA]
        note(f"   {programme_name(k)}: Kruskal p={kp[k]:.3g} q={kq[k]:.3g} | " + ("; ".join(hits) or "no pairwise hit"))
    tests.insert(4, "selected", tests.column.isin(selected))
    return tests


def main():
    cfg = common.load_config()
    args = parse_args(cfg)
    tables = common.out_dir(cfg, "tables")
    rejection_rule = cfg["labels"]["rejection_rule"]
    scores_path = args.scores or f"{tables}/per_biopsy_scores_immune_ATLAS.csv"
    out = args.out or common.out_dir(cfg, "stats")
    os.makedirs(out, exist_ok=True)
    drop = [name for name in args.drop.split(",") if name]

    labelled, cyto = biopsy_scores(args.meta, scores_path, rejection_rule)
    tests = select_programmes(hhca.tl.programmes_by_rejection(labelled, cyto), drop)
    tests.sort_values("kruskal_q_BH_over_18").to_csv(f"{out}/fig6_cytokine_panels_stats.csv", index=False)
    labelled[["patient", "rej"] + cyto].to_csv(f"{out}/fig6_cytokine_biopsy_values.csv")
    note(f"\nwrote {out}/fig6_cytokine_panels_stats.csv ({len(tests)} cytokines)")
    with open(f"{out}/fig6_cytokine_panels_run_info.txt", "w") as handle:
        handle.write("\n".join(RUN_INFO))


if __name__ == "__main__":
    main()
