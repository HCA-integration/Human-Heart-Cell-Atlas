"""The transplant-rejection cohort: biopsies from their cores, and the tests by rejection type, cellular grade and
treatment outcome (Fig 6 d-l, ED9).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

REJECTION_TYPES = ["no_rejection", "cellular_rejection", "antibody_rejection", "mixed_rejection"]
REJECTING = REJECTION_TYPES[1:]
SHORT_TYPES = ["none", "cellular", "antibody", "mixed"]
OUTCOMES = ["resolved", "persistence"]
PAIRS = list(itertools.combinations(range(4), 2))
PROGRAMME_NAMES = {
    "IL-1-beta": "IL-1β",
    "IL-1-alpha": "IL-1α",
    "IFN-beta": "IFN-β",
    "IFN-alpha1": "IFN-α1",
    "IFN-gamma": "IFN-γ",
    "TNF-alpha": "TNF-α",
}


def programme_name(column: str) -> str:
    """Display name of a programme or of its score column (``score_IL-1-beta`` -> IL-1β)."""
    name = column.replace("score_", "")
    return PROGRAMME_NAMES.get(name, name)


# ---------------------------------------------------------------------------------------------------------------
# cores and biopsies


def first_rejection(core_types):
    """Rejection type of a biopsy as the first rejecting type in core order (depends on the order of the cores)."""
    rejecting = [x for x in core_types if x in REJECTING]
    return rejecting[0] if rejecting else ("no_rejection" if "no_rejection" in list(core_types) else np.nan)


def combined_rejection(core_types):
    """Rejection type of a biopsy from its cores, independent of core order.

    Cellular and antibody rejection over the cores (a mixed core counts as both) give mixed rejection; otherwise
    the one rejecting type, then no rejection, then missing.
    """
    rejecting = {x for x in core_types if x in REJECTING}
    if "mixed_rejection" in rejecting or {"cellular_rejection", "antibody_rejection"} <= rejecting:
        return "mixed_rejection"
    if rejecting:
        return rejecting.pop()
    return "no_rejection" if "no_rejection" in list(core_types) else np.nan


REJECTION_RULES = {"first": first_rejection, "combine": combined_rejection}


def clinical_metadata(metadata) -> pd.DataFrame:
    """Clinical metadata per core with the columns used by the analyses.

    Adds ``rej`` (rejection type of the core), ``grade`` (ISHLT cellular grade of the biopsy), ``core_grade``,
    ``pamr`` (antibody grade), ``epi_raw`` (rejection episode of the patient) and ``bxid`` (the biopsy: patient
    and day after transplantation).
    """
    md = metadata.copy()
    md["rej"] = md.biopsy_rejection_type.astype(str).str.strip().replace("nan", np.nan)
    md["grade"] = pd.to_numeric(md.patient_cellular_grading, errors="coerce")
    md["core_grade"] = pd.to_numeric(md.biopsy_cellular_grading, errors="coerce")
    md["pamr"] = md.patient_antibody_grading.astype(str).str.strip()
    md["epi_raw"] = md.patient_rejection_type.astype(str).str.strip()
    md["bxid"] = md.patient_id.astype(str) + "@d" + md.days_from_transplant_to_biopsy.astype(str)
    return md


def biopsy_cohort(metadata, scores, foci=None, atg=None, grade_rule="clinical", rejection_rule="first"):
    """Join the per-core tables and average the cores of each biopsy.

    Parameters
    ----------
    metadata
        Clinical metadata per core (:func:`clinical_metadata`).
    scores
        Per-core table with the programme scores in columns ``score_<programme>``, indexed by core.
    foci
        Per-core immune fraction, T-cell fraction and foci (``immune_fraction``, ``frac_Tcell``,
        ``dbscan_frac_in_agg``, ``dbscan_largest_agg_cells``), indexed by core.
    atg
        Per-core ``drug_target_score`` of the ATG targets, indexed by core.
    grade_rule
        ``clinical``: the grade of the biopsy as recorded; ``max_core``: the highest grade of its cores.
    rejection_rule
        ``combine`` (:func:`combined_rejection`) or ``first`` (:func:`first_rejection`).

    Returns
    -------
    The per-core table, the per-biopsy table (index ``bxid``; a value of a biopsy is the mean of its cores) and
    the names of the programme score columns. ``core_neg`` marks biopsies without rejection in their cores that
    were taken before treatment of a rejection episode.
    """
    cyto = [k for k in scores.columns if k.startswith("score_")]
    cores = metadata.set_index("Sample")
    features = []
    if foci is not None:
        cores["immf"], cores["ff"] = foci.immune_fraction, foci.dbscan_frac_in_agg
        cores["lf"] = foci.dbscan_largest_agg_cells.clip(lower=1)
        cores["tf"] = foci.frac_Tcell
        features += ["immf", "ff", "lf", "tf"]
    if atg is not None:
        cores["atg"] = atg.drug_target_score
        features += ["atg"]
    for k in cyto:
        cores[k] = scores[k]
    by_biopsy = cores.groupby("bxid")
    biopsies = pd.DataFrame(
        {
            "patient": by_biopsy.patient_id.first(),
            "timing": by_biopsy.biopsy_timing.first(),
            "res": by_biopsy.resolution_broad.first(),
            "rej": by_biopsy.rej.apply(REJECTION_RULES[rejection_rule]),
            "rej_first": by_biopsy.rej.apply(first_rejection),
            "rej_combine": by_biopsy.rej.apply(combined_rejection),
            "epi": by_biopsy.epi_raw.first(),
            "grade": by_biopsy.grade.first(),
            "grade_max_core": by_biopsy.core_grade.max(),
            "pamr": by_biopsy.pamr.first(),
            "ncore": by_biopsy.size(),
        }
    )
    biopsies["grade_clinical"] = biopsies["grade"]
    if grade_rule == "max_core":
        biopsies["grade"] = biopsies["grade_max_core"]
    elif grade_rule != "clinical":
        raise ValueError(grade_rule)
    for feature in features + cyto:
        biopsies[feature] = by_biopsy[feature].mean()
    biopsies["core_neg"] = (biopsies.rej == "no_rejection") & (biopsies.timing == "pre")
    return cores, biopsies, cyto


def cohort_summary(cores, biopsies) -> dict:
    """Counts of the cohort, and how the grade and rejection rules differ on it."""
    both_grades = biopsies.grade_clinical.notna() & biopsies.grade_max_core.notna()
    summary = {
        "cores": int(biopsies.ncore.sum()),
        "biopsies": len(biopsies),
        "patients": int(biopsies.patient.nunique()),
        "core_negative": int(biopsies.core_neg.sum()),
        "grade_rule_differs": int(((biopsies.grade_clinical != biopsies.grade_max_core) & both_grades).sum()),
        "grade_max_core_missing": int(biopsies.grade_max_core.isna().sum()),
        "grade_clinical_gt_max_core": int((biopsies.grade_clinical > biopsies.grade_max_core).sum()),
        "grade_clinical_lt_max_core": int((biopsies.grade_clinical < biopsies.grade_max_core).sum()),
    }
    # biopsies whose cores carry two different rejecting types: there the first-type rule depends on core order
    two_types = cores.groupby("bxid").rej.apply(lambda s: len({x for x in s if x in REJECTING}) > 1)
    summary["biopsies_two_rejecting_types"] = sorted(two_types[two_types].index)
    changed = biopsies[biopsies.rej_first.astype(str) != biopsies.rej_combine.astype(str)]
    summary["rejection_rule_first_vs_combine"] = {
        biopsy: f"{row.rej_first} -> {row.rej_combine}" for biopsy, row in changed.iterrows()
    }
    return summary


def core_grades(adata, foci) -> pd.DataFrame:
    """Per core: cellular grade, rejection type and the share of immune cells in foci (Fig 6 e).

    Parameters
    ----------
    adata
        Cells with the per-core ``obs`` columns ``biopsy_cellular_grading`` and ``biopsy_rejection_type``.
    foci
        Per-core table of :func:`hhca.tl.immune_foci`.

    Returns
    -------
    Cores with a grade and foci, indexed by core: ``grade``, ``rej`` and ``ff``.
    """
    core_labels = (
        adata.obs[["Sample", "biopsy_cellular_grading", "biopsy_rejection_type"]]
        .astype(object)
        .drop_duplicates("Sample")
        .set_index("Sample")
    )
    grade = pd.to_numeric(core_labels.biopsy_cellular_grading, errors="coerce")
    rej = core_labels.biopsy_rejection_type.astype(str).str.strip().replace("nan", np.nan)
    cores = pd.DataFrame({"grade": grade, "rej": rej, "ff": foci.dbscan_frac_in_agg})
    return cores.dropna(subset=["grade", "ff"]).rename_axis("Sample")


def core_overview(adata, features, resolution, timing, classes, class_key="cat", min_immune=25) -> pd.DataFrame:
    """Cores of the overview panels (Fig 6 d, ED9 a) in drawing order.

    Cores of patients with a rejection episode, taken at `timing`, with a grade, foci and at least `min_immune`
    immune cells; sorted by the rejection type of the patient, the grade of the core and its immune fraction.

    Parameters
    ----------
    adata
        Cells with the composition class in ``obs[class_key]`` (missing for non-immune cells) and the per-core
        columns ``patient_rejection_type`` and ``biopsy_timing``.
    features
        Per-core table: ``Sample``, ``n_immune``, ``immune_fraction``, ``biopsy_cellular_grading``,
        ``biopsy_antibody_grading``, ``biopsy_rejection_type``, ``dbscan_frac_in_agg`` and
        ``dbscan_largest_agg_frac``.
    resolution
        Core -> outcome of the rejection episode.
    timing
        ``pre`` or ``post`` treatment.
    classes
        The composition classes, in drawing order.

    Returns
    -------
    One row per core: ``Sample``, ``prt``, ``g``, ``pamr``, ``res``, ``immune_fraction``, ``dbscan_frac_in_agg``,
    ``largest_cells``, ``n_immune`` and the share of each class among the immune cells (``comp::<class>``).
    """
    cores = features.dropna(subset=["biopsy_cellular_grading", "dbscan_frac_in_agg"]).copy()
    cores["g"] = cores.biopsy_cellular_grading.astype(int)
    cores["rej"] = cores.biopsy_rejection_type.astype(str).str.strip()
    cores["pamr"] = cores.biopsy_antibody_grading.astype(str).str.strip()
    cores["largest_cells"] = (cores.dbscan_largest_agg_frac * cores.n_immune).clip(lower=1)
    cores["res"] = cores.Sample.map(resolution).fillna("na")

    cells = pd.DataFrame(
        {"Sample": adata.obs["Sample"].astype(object).to_numpy(), "cat": adata.obs[class_key].astype(object).to_numpy()}
    )
    immune = cells[cells.cat.notna()]
    composition = pd.crosstab(immune.Sample, immune.cat, normalize="index").reindex(columns=classes).fillna(0.0)
    n_immune = immune.groupby("Sample").size()

    meta = adata.obs[["Sample", "patient_rejection_type", "biopsy_timing"]].astype(object)
    meta = meta.drop_duplicates("Sample").set_index("Sample")
    cores["prt"] = cores.Sample.map(meta["patient_rejection_type"]).astype(str).str.strip()
    cores = cores[cores.prt.isin(REJECTING)].copy()
    cores = cores[cores.Sample.map(meta["biopsy_timing"]).astype(str).str.strip() == timing].copy()
    cores = cores[cores.Sample.isin(composition.index[n_immune.reindex(composition.index).fillna(0) >= min_immune])]
    cores["ro"] = cores.prt.map({k: i for i, k in enumerate(REJECTING)})
    cores = cores.sort_values(["ro", "g", "immune_fraction"]).reset_index(drop=True)
    composition = composition.loc[cores.Sample.values]
    overview = cores[
        ["Sample", "prt", "g", "pamr", "res", "immune_fraction", "dbscan_frac_in_agg", "largest_cells", "n_immune"]
    ]
    return pd.concat([overview.reset_index(drop=True), composition.reset_index(drop=True).add_prefix("comp::")], axis=1)


# ---------------------------------------------------------------------------------------------------------------
# tests


def foci_by_grade(biopsies, n_permutations=5000, seed=0) -> dict:
    """Spearman correlation of the cellular grade with the share of immune cells in foci, per biopsy.

    The permutation p shuffles the grade between patients, so the biopsies of a patient stay together.
    """
    complete = biopsies.dropna(subset=["grade", "ff"])
    rho, p = stats.spearmanr(complete.grade, complete.ff)
    rng = np.random.default_rng(seed)
    patient_grade = complete.groupby("patient").grade.first()
    null = []
    for _ in range(n_permutations):
        permuted = pd.Series(rng.permutation(patient_grade.values), index=patient_grade.index)
        null.append(abs(stats.spearmanr(complete.patient.map(permuted), complete.ff)[0]))
    p_perm = (np.sum(np.array(null) >= abs(rho)) + 1) / (len(null) + 1)
    return {"n": len(complete), "patients": int(complete.patient.nunique()), "rho": rho, "p": p, "p_perm": p_perm}


def pre_post_pairs(biopsies) -> tuple[pd.DataFrame, dict]:
    """Immune and T-cell fraction before and after treatment, for patients with both biopsies.

    Returns
    -------
    The paired values per patient (``res``, ``immf_pre``, ``immf_post``, ``tf_pre``, ``tf_post``) and, per
    fraction (``f``: immune, ``g``: T cells) and outcome, the Wilcoxon signed-rank test of pre against post.
    """
    with_outcome = biopsies.dropna(subset=["res"])
    pre = with_outcome[with_outcome.timing == "pre"].set_index("patient")
    post = with_outcome[with_outcome.timing == "post"].set_index("patient")
    both = sorted(set(pre.index) & set(post.index))
    pairs = pd.DataFrame(
        {
            "res": pre.loc[both, "res"],
            "immf_pre": pre.loc[both, "immf"],
            "immf_post": post.loc[both, "immf"],
            "tf_pre": pre.loc[both, "tf"],
            "tf_post": post.loc[both, "tf"],
        }
    ).rename_axis("patient")
    tests = {}
    for column, tag in [("immf", "f"), ("tf", "g")]:
        for group in OUTCOMES:
            patients = [patient for patient in both if pre.loc[patient, "res"] == group]
            x_pre = pre.loc[patients, column].values
            x_post = post.loc[patients, column].values
            tests[f"{tag}_{group}"] = {
                "n": len(x_pre),
                "median_pre": float(np.median(x_pre)),
                "median_post": float(np.median(x_post)),
                "p": stats.wilcoxon(x_pre, x_post).pvalue,
            }
    return pairs, tests


def programmes_vs_no_rejection(biopsies, programmes) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Programme scores of each rejection type against no rejection (Fig 6 h).

    Returns
    -------
    The programme medians per rejection type as z-scores over the four types (rows ordered by the q value of
    cellular rejection), and the Mann-Whitney test of every programme and rejecting type against no rejection
    (``programme``, ``group``, ``p``, ``q``: Benjamini-Hochberg over all tests).
    """
    medians = pd.DataFrame({g: biopsies[biopsies.rej == g][programmes].median() for g in REJECTION_TYPES})
    zscores = medians.sub(medians.mean(1), axis=0).div(medians.std(1).replace(0, np.nan), axis=0)
    p_values, names = [], []
    for k in programmes:
        for g in REJECTING:
            p_values.append(
                stats.mannwhitneyu(biopsies[biopsies.rej == g][k], biopsies[biopsies.rej == "no_rejection"][k]).pvalue
            )
            names.append((k, g))
    q_values = multipletests(p_values, method="fdr_bh")[1]
    q_by_name = dict(zip(names, q_values))
    order = sorted(programmes, key=lambda k: q_by_name[(k, "cellular_rejection")])
    tests = pd.DataFrame(
        [{"programme": k, "group": g, "p": p, "q": q} for (k, g), p, q in zip(names, p_values, q_values)]
    )
    return zscores.loc[order].rename_axis("programme"), tests


def programmes_by_rejection(biopsies, programmes) -> pd.DataFrame:
    """Programme scores across the four rejection types (Fig 6 i, ED9 b-c).

    Per programme a Kruskal-Wallis test over the four types, Benjamini-Hochberg over the programmes, and the six
    pairwise Mann-Whitney tests, Benjamini-Hochberg over the six.

    Returns
    -------
    One row per programme: ``cytokine``, ``column``, ``kruskal_p``, ``kruskal_q_BH_over_18``, the size and median
    of each type (``n_<type>``, ``median_<type>``) and ``p_<a>_vs_<b>``, ``q_<a>_vs_<b>`` of the pairs.
    """
    values = {k: [biopsies.loc[biopsies.rej == g, k].values for g in REJECTION_TYPES] for k in programmes}
    kruskal_p = np.array([stats.kruskal(*values[k]).pvalue for k in programmes])
    kruskal_q = multipletests(kruskal_p, method="fdr_bh")[1]
    rows = []
    for k, kp, kq in zip(programmes, kruskal_p, kruskal_q):
        pair_p = [stats.mannwhitneyu(values[k][i], values[k][j]).pvalue for i, j in PAIRS]
        pair_q = multipletests(pair_p, method="fdr_bh")[1]
        row = {"cytokine": programme_name(k), "column": k, "kruskal_p": kp, "kruskal_q_BH_over_18": kq}
        for g, label in zip(REJECTION_TYPES, SHORT_TYPES):
            row[f"n_{label}"] = int((biopsies.rej == g).sum())
            row[f"median_{label}"] = float(np.median(biopsies.loc[biopsies.rej == g, k]))
        for (i, j), p, q in zip(PAIRS, pair_p, pair_q):
            row[f"p_{SHORT_TYPES[i]}_vs_{SHORT_TYPES[j]}"] = p
            row[f"q_{SHORT_TYPES[i]}_vs_{SHORT_TYPES[j]}"] = q
        rows.append(row)
    return pd.DataFrame(rows)


def kruskal_permutation(biopsies, column, n_permutations=5000, seed=1) -> dict:
    """One score across the four rejection types, with a permutation p that keeps the biopsies of a patient together.

    Returns
    -------
    ``n`` per type, ``kruskal_p``, ``p_perm`` (rejection types shuffled between patients) and ``pairs``: the
    Benjamini-Hochberg q of the six pairwise Mann-Whitney tests.
    """
    values = [biopsies[biopsies.rej == g][column].values for g in REJECTION_TYPES]
    kruskal_p = stats.kruskal(*values).pvalue
    pair_p = [stats.mannwhitneyu(values[i], values[j]).pvalue for i, j in PAIRS]
    pair_q = multipletests(pair_p, method="fdr_bh")[1]
    rng = np.random.default_rng(seed)
    patient_type = biopsies.groupby("patient").rej.first()
    null = []
    observed = stats.kruskal(*values).statistic
    for _ in range(n_permutations):
        permuted = pd.Series(rng.permutation(patient_type.values), index=patient_type.index)
        permuted_type = biopsies.patient.map(permuted)
        groups = [biopsies.loc[permuted_type == g, column].values for g in REJECTION_TYPES]
        groups = [group for group in groups if len(group) > 1]
        null.append(stats.kruskal(*groups).statistic if len(groups) > 1 else 0)
    p_perm = (np.sum(np.array(null) >= observed) + 1) / (len(null) + 1)
    return {
        "n": [len(group) for group in values],
        "kruskal_p": kruskal_p,
        "p_perm": p_perm,
        "pairs": {f"{REJECTION_TYPES[i]}|{REJECTION_TYPES[j]}": q for (i, j), q in zip(PAIRS, pair_q)},
    }


def pool_cell_types(celltype_scores, cell_types, name) -> pd.DataFrame:
    """Scores of several cell types of a core as one: their mean weighted by the number of cells.

    Parameters
    ----------
    celltype_scores
        Per core and cell type (:func:`hhca.tl.programme_means`): the programme columns, ``Sample``, ``ct`` and
        ``n_cells``.
    cell_types
        The cell types to pool.
    name
        Name of the pooled class.

    Returns
    -------
    One row per core with at least one of the cell types: ``Sample``, ``ct`` (= `name`) and the programmes.
    """
    programmes = [column for column in celltype_scores.columns if column not in ("Sample", "ct", "n_cells")]
    pooled = []
    for sample, core in celltype_scores[celltype_scores.ct.isin(cell_types)].groupby("Sample"):
        n = core.n_cells.astype(float)
        pooled.append({"Sample": sample, "ct": name, **{p: (core[p] * n).sum() / n.sum() for p in programmes}})
    return pd.DataFrame(pooled)


def outcome_cohort(metadata) -> pd.DataFrame:
    """Biopsies taken before treatment of a rejection episode with a recorded outcome (index ``bxid``)."""
    by_biopsy = metadata.groupby("bxid")
    biopsies = pd.DataFrame(
        {
            "timing": by_biopsy.biopsy_timing.first(),
            "res": by_biopsy.resolution_broad.first(),
            "epi": by_biopsy.epi_raw.first(),
        }
    )
    return biopsies[(biopsies.timing == "pre") & biopsies.epi.isin(REJECTING) & biopsies.res.notna()]


def outcome_test(biopsies, cores, scores, programme) -> tuple[pd.DataFrame, dict]:
    """One programme of one cell class in persistent against resolved rejection (the test of Fig 6 j).

    Parameters
    ----------
    biopsies, cores
        Per-biopsy and per-core table (:func:`biopsy_cohort`).
    scores
        Per-core scores of the cell class: ``Sample`` and the programme columns.
    programme
        The programme column.

    Returns
    -------
    The tested biopsies (pre-treatment, rejection episode, outcome and score known) with the score as ``il4``,
    and the two-sided Mann-Whitney test with its AUC = U / (n1 n2).
    """
    core_score = scores.set_index("Sample")[programme].rename("il4")
    biopsy_score = cores[["bxid"]].join(core_score).dropna().groupby("bxid").il4.mean()
    tested = biopsies.join(biopsy_score.rename("il4"))
    tested = tested[(tested.timing == "pre") & (tested.epi.isin(REJECTING))].dropna(subset=["il4", "res"])
    resolved, persistence = (tested[tested.res == group].il4.values for group in OUTCOMES)
    mwu = stats.mannwhitneyu(persistence, resolved, alternative="two-sided")
    test = {
        "n": len(tested),
        "resolved": len(resolved),
        "persistence": len(persistence),
        "p": mwu.pvalue,
        "auc": mwu.statistic / (len(resolved) * len(persistence)),
    }
    return tested, test


def outcome_tests(scores, biopsies, min_biopsies=3, pooled=None) -> pd.DataFrame:
    """Every cell type and programme in persistent against resolved rejection.

    Per cell type the score of a biopsy is the mean over its cores. Two-sided Mann-Whitney test of persistence
    against resolved, AUC = U / (n1 n2), for pairs with at least `min_biopsies` biopsies per outcome;
    Benjamini-Hochberg over all tests (``q_BH_all``) and over the tests without the class `pooled`
    (``q_BH_percelltype_family``).

    Parameters
    ----------
    scores
        Per core and cell type: ``Sample``, ``ct``, ``bxid`` and the programme columns.
    biopsies
        The tested biopsies with their outcome ``res`` (:func:`outcome_cohort`).

    Returns
    -------
    One row per test, sorted by p, with its ``rank``.
    """
    programmes = [column for column in scores.columns if column not in ("Sample", "ct", "bxid", "n_cells")]
    rows = []
    for cell_type, type_scores in scores.groupby("ct"):
        biopsy_scores = type_scores.groupby("bxid")[programmes].mean().join(biopsies, how="inner")
        for programme in programmes:
            resolved = biopsy_scores[biopsy_scores.res == "resolved"][programme].dropna().values
            persistence = biopsy_scores[biopsy_scores.res == "persistence"][programme].dropna().values
            if len(resolved) < min_biopsies or len(persistence) < min_biopsies:
                continue
            mwu = stats.mannwhitneyu(persistence, resolved, alternative="two-sided")
            rows.append(
                {
                    "ct": cell_type,
                    "programme": programme,
                    "n_resolved": len(resolved),
                    "n_persistence": len(persistence),
                    "p": mwu.pvalue,
                    "auc": mwu.statistic / (len(resolved) * len(persistence)),
                    "median_resolved": float(np.median(resolved)),
                    "median_persistence": float(np.median(persistence)),
                }
            )
    tests = pd.DataFrame(rows)
    tests["q_BH_all"] = multipletests(tests.p, method="fdr_bh")[1]
    per_celltype = tests[tests.ct != pooled]
    tests["q_BH_percelltype_family"] = np.nan
    tests.loc[per_celltype.index, "q_BH_percelltype_family"] = multipletests(per_celltype.p, method="fdr_bh")[1]
    tests = tests.sort_values("p").reset_index(drop=True)
    tests["rank"] = np.arange(1, len(tests) + 1)
    return tests


def score_correlations(biopsies, score="atg", features=("immf", "tf")) -> dict:
    """Spearman correlation of a per-biopsy score with each feature: ``n``, ``rho`` and ``p`` per feature."""
    tests = {}
    for column in features:
        complete = biopsies.dropna(subset=[column, score])
        rho, p = stats.spearmanr(complete[column], complete[score])
        tests[column] = {"n": len(complete), "rho": rho, "p": p}
    return tests
