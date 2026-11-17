"""Cytokine-response programmes in the immune cells of the atlas: per-donor scores and their sex and age effects."""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as ss

from hhca.tl._donors import grouped, obs_labels, var_names

SEX_COLUMNS = [
    "feature",
    "celltype",
    "effect",
    "hedges_g",
    "p",
    "mwu_p",
    "q",
    "p_sens",
    "direction",
    "n_donors_M",
    "n_donors_F",
    "mean_male",
    "mean_female",
    "n_present",
    "nearest_myeloid_proxy",
]
AGE_COLUMNS = [
    "feature",
    "celltype",
    "beta_age",
    "per_decade",
    "spearman_rho",
    "p",
    "q",
    "p_sens",
    "direction",
    "n_donors_M",
    "n_donors_F",
    "mean_male",
    "mean_female",
    "n_present",
    "nearest_myeloid_proxy",
]


def model_fit_errors():
    """What statsmodels' formula OLS raises when a model cannot be fitted for a group of donors.

    No complete rows, a missing factor level or coefficient, a decomposition that does not converge.
    """
    from patsy import PatsyError

    return (ValueError, IndexError, KeyError, np.linalg.LinAlgError, PatsyError)


def cytokine_programmes(curated, match, up_log_fc=0.5, up_adj_p=0.05, top_n=50):
    """Cytokine-response programmes of the atlas immune types from the huCIRA dictionary.

    Per atlas immune type and cytokine, the rows of the matched huCIRA cell types with ``log_fc > up_log_fc`` and
    ``adj_p_value < up_adj_p`` are the up-regulated genes. Genes are ranked by the number of matched cell types
    in which they are up, then by their mean log fold change, and the first `top_n` form the programme.

    Parameters
    ----------
    curated
        Dictionary rows with columns ``gene``, ``log_fc``, ``adj_p_value``, ``celltype``, ``cytokine``.
    match
        ``{"programs": atlas type -> huCIRA cell types, "cytokines": [...], "unmatched": [...],
        "nearest_myeloid": [...]}``.

    Returns
    -------
    ``{"programs": atlas type -> cytokine -> genes, "unmatched": ..., "nearest_myeloid": ...}`` and a table with
    one row per non-empty programme.
    """
    up = curated[(curated.log_fc > up_log_fc) & (curated.adj_p_value < up_adj_p)]
    programs, meta = {}, []
    for atlas_type, hucira_types in match["programs"].items():
        programs[atlas_type] = {}
        for cytokine in sorted(match["cytokines"]):
            rows = up[up.celltype.isin(hucira_types) & (up.cytokine == cytokine)]
            per_gene = rows.groupby("gene").log_fc.agg(n="size", mean="mean")
            genes = list(per_gene.sort_values(["n", "mean"], ascending=False).head(top_n).index)
            programs[atlas_type][cytokine] = genes
            if genes:
                meta.append(
                    {
                        "atlas_celltype": atlas_type,
                        "cytokine": cytokine,
                        "n_genes": len(genes),
                        "hucira_celltypes_used": "|".join(hucira_types),
                        "genes": ",".join(genes),
                    }
                )
    out = {"programs": programs, "unmatched": match["unmatched"], "nearest_myeloid": match["nearest_myeloid"]}
    return out, pd.DataFrame(meta)


def donor_pseudobulk(atlas, genes, level, chunk=20_000, log=None):
    """Log-normalised expression of selected genes, summed per donor and cell type of the atlas.

    Every nucleus is scaled to 10,000 counts over all genes and log1p-transformed; the counts are read in row
    blocks.

    Parameters
    ----------
    atlas
        Path of the atlas h5ad (``layers["counts"]``).
    genes
        Genes to keep; those missing from the atlas are left out.
    level
        ``obs`` column with the cell-type labels.
    chunk
        Number of nuclei read at a time.
    log
        Function called with progress messages.

    Returns
    -------
    AnnData with one row per (donor, cell type) and one column per gene: ``X`` is the sum of log1p CP10k over the
    nuclei of the group and ``obs["n_nuclei"]`` their number.
    """
    with h5py.File(atlas, "r") as h5:
        atlas_genes = var_names(h5)
        obs = h5["obs"]
        donors, donor_codes = obs_labels(obs, "donor_id")
        level_cats, level_codes = obs_labels(obs, level)
    atlas_index = {g: i for i, g in enumerate(atlas_genes)}
    present = [g for g in genes if g in atlas_index]
    gene_idx = np.array([atlas_index[g] for g in present])
    if log:
        log(f"genes present {len(present)}/{len(genes)}")
    n_donors, n_levels, n_genes_atlas = len(donors), len(level_cats), len(atlas_genes)
    group = donor_codes.astype(np.int64) * n_levels + level_codes.astype(np.int64)
    n_groups, n_obs = n_donors * n_levels, len(group)
    sums = np.zeros((n_groups, len(gene_idx)), np.float64)
    cnts = np.zeros(n_groups, np.int64)
    with h5py.File(atlas, "r") as h5:
        counts = h5["layers/counts"]
        data, indices = counts["data"], counts["indices"]
        indptr = counts["indptr"][:]
        for start in range(0, n_obs, chunk):
            end = min(start + chunk, n_obs)
            lo, hi = int(indptr[start]), int(indptr[end])
            X = sp.csr_matrix(
                (data[lo:hi].astype(np.float32), indices[lo:hi], indptr[start : end + 1] - lo),
                shape=(end - start, n_genes_atlas),
            )
            total = np.asarray(X.sum(1)).ravel()
            selected = X[:, gene_idx].toarray().astype(np.float64)
            scale = np.zeros_like(total)
            nonzero = total > 0
            scale[nonzero] = 1e4 / total[nonzero]
            selected = np.log1p(selected * scale[:, None])
            idx = group[start:end]
            np.add.at(sums, idx, selected)
            cnts += np.bincount(idx, minlength=n_groups)
            if log and (start // chunk) % 8 == 0:
                log(f"  {end:,}/{n_obs:,}")
    return grouped(sums, donors, level_cats, cnts, pd.DataFrame(index=present))


def _blocks(pseudobulk):
    """Sums (donors x cell types x genes), nuclei (donors x cell types) and the cell types of a pseudobulk."""
    cell_types = list(pseudobulk.obs["cell_type"].cat.categories)
    n_donors = len(pseudobulk.obs["donor"].cat.categories)
    sums = np.asarray(pseudobulk.X).reshape(n_donors, len(cell_types), pseudobulk.n_vars)
    cnts = pseudobulk.obs["n_nuclei"].to_numpy().reshape(n_donors, len(cell_types))
    return sums, cnts, cell_types


def programme_scores(pseudobulk, programmes, cell_types, min_genes=5):
    """Score of every cytokine programme per donor and immune cell type.

    Per donor and cell type the mean log1p CP10k of each gene is taken first, then the mean over the genes of
    the programme that are in the atlas. Programmes with fewer than `min_genes` genes present are not scored (NaN).

    Parameters
    ----------
    pseudobulk
        AnnData of :func:`donor_pseudobulk` with the programme genes.
    programmes
        Mapping of cell type -> cytokine -> genes.
    cell_types
        Immune cell types to score.

    Returns
    -------
    AnnData with one row per (donor, immune cell type) and one column per cytokine.
    """
    sums, cnts, level_cats = _blocks(pseudobulk)
    gene_column = {g: k for k, g in enumerate(pseudobulk.var_names)}
    cytokines = sorted({cy for ct in programmes for cy in programmes[ct]})
    celltypes = [ct for ct in cell_types if ct in level_cats]
    scores = np.full((sums.shape[0], len(celltypes), len(cytokines)), np.nan)
    nuclei = np.zeros((sums.shape[0], len(celltypes)), np.int64)
    for ai, celltype in enumerate(celltypes):
        ci = level_cats.index(celltype)
        n = cnts[:, ci].astype(float)
        nuclei[:, ai] = cnts[:, ci]
        donor_mean = sums[:, ci, :] / np.maximum(n, 1)[:, None]
        for bi, cytokine in enumerate(cytokines):
            columns = [gene_column[g] for g in programmes[celltype].get(cytokine, []) if g in gene_column]
            if len(columns) >= min_genes:
                scores[:, ai, bi] = donor_mean[:, columns].mean(1)
    return grouped(
        scores.reshape(-1, len(cytokines)),
        pseudobulk.obs["donor"].cat.categories,
        celltypes,
        nuclei.reshape(-1),
        pd.DataFrame(index=cytokines),
    )


def _hedges_g(a, b):
    """Hedges g of a minus b from the standard deviations; NaN with fewer than two values in a group or no spread."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_sd = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    return np.nan if pooled_sd == 0 else (a.mean() - b.mean()) / pooled_sd * (1 - 3 / (4 * (na + nb) - 9))


def _has_two_levels(d, covariate):
    """True when the covariate has at least two levels with two or more donors each."""
    return (d[covariate].value_counts() >= 2).sum() >= 2


def _sex_effect(d, male_scores, female_scores, base):
    """Sex effect of one programme in one cell type.

    A model that cannot be fitted leaves the Mann-Whitney p (main model) or NaN (sensitivity model).
    """
    import statsmodels.formula.api as smf

    row = dict(base)
    row["effect"] = row["mean_male"] - row["mean_female"]
    row["hedges_g"] = _hedges_g(male_scores, female_scores)
    try:
        row["mwu_p"] = float(ss.mannwhitneyu(male_scores, female_scores, alternative="two-sided")[1])
    except ValueError:
        row["mwu_p"] = np.nan
    try:
        fit = smf.ols("y ~ C(sex, Treatment('female')) + age", data=d).fit()
        sex_term = [x for x in fit.params.index if x.startswith("C(sex")][0]
        row["p"] = float(fit.pvalues[sex_term])
    except model_fit_errors():
        row["p"] = row["mwu_p"]
    terms = (
        ["C(sex, Treatment('female'))", "age"]
        + (["C(assay)"] if _has_two_levels(d, "assay") else [])
        + (["C(tissue)"] if _has_two_levels(d, "tissue") else [])
    )
    try:
        fit = smf.ols("y ~ " + " + ".join(terms), data=d).fit()
        sex_term = [x for x in fit.params.index if x.startswith("C(sex")][0]
        row["p_sens"] = float(fit.pvalues[sex_term])
    except model_fit_errors():
        row["p_sens"] = np.nan
    row["direction"] = "male_higher" if row["effect"] > 0 else "female_higher"
    return row


def _age_effect(d, base):
    """Age effect of one programme in one cell type; a model that cannot be fitted leaves NaN."""
    import statsmodels.formula.api as smf

    row = dict(base)
    row["spearman_rho"] = float(ss.spearmanr(d.age, d.y)[0])
    try:
        fit = smf.ols("y ~ age + C(sex)", data=d).fit()
        row["beta_age"] = float(fit.params["age"])
        row["per_decade"] = row["beta_age"] * 10
        row["p"] = float(fit.pvalues["age"])
    except model_fit_errors():
        row["beta_age"] = np.nan
        row["per_decade"] = np.nan
        row["p"] = np.nan
    terms = (
        ["age", "C(sex)"]
        + (["C(assay)"] if _has_two_levels(d, "assay") else [])
        + (["C(tissue)"] if _has_two_levels(d, "tissue") else [])
    )
    try:
        fit = smf.ols("y ~ " + " + ".join(terms), data=d).fit()
        row["p_sens"] = float(fit.pvalues["age"])
    except model_fit_errors():
        row["p_sens"] = np.nan
    row["direction"] = "up_with_age" if (row["beta_age"] or 0) > 0 else "down_with_age"
    return row


def _add_fdr(table):
    """Add the BH-FDR q of the tests with a p-value."""
    from statsmodels.stats.multitest import multipletests

    tested = table["p"].notna()
    table["q"] = np.nan
    if tested.any():
        table.loc[tested, "q"] = multipletests(table.loc[tested, "p"], method="fdr_bh")[1]


def programme_effects(
    pseudobulk,
    donors,
    programmes,
    cell_types,
    nearest_myeloid=(),
    min_cells=20,
    min_donors=8,
    min_genes=5,
    adult_age=(20, 89),
):
    """Sex and age effect of every cytokine programme in every immune cell type, on the atlas donor ids.

    Donors are the adults with at least `min_cells` nuclei of the cell type; a programme is tested when at least
    `min_donors` donors of each sex remain. Sex: OLS ``score ~ sex + age``, Hedges g (male minus female) and a
    Mann-Whitney U test. Age: OLS ``score ~ age + sex`` and Spearman rho. ``p_sens`` is the p-value of the same
    term with tissue and assay added to the model, ``q`` the Benjamini-Hochberg adjustment over all tests of a
    table.

    Parameters
    ----------
    pseudobulk
        AnnData of :func:`donor_pseudobulk` with the programme genes.
    donors
        Donor table (:func:`hhca.tl.donor_table`) in the donor order of `pseudobulk`.
    programmes
        Mapping of cell type -> cytokine -> genes.
    nearest_myeloid
        Cell types scored with the programmes of a related huCIRA cell type (flagged in the tables).

    Returns
    -------
    The sex table and the age table, one row per programme and cell type, sorted by p.
    """
    sums, cnts, level_cats = _blocks(pseudobulk)
    gene_column = {g: k for k, g in enumerate(pseudobulk.var_names)}
    cytokines = sorted({cy for ct in programmes for cy in programmes[ct]})
    donor_table = donors.reset_index()[["donor", "sex", "age", "tissue", "assay"]]
    adult = ((donor_table.age >= adult_age[0]) & (donor_table.age <= adult_age[1])).values
    sex_rows, age_rows = [], []
    for celltype in cell_types:
        if celltype not in level_cats:
            continue
        ci = level_cats.index(celltype)
        n = cnts[:, ci].astype(float)
        use = adult & (n >= min_cells)
        if use.sum() < min_donors * 2:
            continue
        donor_mean = sums[:, ci, :] / np.maximum(n, 1)[:, None]  # donor x gene mean in this cell type
        donors_used = donor_table[use].copy()
        for cytokine in cytokines:
            genes = [g for g in programmes[celltype].get(cytokine, []) if g in gene_column]
            if len(genes) < min_genes:
                continue
            score = donor_mean[:, [gene_column[g] for g in genes]].mean(1)  # programme score per donor
            d = donors_used.copy()
            d["y"] = score[use]
            male = d.sex.values == "male"
            female = d.sex.values == "female"
            if male.sum() < min_donors or female.sum() < min_donors:
                continue
            male_scores, female_scores = d["y"].values[male], d["y"].values[female]
            base = {
                "feature": cytokine,
                "celltype": celltype,
                "n_present": len(genes),
                "nearest_myeloid_proxy": (celltype in nearest_myeloid),
                "n_donors_M": int(male.sum()),
                "n_donors_F": int(female.sum()),
                "mean_male": float(male_scores.mean()),
                "mean_female": float(female_scores.mean()),
            }
            sex_rows.append(_sex_effect(d, male_scores, female_scores, base))
            age_rows.append(_age_effect(d, base))
    sex_table, age_table = pd.DataFrame(sex_rows), pd.DataFrame(age_rows)
    _add_fdr(sex_table)
    _add_fdr(age_table)
    return sex_table[SEX_COLUMNS].sort_values("p"), age_table[AGE_COLUMNS].sort_values("p")


def _weighted_mode(values, weights):
    """Value with the largest summed weight."""
    total = {}
    for value, weight in zip(values, weights):
        total[value] = total.get(value, 0) + weight
    return max(total, key=total.get)


def _weighted_median(values, weights):
    """Value at which the cumulative weight reaches half of the total."""
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    cumulative = np.cumsum(weights)
    return values[np.searchsorted(cumulative, cumulative[-1] / 2.0)]


def pool_programme_scores(scores, donors, donor_map):
    """Pool the programme scores of the atlas donor ids that belong to the same donor.

    The scores are means over nuclei, so a pooled score is the nucleus-weighted mean of its parts. Age is the
    nucleus-weighted median, sex, tissue and assay the nucleus-weighted mode.

    Parameters
    ----------
    scores
        AnnData of :func:`programme_scores`.
    donors
        Donor table with columns ``sex``, ``age``, ``tissue``, ``assay`` and ``ncells``, in the donor order of
        `scores`.
    donor_map
        Mapping of atlas donor id -> donor.

    Returns
    -------
    The pooled scores and the table of the pooled donors.
    """
    celltypes, cytokines = list(scores.obs["cell_type"].cat.categories), list(scores.var_names)
    data = {
        "donors": np.array([str(x) for x in donors.index]),
        "age": np.asarray(donors["age"], float),
        "sex": np.array([str(x) for x in donors["sex"]]),
        "tissue": np.array([str(x) for x in donors["tissue"]]),
        "assay": np.array([str(x) for x in donors["assay"]]),
        "ncells": np.asarray(donors["ncells"], float),
        "scores": np.asarray(scores.X).astype(float).reshape(len(donors), len(celltypes), len(cytokines)),
        "cnts": scores.obs["n_nuclei"].to_numpy().astype(float).reshape(len(donors), len(celltypes)),
    }
    pooled_ids = np.array([donor_map.get(d, d) for d in data["donors"]])
    pooled_donors = pd.unique(pooled_ids)
    n_donors, n_celltypes, n_cytokines = len(pooled_donors), len(celltypes), len(cytokines)
    pooled_scores = np.full((n_donors, n_celltypes, n_cytokines), np.nan)
    cnts = np.zeros((n_donors, n_celltypes))
    age = np.zeros(n_donors)
    sex, tissue, assay = (np.empty(n_donors, object) for _ in range(3))
    for i, donor in enumerate(pooled_donors):
        idx = np.where(pooled_ids == donor)[0]
        nuclei = data["ncells"][idx]
        age[i] = _weighted_median(data["age"][idx], nuclei)
        sex[i] = _weighted_mode(data["sex"][idx], nuclei)
        tissue[i] = _weighted_mode(data["tissue"][idx], nuclei)
        assay[i] = _weighted_mode(data["assay"][idx], nuclei)
        for ct in range(n_celltypes):
            n = data["cnts"][idx, ct]
            cnts[i, ct] = n.sum()
            for cy in range(n_cytokines):
                values = data["scores"][idx, ct, cy]
                finite = np.isfinite(values) & (n > 0)
                if finite.any():
                    pooled_scores[i, ct, cy] = np.sum(values[finite] * n[finite]) / np.sum(n[finite])
    table = pd.DataFrame(
        {"sex": sex.astype(str), "age": age, "tissue": tissue.astype(str), "assay": assay.astype(str)},
        index=pd.Index(pooled_donors, name="donor"),
    )
    pooled = grouped(
        pooled_scores.reshape(-1, n_cytokines),
        pooled_donors,
        celltypes,
        cnts.reshape(-1),
        pd.DataFrame(index=cytokines),
    )
    return pooled, table


def _formula_safe(labels):
    """Category labels as formula-safe strings."""
    return pd.Series([str(x) for x in labels]).str.replace(r"[^A-Za-z0-9]", "_", regex=True).values


def age_effects(scores, donors, cell_types, min_cells=20, adult_age=(20, 89)) -> pd.DataFrame:
    """Age effect of every cytokine programme in the given immune cell types, on the pooled donors.

    Donors are the adults with at least `min_cells` nuclei of the cell type and a score. Per programme: OLS
    ``score ~ age + C(sex)`` (``beta_age``, ``p``), Spearman rho of score against age, and ``p_sens`` from the
    model with tissue and assay added; ``q`` is the Benjamini-Hochberg adjustment over all tests.

    Parameters
    ----------
    scores, donors
        Pooled scores and donor table of :func:`pool_programme_scores`.

    Returns
    -------
    One row per programme and cell type.
    """
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests

    age, sex = np.asarray(donors["age"], float), np.array([str(x) for x in donors["sex"]])
    tissue, assay = np.array([str(x) for x in donors["tissue"]]), np.array([str(x) for x in donors["assay"]])
    celltypes, cytokines = list(scores.obs["cell_type"].cat.categories), list(scores.var_names)
    values = np.asarray(scores.X).reshape(len(donors), len(celltypes), len(cytokines))
    cnts = scores.obs["n_nuclei"].to_numpy().reshape(len(donors), len(celltypes))
    adult = (age >= adult_age[0]) & (age <= adult_age[1])
    rows = []
    for celltype in cell_types:
        ci = celltypes.index(celltype)
        for cytokine in cytokines:
            score = values[:, ci, cytokines.index(cytokine)]
            use = adult & (cnts[:, ci] >= min_cells) & np.isfinite(score)
            if use.sum() == 0:
                continue
            d = pd.DataFrame(
                {
                    "score": score[use],
                    "age": age[use],
                    "sex": np.char.lower(sex[use].astype(str)),
                    "tissue": _formula_safe(tissue[use]),
                    "assay": _formula_safe(assay[use]),
                }
            )
            rho, _ = ss.spearmanr(d["age"], d["score"])
            fit = smf.ols("score ~ age + C(sex)", d).fit()
            beta, p = fit.params["age"], fit.pvalues["age"]
            try:
                p_sens = smf.ols("score ~ age + C(sex)+C(tissue)+C(assay)", d).fit().pvalues["age"]
            except model_fit_errors():
                p_sens = np.nan
            rows.append(
                {
                    "feature": cytokine,
                    "celltype": celltype,
                    "beta_age": beta,
                    "per_decade": beta * 10,
                    "spearman_rho": rho,
                    "p": p,
                    "p_sens": p_sens,
                    "direction": "up_with_age" if beta > 0 else "down_with_age",
                    "n_donors_M": (d["sex"] == "male").sum(),
                    "n_donors_F": (d["sex"] == "female").sum(),
                    "mean_male": d.loc[d.sex == "male", "score"].mean(),
                    "mean_female": d.loc[d.sex == "female", "score"].mean(),
                }
            )
    table = pd.DataFrame(rows)
    table["q"] = multipletests(table["p"].values, method="fdr_bh")[1]
    return table


def changing_programmes(effects, cell_type, alpha=0.05) -> pd.DataFrame:
    """Programmes of one cell type with q below `alpha`.

    ``robust`` marks those whose sensitivity model also has p below `alpha`; they come first, each group by q.
    """
    effects = effects[effects["celltype"] == cell_type].copy()
    effects["robust"] = (effects["q"] < alpha) & (effects["p_sens"] < alpha)
    return effects[effects["q"] < alpha].sort_values(["robust", "q"], ascending=[False, True])


def programme_matrix(effects, cell_types, alpha=0.05) -> dict:
    """Age effects as programmes x cell types, for the programmes with q below `alpha` in any cell type.

    Parameters
    ----------
    effects
        Table of :func:`age_effects`.
    cell_types
        Cell types, in column order.

    Returns
    -------
    ``{"rho": Spearman rho, "robust": q and sensitivity p below alpha, "fdr": q below alpha}``, the programmes
    sorted by their mean rho.
    """
    effects = effects.copy()
    effects["robust"] = (effects["q"] < alpha) & (effects["p_sens"] < alpha)
    effects["fdr"] = effects["q"] < alpha
    rho = effects.pivot_table(index="feature", columns="celltype", values="spearman_rho", aggfunc="mean")
    robust = effects.pivot_table(index="feature", columns="celltype", values="robust", aggfunc="max")
    fdr = effects.pivot_table(index="feature", columns="celltype", values="fdr", aggfunc="max")
    rho = rho.reindex(columns=[c for c in cell_types if c in rho.columns])
    robust = robust.reindex(index=rho.index, columns=rho.columns)
    fdr = fdr.reindex(index=rho.index, columns=rho.columns)
    keep = fdr.eq(True).any(axis=1)
    rho, robust, fdr = rho.loc[keep], robust.loc[keep], fdr.loc[keep]
    order = rho.mean(axis=1).sort_values().index
    return {"rho": rho.loc[order], "robust": robust.loc[order], "fdr": fdr.loc[order]}


def loess_band(x, y, grid, frac=0.75, n_boot=300, seed=0) -> pd.DataFrame:
    """LOESS fit of y on x evaluated on `grid`, with a 95 % band from `n_boot` bootstrap resamples.

    Returns
    -------
    Table indexed by the grid with columns ``fit``, ``low`` and ``high``.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    def fit(xx, yy):
        lo = lowess(yy, xx, frac=frac, return_sorted=True)
        return np.interp(grid, lo[:, 0], lo[:, 1])

    center = fit(x, y)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, len(grid)))
    n = len(x)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = fit(x[idx], y[idx])
    lo = np.nanpercentile(boots, 2.5, axis=0)
    hi = np.nanpercentile(boots, 97.5, axis=0)
    return pd.DataFrame({"fit": center, "low": lo, "high": hi}, index=pd.Index(grid, name="age"))


def age_trajectories(
    scores, donors, programmes, cell_type, grid, frac=0.8, n_boot=200, seed=100, adult_age=(20, 89)
) -> dict:
    """Course of programme scores over age in one cell type.

    The scores of the adult donors are z-standardised within each programme and smoothed over age
    (:func:`loess_band`); programme k of the list uses the random seed ``seed + k``.

    Parameters
    ----------
    scores, donors
        Pooled scores and donor table of :func:`pool_programme_scores`.
    programmes
        Cytokines to fit, in order.
    grid
        Ages at which the fit is evaluated.

    Returns
    -------
    Mapping of programme -> table of :func:`loess_band`.
    """
    rows = (scores.obs["cell_type"] == cell_type).to_numpy()
    age = np.asarray(donors["age"], float)
    out = {}
    for k, programme in enumerate(programmes):
        table = pd.DataFrame({"age": age, "score": np.asarray(scores[rows, programme].X).ravel()})
        table = table.dropna(subset=["age", "score"])
        table = table[(table["age"] >= adult_age[0]) & (table["age"] <= adult_age[1])].reset_index(drop=True)
        x, y = table["age"].values, table["score"].values
        mu, sd = np.nanmean(y), np.nanstd(y)
        z = (y - mu) / sd if sd > 0 else y * 0
        out[programme] = loess_band(x, z, grid, frac=frac, n_boot=n_boot, seed=seed + k)
    return out
