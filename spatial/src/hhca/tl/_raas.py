"""Richness-adjusted RAAS-enzyme levels across vessel types.

Raw per-vessel-type means are confounded by transcript richness (library size), so enzyme levels are compared on
marginal means adjusted for richness and section.

Model (one row per structure and gene):
  y = log1p(CP10k), CP10k = 1e4 * count / lib
  y ~ C(gene):C(vessel_type)              # cell means
    + C(gene):log_lib_centred             # gene-specific richness slope
    + C(gene):C(slide, Sum)               # gene-specific section effects (deviation coded)
  All genes are fitted jointly so that contrasts between enzymes share one covariance; standard errors are
  cluster-robust with the structure as cluster (each structure contributes every gene).

  Section is sum-to-zero coded and log-lib is centred, so each cell-mean coefficient is the estimated marginal
  mean (EMM) at the mean richness, balanced over sections, and any contrast is a difference of two coefficients
  with its variance read from the covariance matrix.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests


def _long_table(structures, genes, vessel_types):
    """Structures with a library, and the long table (one row per structure and gene) of the model."""
    structs = structures[["struct", "slide", "vessel_type", "lib", "n_bins", *genes]]
    structs = structs[structs["vessel_type"].isin(vessel_types)].copy()
    structs = structs[structs["lib"] > 0].copy()
    structs["gid"] = structs["slide"].astype(str) + "|" + structs["struct"].astype(str)  # structure id across slides
    log_lib = np.log(structs["lib"].values)
    log_lib_mean = log_lib.mean()
    per_gene = []
    for gene in genes:
        cp10k = 1e4 * structs[gene].values / structs["lib"].values
        per_gene.append(
            pd.DataFrame(
                {
                    "gid": structs["gid"].values,
                    "slide": structs["slide"].values,
                    "vessel_type": structs["vessel_type"].values,
                    "log_lib_c": log_lib - log_lib_mean,
                    "gene": gene,
                    "y": np.log1p(cp10k),
                }
            )
        )
    return structs, pd.concat(per_gene, ignore_index=True)


def _design(long, slides, genes, vessel_types):
    """Design matrix X, its column names, and the column index of each (gene, vessel type) cell mean.

    Columns:
      25  cell means           : 1 if row is (gene g, vessel v)                -> EMM(g,v)
       5  gene x log_lib slope  : (gene==g) * log_lib_c
      50  gene x slide deviation: (gene==g) * sumcode(slide)  [reference = last slide]
    """
    gene_of_row = long["gene"].values
    vessel_of_row = long["vessel_type"].values
    slide_of_row = long["slide"].values
    log_lib_c = long["log_lib_c"].values

    columns, names, cell_idx = [], [], {}
    for gene in genes:
        for vessel in vessel_types:
            cell_idx[(gene, vessel)] = len(columns)
            columns.append(((gene_of_row == gene) & (vessel_of_row == vessel)).astype(np.float64))
            names.append(f"mu[{gene}|{vessel}]")
    for gene in genes:
        columns.append(np.where(gene_of_row == gene, log_lib_c, 0.0))
        names.append(f"loglib[{gene}]")
    reference = slides[-1]
    for gene in genes:
        is_gene = (gene_of_row == gene).astype(np.float64)
        for slide in slides[:-1]:
            sum_code = np.where(slide_of_row == slide, 1.0, np.where(slide_of_row == reference, -1.0, 0.0))
            columns.append(is_gene * sum_code)
            names.append(f"slide[{gene}|{slide}]")
    X = np.column_stack(columns)
    return X, names, cell_idx


def _slide_pseudobulk(structs, genes, min_bins):
    """Pseudobulk CP10k rate per (slide, vessel type) = 1e4 * sum(count) / sum(lib).

    Used for the slide-level robustness column only: with the slide as replicate, an ordering that also holds
    here does not rest on many structures of a few slides.
    """
    rows = []
    for (slide, vessel), group in structs.groupby(["slide", "vessel_type"]):
        if group["n_bins"].sum() < min_bins:
            continue
        lib_sum = group["lib"].sum()
        row = {"slide": slide, "vessel_type": vessel}
        for gene in genes:
            row[gene] = 1e4 * group[gene].sum() / lib_sum if lib_sum > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _slide_robustness(rates, vessel, higher, lower):
    """Compare the pseudobulk rates of enzymes `higher` and `lower` at a vessel type, paired across slides."""
    sub = rates[rates["vessel_type"] == vessel]
    a, b = sub[higher].values, sub[lower].values
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    n = len(a)
    if n == 0:
        return {"n_slides": 0, "frac_slides_higher": np.nan, "p_slide": np.nan}
    frac = float(np.mean(a > b))
    p = np.nan
    if n >= 4 and np.any(a - b != 0):
        try:
            p = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
        except ValueError:  # too few non-zero differences
            p = np.nan
    return {"n_slides": int(n), "frac_slides_higher": frac, "p_slide": p}


def _contrast(params, cov, i, j):
    """EMM_i - EMM_j (difference of two cell-mean coefficients): difference, SE, z and p."""
    d = params[i] - params[j]
    var = cov[i, i] + cov[j, j] - 2 * cov[i, j]
    se = np.sqrt(max(var, 0.0))
    z = d / se if se > 0 else np.nan
    p = 2 * stats.norm.sf(abs(z)) if se > 0 else np.nan
    return d, se, z, p


def _bron_kerbosch(nodes, adj):
    """All maximal cliques of the 'not significantly different' graph."""
    cliques = []

    def expand(clique, candidates, excluded):
        if not candidates and not excluded:
            cliques.append(set(clique))
            return
        pivot = max(candidates | excluded, key=lambda u: len(adj[u])) if (candidates | excluded) else None
        todo = candidates - adj[pivot] if pivot is not None else set(candidates)
        for v in list(todo):
            expand(clique | {v}, candidates & adj[v], excluded & adj[v])
            candidates = candidates - {v}
            excluded = excluded | {v}

    expand(set(), set(nodes), set())
    return cliques


def _compact_letters(genes_sorted, notdiff_pairs):
    """Compact letter display: {gene: letters}.

    genes_sorted: enzymes by decreasing EMM; notdiff_pairs: frozenset pairs that are not significantly
    different. Two enzymes share a letter if a maximal clique of the not-different graph holds both, so
    enzymes without a common letter differ significantly.
    """
    nodes = list(genes_sorted)
    adj = {u: set() for u in nodes}
    for a, b in itertools.combinations(nodes, 2):
        if frozenset((a, b)) in notdiff_pairs:
            adj[a].add(b)
            adj[b].add(a)
    cliques = _bron_kerbosch(nodes, adj)
    # cliques ordered by their highest member, so that 'a' marks the top group
    rank = {gene: i for i, gene in enumerate(genes_sorted)}
    cliques = sorted(cliques, key=lambda clique: min(rank[gene] for gene in clique))
    letters = dict.fromkeys(nodes, "")
    for i, clique in enumerate(cliques):
        letter = chr(ord("a") + i)
        for gene in clique:
            letters[gene] += letter
    return {gene: "".join(sorted(set(letters[gene]))) for gene in nodes}


def _genes_by_emm(params, cell_idx, vessel, genes):
    """Enzymes by decreasing EMM at a vessel type."""
    return sorted(genes, key=lambda gene: -params[cell_idx[(gene, vessel)]])


def _emm_table(params, cov, cell_idx, n_struct, genes, vessel_types):
    """Adjusted marginal mean of every (gene, vessel type) with its 95 % interval, on both scales."""
    rows = []
    for gene in genes:
        for vessel in vessel_types:
            i = cell_idx[(gene, vessel)]
            mean = params[i]
            se = np.sqrt(cov[i, i])
            lo, hi = mean - 1.96 * se, mean + 1.96 * se
            rows.append(
                {
                    "gene": gene,
                    "vessel_type": vessel,
                    "n_structures": int(n_struct.get(vessel, 0)),
                    "emm_log1p_cp10k": mean,
                    "se": se,
                    "ci_lo_log1p": lo,
                    "ci_hi_log1p": hi,
                    "adj_cp10k": np.expm1(mean),
                    "adj_cp10k_lo": np.expm1(lo),
                    "adj_cp10k_hi": np.expm1(hi),
                }
            )
    return pd.DataFrame(rows)


def _enzyme_contrasts(params, cov, cell_idx, genes, vessel_types):
    """Pairwise contrasts between enzymes within each vessel type, BH-FDR over all of them."""
    rows = []
    for vessel in vessel_types:
        for a, b in itertools.combinations(genes, 2):
            d, se, z, p = _contrast(params, cov, cell_idx[(a, vessel)], cell_idx[(b, vessel)])
            rows.append(
                {
                    "vessel_type": vessel,
                    "gene_a": a,
                    "gene_b": b,
                    "diff_log1p": d,
                    "se": se,
                    "z": z,
                    "p_raw": p,
                    "ci_lo": d - 1.96 * se,
                    "ci_hi": d + 1.96 * se,
                }
            )
    pairs = pd.DataFrame(rows)
    pairs["p_fdr"] = multipletests(pairs["p_raw"].values, method="fdr_bh")[1]
    return pairs


def _enzyme_orderings(pairs, params, cell_idx, rates, vessel_types, alpha, robust_fraction):
    """Every enzyme pair written as higher > lower, with its slide-level robustness."""
    rows = []
    for _, pair in pairs.iterrows():
        vessel = pair["vessel_type"]
        if pair["diff_log1p"] >= 0:
            higher, lower, d = pair["gene_a"], pair["gene_b"], pair["diff_log1p"]
            ci_lo, ci_hi = pair["ci_lo"], pair["ci_hi"]
        else:
            higher, lower, d = pair["gene_b"], pair["gene_a"], -pair["diff_log1p"]
            ci_lo, ci_hi = -pair["ci_hi"], -pair["ci_lo"]
        emm_higher = params[cell_idx[(higher, vessel)]]
        emm_lower = params[cell_idx[(lower, vessel)]]
        robustness = _slide_robustness(rates, vessel, higher, lower)
        significant = bool(pair["p_fdr"] < alpha)
        # robust = FDR-significant and the same direction on at least 70 % of the slides
        robust = bool(
            significant
            and np.isfinite(robustness["frac_slides_higher"])
            and robustness["frac_slides_higher"] >= robust_fraction
        )
        rows.append(
            {
                "vessel_type": vessel,
                "higher_enzyme": higher,
                "lower_enzyme": lower,
                "emm_higher_log1p": emm_higher,
                "emm_lower_log1p": emm_lower,
                "diff_log1p": d,
                "ci_lo_log1p": ci_lo,
                "ci_hi_log1p": ci_hi,
                "adj_cp10k_higher": np.expm1(emm_higher),
                "adj_cp10k_lower": np.expm1(emm_lower),
                "p_raw": pair["p_raw"],
                "p_fdr": pair["p_fdr"],
                "significant": significant,
                "slide_n": robustness["n_slides"],
                "slide_frac_higher": robustness["frac_slides_higher"],
                "slide_p_wilcoxon": robustness["p_slide"],
                "robust_across_slides": robust,
            }
        )
    orderings = pd.DataFrame(rows)
    orderings["vessel_type"] = pd.Categorical(orderings["vessel_type"], vessel_types, ordered=True)
    return orderings.sort_values(["vessel_type", "significant", "p_fdr"], ascending=[True, False, True]).reset_index(
        drop=True
    )


def _letter_displays(pairs, orderings, params, cell_idx, genes, vessel_types, alpha):
    """Compact letter displays per vessel type.

    Returns (cld, cld_rob): cld counts every FDR-significant difference, cld_rob only those that are also
    robust across slides.
    """
    robust_pairs = {
        (row["vessel_type"], frozenset((row["higher_enzyme"], row["lower_enzyme"])))
        for _, row in orderings.iterrows()
        if row["robust_across_slides"]
    }
    cld, cld_rob = {}, {}
    for vessel in vessel_types:
        sub = pairs[pairs["vessel_type"] == vessel]
        notdiff = {frozenset((row["gene_a"], row["gene_b"])) for _, row in sub.iterrows() if not (row["p_fdr"] < alpha)}
        notdiff_rob = {
            frozenset((row["gene_a"], row["gene_b"]))
            for _, row in sub.iterrows()
            if (vessel, frozenset((row["gene_a"], row["gene_b"]))) not in robust_pairs
        }
        genes_sorted = _genes_by_emm(params, cell_idx, vessel, genes)
        cld[vessel] = _compact_letters(genes_sorted, notdiff)
        cld_rob[vessel] = _compact_letters(genes_sorted, notdiff_rob)
    return cld, cld_rob


def _vessel_contrasts(params, cov, cell_idx, genes, vessel_types):
    """Pairwise contrasts between vessel types for each enzyme, BH-FDR over all of them."""
    rows = []
    for gene in genes:
        for a, b in itertools.combinations(vessel_types, 2):
            d, se, z, p = _contrast(params, cov, cell_idx[(gene, a)], cell_idx[(gene, b)])
            rows.append(
                {
                    "gene": gene,
                    "vessel_a": a,
                    "vessel_b": b,
                    "diff_log1p": d,
                    "se": se,
                    "z": z,
                    "p_raw": p,
                    "ci_lo": d - 1.96 * se,
                    "ci_hi": d + 1.96 * se,
                }
            )
    across = pd.DataFrame(rows)
    across["p_fdr"] = multipletests(across["p_raw"].values, method="fdr_bh")[1]
    return across


def _highest_vessel(across, params, cov, cell_idx, genes, vessel_types, alpha):
    """Vessel type with the highest level of each enzyme, tested against the second highest."""
    rows = []
    for gene in genes:
        order = sorted(vessel_types, key=lambda vessel: -params[cell_idx[(gene, vessel)]])
        top, second = order[0], order[1]
        d = _contrast(params, cov, cell_idx[(gene, top)], cell_idx[(gene, second)])[0]
        pair = across[
            (across.gene == gene)
            & (
                ((across.vessel_a == top) & (across.vessel_b == second))
                | ((across.vessel_a == second) & (across.vessel_b == top))
            )
        ]
        p_fdr = float(pair["p_fdr"].iloc[0]) if len(pair) else np.nan
        rows.append(
            {
                "gene": gene,
                "highest_vessel": top,
                "second_vessel": second,
                "diff_log1p_top_vs_2nd": abs(d),
                "p_fdr_top_vs_2nd": p_fdr,
                "top_sig_gt_second": bool(p_fdr < alpha),
            }
        )
    return pd.DataFrame(rows)


def fit_raas(structures, genes, vessel_types, alpha=0.05, min_bins=20, robust_fraction=0.70) -> dict:
    """Fit the joint model of the RAAS genes and compare the adjusted means between enzymes and vessel types.

    Parameters
    ----------
    structures
        One row per vessel structure with ``slide``, ``struct``, ``vessel_type``, the library ``lib``, ``n_bins``
        and the counts of `genes` (:func:`hhca.tl.structure_pseudobulk`, all sections).
    genes
        Genes of the model.
    vessel_types
        Vessel types of the model, in display order.
    alpha
        BH-FDR level of the contrasts.
    min_bins
        Lattice cells needed per (section, vessel type) for a pseudobulk rate.
    robust_fraction
        An ordering between two enzymes is robust when it is FDR-significant and holds on at least this share
        of the sections.

    Returns
    -------
    Dictionary with the tables ``emm`` (adjusted mean of every gene and vessel type with its 95 % interval, on
    the log1p and the CP10k scale), ``rates`` (pseudobulk CP10k per section and vessel type), ``orderings``
    (enzyme pairs within a vessel type), ``across`` (vessel-type pairs per enzyme), ``highest`` (vessel type
    with the highest level of each enzyme against the second), and with ``structures`` (the rows that entered
    the model), ``design_shape``, ``full_rank``, ``params``, ``cov``, ``names``, ``n_structures``, ``letters``
    and ``letters_robust`` (compact letter displays per vessel type).
    """
    structs, long = _long_table(structures, genes, vessel_types)
    slides = sorted(long["slide"].unique().tolist())
    X, names, cell_idx = _design(long, slides, genes, vessel_types)
    fit = sm.OLS(long["y"].values, X).fit(cov_type="cluster", cov_kwds={"groups": long["gid"].values})
    params, cov = fit.params, fit.cov_params()

    n_struct = structs["vessel_type"].value_counts().to_dict()
    emm = _emm_table(params, cov, cell_idx, n_struct, genes, vessel_types)
    pairs = _enzyme_contrasts(params, cov, cell_idx, genes, vessel_types)
    rates = _slide_pseudobulk(structs, genes, min_bins)  # section as replicate
    orderings = _enzyme_orderings(pairs, params, cell_idx, rates, vessel_types, alpha, robust_fraction)
    letters, letters_robust = _letter_displays(pairs, orderings, params, cell_idx, genes, vessel_types, alpha)
    across = _vessel_contrasts(params, cov, cell_idx, genes, vessel_types)
    highest = _highest_vessel(across, params, cov, cell_idx, genes, vessel_types, alpha)
    return {
        "emm": emm,
        "rates": rates,
        "orderings": orderings,
        "across": across,
        "highest": highest,
        "structures": structs,
        "design_shape": X.shape,
        "full_rank": bool(np.linalg.matrix_rank(X) == X.shape[1]),
        "params": params,
        "cov": cov,
        "names": names,
        "cell_index": cell_idx,
        "n_structures": n_struct,
        "letters": letters,
        "letters_robust": letters_robust,
    }
