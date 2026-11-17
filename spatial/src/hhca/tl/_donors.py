"""Donors of the snRNA atlas: their attributes, the donors that were released twice and per-donor gene-set scores."""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from hhca.donors import parse_age
from hhca.io import decode

DONOR_COLUMNS = ["sex", "age", "tissue", "tissue_frac", "assay", "suspension", "disease", "ncells"]


def var_names(h5) -> np.ndarray:
    """Gene names (var index) of an open atlas h5ad."""
    index_key = decode(h5["var"].attrs.get("_index", "_index"))
    return np.array([decode(name) for name in h5["var"][index_key][:]])


def obs_labels(obs, column):
    """Categories (list) and integer codes of an obs column (categorical or plain array) of an open h5ad."""
    node = obs[column]
    if isinstance(node, h5py.Group) and "categories" in node:
        return [decode(category) for category in node["categories"][:]], node["codes"][:]
    values = node[:]
    if values.dtype.kind in "SO":
        values = np.array([decode(value) for value in values])
    categories, codes = np.unique(values, return_inverse=True)
    return list(categories), codes


def most_frequent(values) -> str:
    """Most frequent value, as a string."""
    labels, counts = np.unique(values.astype(str), return_counts=True)
    return labels[np.argmax(counts)]


def _read_labels(atlas, levels):
    """Per-nucleus labels of the atlas: donor code, covariates, and the cell-type codes of every level."""
    with h5py.File(atlas, "r") as h5:
        obs = h5["obs"]
        sex_cats, sex_codes = obs_labels(obs, "sex")
        stage_cats, stage_codes = obs_labels(obs, "development_stage")
        donors, donor_codes = obs_labels(obs, "donor_id")
        tissue_cats, tissue_codes = obs_labels(obs, "tissue_general")
        assay_cats, assay_codes = obs_labels(obs, "assay")
        suspension_cats, suspension_codes = obs_labels(obs, "suspension_type")
        disease_cats, disease_codes = obs_labels(obs, "disease")
        level_codes, level_cats = {}, {}
        for level, column in levels.items():
            level_cats[level], level_codes[level] = obs_labels(obs, column)
    return {
        "donors": list(donors),
        "donor_code": donor_codes.astype(np.int64),
        "sex": np.array(sex_cats, dtype=object)[sex_codes],
        "age": np.array([parse_age(c) for c in stage_cats])[stage_codes],
        "tissue": np.array(tissue_cats, dtype=object)[tissue_codes],
        "assay": np.array(assay_cats, dtype=object)[assay_codes],
        "suspension": np.array(suspension_cats, dtype=object)[suspension_codes],
        "disease": np.array(disease_cats, dtype=object)[disease_codes],
        "level_cats": level_cats,
        "level_codes": level_codes,
    }


def _donor_attributes(labels):
    """Per-donor attributes: mode (median for age) over the donor's nuclei."""
    donor = {key: [] for key in DONOR_COLUMNS}
    for di in range(len(labels["donors"])):
        m = labels["donor_code"] == di
        donor["ncells"].append(int(m.sum()))
        donor["sex"].append(most_frequent(labels["sex"][m]))
        ages = labels["age"][m]
        ages = ages[~np.isnan(ages)]
        donor["age"].append(float(np.median(ages)) if ages.size else np.nan)
        tissues, tissue_counts = np.unique(labels["tissue"][m].astype(str), return_counts=True)
        donor["tissue"].append(tissues[np.argmax(tissue_counts)])
        # share of the donor's nuclei from its most frequent tissue
        donor["tissue_frac"].append(float(tissue_counts.max() / tissue_counts.sum()))
        donor["assay"].append(most_frequent(labels["assay"][m]))
        donor["suspension"].append(most_frequent(labels["suspension"][m]))
        donor["disease"].append(most_frequent(labels["disease"][m]))
    return donor


def donor_table(atlas) -> pd.DataFrame:
    """Attributes of every donor id of the atlas, from the ``obs`` of its nuclei.

    Sex, tissue, assay, suspension type and disease are the most frequent value over the donor's nuclei, age is
    the median of the midpoints of the ``development_stage`` bands, ``tissue_frac`` the share of the nuclei that
    come from the most frequent tissue and ``ncells`` the number of nuclei.

    Parameters
    ----------
    atlas
        Path of the atlas h5ad.

    Returns
    -------
    One row per donor id, in the order of the ``donor_id`` categories.
    """
    labels = _read_labels(atlas, {})
    return pd.DataFrame(_donor_attributes(labels), index=pd.Index(labels["donors"], name="donor"))[DONOR_COLUMNS]


def gene_set_weights(gene_sets, genes):
    """Weights that turn expression into gene-set means.

    W (genes x gene sets) holds 1/n for the n genes of a set that are in `genes`, so ``X @ W`` is the mean
    expression over the set. Gene sets without a gene in `genes` are left out.

    Returns
    -------
    W, the names of the kept sets, their number of genes present and the genes present (comma-separated).
    """
    gene_index = {g: i for i, g in enumerate(genes)}
    rows, cols, data = [], [], []
    names, n_present, genes_present = [], [], []
    for name, members in gene_sets.items():
        present = [gene_index[g] for g in dict.fromkeys(members) if g in gene_index]
        if not present:
            continue
        weight = 1.0 / len(present)
        column = len(names)
        for gene in present:
            rows.append(gene)
            cols.append(column)
            data.append(weight)
        names.append(name)
        n_present.append(len(present))
        genes_present.append(",".join(genes[i] for i in present))
    W = sp.csr_matrix(
        (np.asarray(data, np.float32), (np.asarray(rows), np.asarray(cols))), shape=(len(genes), len(names))
    )
    return W, names, n_present, genes_present


def _stream_scores(atlas, W, labels, levels, chunk, log):
    """Gene-set means (log1p CP10k @ W) summed per (donor, cell type) of every level, and the nuclei per group."""
    n_obs, n_donors = len(labels["donor_code"]), len(labels["donors"])
    n_genes, n_sets = W.shape
    n_types = {level: len(labels["level_cats"][level]) for level in levels}
    sums = {level: np.zeros((n_donors * n_types[level], n_sets), np.float64) for level in levels}
    cnts = {level: np.zeros(n_donors * n_types[level], np.int64) for level in levels}
    group = {
        level: labels["donor_code"] * n_types[level] + labels["level_codes"][level].astype(np.int64) for level in levels
    }
    with h5py.File(atlas, "r") as h5:
        counts = h5["layers/counts"]
        data, indices = counts["data"], counts["indices"]
        indptr = counts["indptr"][:]
        for start in range(0, n_obs, chunk):
            end = min(start + chunk, n_obs)
            lo, hi = int(indptr[start]), int(indptr[end])
            X = sp.csr_matrix(
                (data[lo:hi].astype(np.float32), indices[lo:hi], indptr[start : end + 1] - lo),
                shape=(end - start, n_genes),
            )
            total = np.asarray(X.sum(1)).ravel()
            scale = np.zeros_like(total)
            nonzero = total > 0
            scale[nonzero] = 1e4 / total[nonzero]
            X = X.multiply(scale[:, None]).tocsr()
            X.data = np.log1p(X.data)
            scores = np.asarray(X.dot(W).todense(), np.float64)  # (nuclei of the block, gene sets)
            for level in levels:
                idx = group[level][start:end]
                np.add.at(sums[level], idx, scores)
                cnts[level] += np.bincount(idx, minlength=n_donors * n_types[level])
            if log and (start // chunk) % 5 == 0:
                log(f"  {end:,}/{n_obs:,}")
    return sums, cnts


def grouped(X, donors, cell_types, n_nuclei, var, **obs_columns):
    """AnnData with one row per (donor, cell type), donors in the outer and cell types in the inner order."""
    import anndata as ad

    donors, cell_types = [str(d) for d in donors], [str(c) for c in cell_types]
    obs = pd.DataFrame(
        {
            "donor": pd.Categorical(np.repeat(donors, len(cell_types)), categories=donors),
            "cell_type": pd.Categorical(np.tile(cell_types, len(donors)), categories=cell_types),
            "n_nuclei": n_nuclei,
            **obs_columns,
        }
    )
    obs.index = obs["donor"].astype(str) + " | " + obs["cell_type"].astype(str)
    return ad.AnnData(X=X, obs=obs, var=var)


def gene_set_scores(atlas, gene_sets, levels, chunk=20_000, log=None) -> dict:
    """Gene-set scores of every nucleus of the atlas, summed per donor and cell type.

    The score of a gene set in a nucleus is the mean log1p CP10k over the genes of the set that are in the atlas
    (the definition of ``drug2cell.score(method="mean")``). The counts are read in row blocks; nothing is kept per
    nucleus.

    Parameters
    ----------
    atlas
        Path of the atlas h5ad (``layers["counts"]``).
    gene_sets
        Mapping of gene-set name -> genes.
    levels
        Mapping of level name (``"L3"``) -> ``obs`` column with the cell-type labels of that level.
    chunk
        Number of nuclei read at a time.
    log
        Function called with progress messages.

    Returns
    -------
    Mapping of level name -> AnnData with one row per (donor, cell type) and one column per gene set. ``X`` is
    the sum of the scores over the nuclei of the group and ``obs["n_nuclei"]`` their number, so the mean score of
    a group is ``X / n_nuclei`` and groups can be pooled by adding rows. ``var`` lists the genes present.
    """
    with h5py.File(atlas, "r") as h5:
        genes = var_names(h5)
    W, names, n_present, genes_present = gene_set_weights(gene_sets, genes)
    if log:
        log(f"{len(genes)} genes, {len(names)} of {len(gene_sets)} gene sets with a gene in the atlas")
    labels = _read_labels(atlas, levels)
    if log:
        log(f"{len(labels['donor_code']):,} nuclei, {len(labels['donors'])} donors")
    sums, cnts = _stream_scores(atlas, W, labels, levels, chunk, log)
    var = pd.DataFrame({"n_genes": n_present, "genes": genes_present}, index=names)
    return {
        level: grouped(sums[level], labels["donors"], labels["level_cats"][level], cnts[level], var.copy())
        for level in levels
    }


def donor_score(scores, gene_set, cell_type, min_nuclei=1) -> pd.Series:
    """Mean score of one gene set in one cell type per donor; NaN for donors with fewer than `min_nuclei` nuclei.

    Parameters
    ----------
    scores
        AnnData of :func:`gene_set_scores` (one level).

    Returns
    -------
    Series indexed by donor.
    """
    rows = (scores.obs["cell_type"] == cell_type).to_numpy()
    n = scores.obs["n_nuclei"].to_numpy()[rows]
    sums = np.asarray(scores[rows, gene_set].X).ravel()
    values = np.where(n >= min_nuclei, sums / np.maximum(n, 1), np.nan)
    return pd.Series(
        values, index=pd.Index(scores.obs["donor"].astype(str).to_numpy()[rows], name="donor"), name=gene_set
    )


def pool_plan(donors, donor_map):
    """Group the atlas donor ids by donor, in first-seen order, and pick the id that represents each donor.

    The representative is the id that equals the donor's own id, otherwise the first one.

    Returns
    -------
    The donors, the positions of the atlas ids of each and the position of each representative.
    """
    donors = np.array([str(d) for d in donors])
    pooled_ids = np.array([donor_map.get(d, d) for d in donors])
    order, groups = [], {}
    for i, pooled_id in enumerate(pooled_ids):
        if pooled_id not in groups:
            groups[pooled_id] = []
            order.append(pooled_id)
        groups[pooled_id].append(i)
    index_lists = [groups[pooled_id] for pooled_id in order]
    representatives = [
        next((i for i in indices if donors[i] == pooled_id), indices[0])
        for pooled_id, indices in zip(order, index_lists)
    ]
    return np.array(order), index_lists, np.array(representatives)


def pool_donors(scores, donor_map):
    """Add up the rows of the atlas donor ids that belong to the same donor.

    Parameters
    ----------
    scores
        AnnData of :func:`gene_set_scores`: sums in ``X``, nuclei in ``obs["n_nuclei"]``.
    donor_map
        Mapping of atlas donor id -> donor.

    Returns
    -------
    AnnData of the same layout with one block of rows per donor, donors in first-seen order.
    """
    cell_types = list(scores.obs["cell_type"].cat.categories)
    pooled_ids, index_lists, _ = pool_plan(scores.obs["donor"].cat.categories, donor_map)
    sums, cnts = np.asarray(scores.X), scores.obs["n_nuclei"].to_numpy()
    n_celltypes = len(cell_types)
    pooled_sums = np.zeros((len(index_lists) * n_celltypes, sums.shape[1]), dtype=sums.dtype)
    pooled_cnts = np.zeros((len(index_lists) * n_celltypes,), dtype=cnts.dtype)
    for j, indices in enumerate(index_lists):
        for c in range(n_celltypes):
            rows = [d * n_celltypes + c for d in indices]
            pooled_sums[j * n_celltypes + c] = sums[rows].sum(0)
            pooled_cnts[j * n_celltypes + c] = cnts[rows].sum()
    return grouped(pooled_sums, pooled_ids, cell_types, pooled_cnts, scores.var.copy())


def pool_donor_table(donors, donor_map) -> pd.DataFrame:
    """Donor table of the pooled donors: the attributes of the representative id, nuclei added up.

    Parameters
    ----------
    donors
        Table of :func:`donor_table`.
    donor_map
        Mapping of atlas donor id -> donor.
    """
    pooled_ids, index_lists, representatives = pool_plan(donors.index, donor_map)
    pooled = donors.iloc[representatives].copy()
    pooled.index = pd.Index(pooled_ids, name=donors.index.name)
    pooled["ncells"] = np.array([donors["ncells"].to_numpy()[indices].sum() for indices in index_lists], dtype=np.int64)
    return pooled


def hedges_g(x, y) -> float:
    """Hedges g of x minus y: difference of the means over the pooled standard deviation, small-sample corrected.

    NaN with fewer than two values in a group, 0 when the pooled standard deviation is 0.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    pooled_sd = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if pooled_sd == 0:
        return 0.0
    return (x.mean() - y.mean()) / pooled_sd * (1 - 3 / (4 * (nx + ny) - 9))


def compare_sexes(female, male, min_donors=1) -> dict:
    """Female against male donors: two-sided Mann-Whitney U test and Hedges g (female minus male).

    With fewer than `min_donors` donors of a sex the comparison is not made (NaN).
    """
    from scipy.stats import mannwhitneyu

    tested = len(female) >= min_donors and len(male) >= min_donors
    return {
        "n_female": len(female),
        "n_male": len(male),
        "p": mannwhitneyu(female, male).pvalue if tested else np.nan,
        "hedges_g": hedges_g(female, male) if tested else np.nan,
    }


def split_by_sex(table, value="score", adult_age=None, bands=None):
    """Values of the female and the male donors, optionally of the adults only and per age band.

    Parameters
    ----------
    table
        One row per donor with columns ``sex``, ``age`` and `value`; rows without a value are left out.
    adult_age
        (youngest, oldest) age kept.
    bands
        Age bands as (from, to) pairs, the upper end excluded.

    Returns
    -------
    (female values, male values), or a list of such pairs, one per band.
    """
    keep = table[value].notna().to_numpy()
    age, sex, values = table["age"].to_numpy(float), table["sex"].to_numpy().astype(str), table[value].to_numpy()
    if adult_age is not None:
        keep = keep & (age >= adult_age[0]) & (age <= adult_age[1])
    if bands is None:
        return values[keep & (sex == "female")], values[keep & (sex == "male")]
    out = []
    for lo, hi in bands:
        m = keep & (age >= lo) & (age < hi)
        out.append((values[m & (sex == "female")], values[m & (sex == "male")]))
    return out
