"""Donors that appear twice in the atlas, found from the expression of their nuclei."""

from __future__ import annotations

import re
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from hhca.io import decode
from hhca.tl._donors import most_frequent, var_names

CALLED_COLUMNS = [
    "donor",
    "match",
    "score",
    "z",
    "r_whole_donor",
    "n_l2_types",
    "shared_barcodes",
    "sex",
    "match_sex",
    "stage",
    "match_stage",
]


def _obs_labels(obs, column):
    """Categories (object array) and int64 codes of an obs column (categorical or plain string array)."""
    node = obs[column]
    if isinstance(node, h5py.Group) and "categories" in node:
        categories = np.array([decode(x) for x in node["categories"][:]], dtype=object)
        return categories, node["codes"][:].astype(np.int64)
    values = np.array([decode(x) for x in node[:]], dtype=object)
    categories, codes = np.unique(values, return_inverse=True)
    return categories, codes.astype(np.int64)


def _centred_logcp10k(sums, groups_of_rows, min_total):
    """Sums (n x G) of raw counts as log1p CP10k, centred on the mean of the row's group (dataset).

    Rows with < min_total counts or alone in their group are NaN.
    """
    total = sums.sum(1)
    ok = total >= min_total
    X = np.full(sums.shape, np.nan)
    X[ok] = np.log1p(sums[ok] / total[ok, None] * 1e4)
    R = np.full(sums.shape, np.nan)
    for g in np.unique(groups_of_rows):
        m = ok & (groups_of_rows == g)
        if m.sum() >= 2:
            R[m] = X[m] - X[m].mean(0)
    return R


def _corr_top_var(R, keep_genes, n_top):
    """Donor x donor correlation over the n_top genes with the largest variance; rows with NaN stay NaN."""
    ok = ~np.isnan(R).any(1)
    C = np.full((R.shape[0], R.shape[0]), np.nan)
    if ok.sum() < 3:
        return C
    Z = R[ok][:, keep_genes]
    top = np.argsort(Z.var(0))[::-1][:n_top]
    Z = Z[:, top]
    Z = (Z - Z.mean(1, keepdims=True)) / Z.std(1, keepdims=True)
    idx = np.where(ok)[0]
    C[np.ix_(idx, idx)] = Z @ Z.T / Z.shape[1]
    return C


def _donor_metadata(labels):
    """Per donor: nuclei and the modes of dataset, sex and development stage over its nuclei, and its tissues."""
    donor_cats, donor = labels["donor_id"]
    dataset_cats, dataset = labels["dataset"]
    sex_cats, sex = labels["sex"]
    stage_cats, stage = labels["development_stage"]
    tissue_cats, tissue = labels["tissue_general"]
    meta = []
    for d in range(len(donor_cats)):
        m = donor == d
        meta.append(
            {
                "donor": donor_cats[d],
                "n_nuclei": int(m.sum()),
                "dataset": most_frequent(dataset_cats[dataset[m]]),
                "sex": most_frequent(sex_cats[sex[m]]),
                "development_stage": most_frequent(stage_cats[stage[m]]),
                "tissues": ";".join(sorted(set(tissue_cats[tissue[m]]))),
            }
        )
    return pd.DataFrame(meta)


def _shared_barcodes(names, donor, log):
    """Number of 16-nt cell barcodes shared by each pair of donors, {(donor code, donor code): n}."""
    pattern = re.compile(r"[ACGT]{16}")
    barcodes = np.array(
        [(pattern.search(str(x)).group(0) if pattern.search(str(x)) else "") for x in names], dtype=object
    )
    owners = defaultdict(set)
    for barcode, d in zip(barcodes, donor):
        if barcode:
            owners[barcode].add(int(d))
    shared = defaultdict(int)
    for donors_of_barcode in owners.values():
        if 1 < len(donors_of_barcode) <= 20:
            ordered = sorted(donors_of_barcode)
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    shared[(ordered[i], ordered[j])] += 1
    if log:
        log(f"barcodes parsed for {np.mean(barcodes != ''):.1%} of nuclei")
    return shared


def _stream_count_sums(h5, group, n_groups, n_genes, chunk, log):
    """Raw counts summed per group of nuclei (groups x genes)."""
    n_obs = len(group)
    sums = np.zeros((n_groups, n_genes), np.float64)
    counts = h5["layers/counts"]
    indptr = counts["indptr"][:]
    for start in range(0, n_obs, chunk):
        end = min(start + chunk, n_obs)
        lo, hi = int(indptr[start]), int(indptr[end])
        X = sp.csr_matrix(
            (counts["data"][lo:hi].astype(np.float64), counts["indices"][lo:hi], indptr[start : end + 1] - lo),
            shape=(end - start, n_genes),
        )
        g = group[start:end]
        one_hot = sp.csr_matrix((np.ones(end - start), (g, np.arange(end - start))), shape=(n_groups, end - start))
        block = (one_hot @ X).tocoo()  # unique (row, col) entries
        sums[block.row, block.col] += block.data
        if log and (start // chunk) % 10 == 0:
            log(f"  {end:,}/{n_obs:,}")
    return sums


def _identity_scores(sums, nuclei, genes, datasets, level2_cats, sex_genes, min_nuclei, n_top):
    """Donor x donor identity score: median over Level-2 types of the within-type correlation.

    Pairs without a common type fall back to the whole-donor correlation. Returns the score, the whole-donor
    correlation and the number of Level-2 types behind each score.
    """
    not_sex_gene = np.array([g not in sex_genes for g in genes])
    whole = sums.sum(1)
    keep_genes = np.where(not_sex_gene & (whole.sum(0) > 0))[0]
    corr_whole = _corr_top_var(_centred_logcp10k(whole, datasets, 1e5), keep_genes, n_top)
    corr_types = []
    for li in range(len(level2_cats)):
        if level2_cats[li] in ("Unassigned",):
            continue
        type_sums = sums[:, li, :].copy()
        type_sums[nuclei[:, li] < min_nuclei] = 0
        corr_types.append(_corr_top_var(_centred_logcp10k(type_sums, datasets, 2e4), keep_genes, n_top))
    corr_types = np.array(corr_types)
    with np.errstate(all="ignore"):
        corr_median = np.nanmedian(corr_types, axis=0)
    n_types = np.sum(~np.isnan(corr_types), axis=0)
    score = np.where(np.isnan(corr_median), corr_whole, corr_median)
    return score, corr_whole, n_types


def _best_matches(score, corr_whole, n_types, shared, meta, donor_cats):
    """The three best cross-dataset matches of every donor, and each donor's best match (donor index)."""
    datasets = meta.dataset.values
    rows = []
    best = {}
    for i in range(len(donor_cats)):
        other = np.where(datasets != datasets[i])[0]
        v = score[i, other]
        finite = ~np.isnan(v)
        if finite.sum() < 10:
            continue
        o, v = other[finite], v[finite]
        order = np.argsort(v)[::-1]
        mu, sd = np.mean(v[order[1:]]), np.std(v[order[1:]])
        best[i] = o[order[0]]
        for r in order[:3]:
            j = o[r]
            a, b = sorted((i, j))
            rows.append(
                {
                    "donor": donor_cats[i],
                    "dataset": datasets[i],
                    "match": donor_cats[j],
                    "match_dataset": datasets[j],
                    "rank": int(np.where(order == r)[0][0]) + 1,
                    "score": float(v[r]),
                    "z": float((v[r] - mu) / sd) if sd > 0 else np.nan,
                    "r_whole_donor": float(corr_whole[i, j]),
                    "n_l2_types": int(n_types[i, j]),
                    "shared_barcodes": int(shared.get((a, b), 0)),
                    "n_nuclei": int(meta.n_nuclei[i]),
                    "match_n_nuclei": int(meta.n_nuclei[j]),
                    "sex": meta.sex[i],
                    "match_sex": meta.sex[j],
                    "stage": meta.development_stage[i],
                    "match_stage": meta.development_stage[j],
                }
            )
    return pd.DataFrame(rows), best


def _call_pairs(candidates, best, donor_cats, z_min):
    """Mutual best matches with z >= z_min, one row per pair."""
    top1 = candidates[candidates["rank"] == 1].copy()
    mutual = top1.apply(
        lambda r: best.get(int(np.where(donor_cats == r.match)[0][0])) == int(np.where(donor_cats == r.donor)[0][0]),
        axis=1,
    )
    called = top1[mutual & (top1.z >= z_min)].copy()
    called["pair"] = called.apply(lambda r: tuple(sorted((r.donor, r.match))), axis=1)
    return called.drop_duplicates("pair")


def duplicate_donors(atlas, level, sex_genes, chunk=20_000, min_nuclei=50, n_top=2000, z_min=5.0, log=None) -> dict:
    """Find the donors that appear twice in the atlas from their nuclei, without using the donor ids.

    The raw counts are summed per donor and cell type and log-normalised; each profile is centred on the mean of
    its dataset, which removes the study batch, the sex genes are dropped and donors are correlated over the
    `n_top` genes with the largest remaining variance. The identity score of two donors is the median of this
    correlation over the cell types with at least `min_nuclei` nuclei in both (the whole-donor correlation is
    reported alongside). Two donors of different datasets are called the same person when they are each other's
    best match and the score lies at least `z_min` standard deviations above the donor's other cross-dataset
    scores. Shared 16-nt cell barcodes are counted as well: a re-release of the same libraries shares them.

    Parameters
    ----------
    atlas
        Path of the atlas h5ad.
    level
        ``obs`` column with the cell types within which donors are compared.
    sex_genes
        Genes left out of the correlation.

    Returns
    -------
    ``{"donors": per-donor table, "candidates": the three best matches of every donor, "pairs": the called
    pairs}``.
    """
    sex_genes = set(sex_genes)
    with h5py.File(atlas, "r") as h5:
        obs = h5["obs"]
        names = np.array([decode(x) for x in obs[decode(obs.attrs.get("_index", "_index"))][:]], dtype=object)
        labels = {
            column: _obs_labels(obs, column)
            for column in ("donor_id", "dataset", "sex", "development_stage", "tissue_general")
        }
        donor_cats, donor = labels["donor_id"]
        level2_cats, level2 = _obs_labels(obs, level)
        genes = var_names(h5)
        n_donors, n_types, n_genes = len(donor_cats), len(level2_cats), len(genes)
        if log:
            log(f"nuclei {len(names):,}  donors {n_donors}  cell types {n_types}  genes {n_genes}")
        meta = _donor_metadata(labels)
        shared = _shared_barcodes(names, donor, log)
        group = donor * n_types + level2
        sums = _stream_count_sums(h5, group, n_donors * n_types, n_genes, chunk, log)
    nuclei = np.bincount(group, minlength=n_donors * n_types).reshape(n_donors, n_types)
    sums = sums.reshape(n_donors, n_types, n_genes)

    score, corr_whole, n_l2_types = _identity_scores(
        sums, nuclei, genes, meta.dataset.values, level2_cats, sex_genes, min_nuclei, n_top
    )
    candidates, best = _best_matches(score, corr_whole, n_l2_types, shared, meta, donor_cats)
    return {"donors": meta, "candidates": candidates, "pairs": _call_pairs(candidates, best, donor_cats, z_min)}


def compare_with_map(pairs, donor_map) -> pd.DataFrame:
    """Called duplicate pairs against the pairs of the donor map.

    Parameters
    ----------
    pairs
        Called pairs of :func:`duplicate_donors`.
    donor_map
        Mapping of atlas donor id -> donor.

    Returns
    -------
    One row per pair with ``in_map`` and ``called``.
    """
    mapped = {tuple(sorted((raw, donor))) for raw, donor in donor_map.items() if raw != donor}
    called_set = set(pairs.pair)
    return pd.DataFrame({"pair": sorted(called_set | mapped)}).assign(
        in_map=lambda d: d.pair.isin(mapped), called=lambda d: d.pair.isin(called_set)
    )
