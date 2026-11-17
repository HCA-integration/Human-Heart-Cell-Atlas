"""Hormone production and reception in the atlas with hormone2cell: strength per cell state and donor subgroup,
the relevance gate, per-donor sex differences in the left ventricle and the relay to vascular cells.

hormone2cell and its Hormone-Receptor Reference Database (HRDB v2.0.0) are only installed in the environment of
the hormone steps, so they are imported inside the functions that need them.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import re
import tempfile
import time

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from hhca.donors import parse_age
from hhca.io import decode, read_categorical
from hhca.tl._cytokines import model_fit_errors
from hhca.tl._donors import grouped, hedges_g, pool_plan, var_names

PRODUCING_PICKLE = "hormone_producing_definition_v2.0.0.pkl"
RECEPTOR_PICKLE = "hormone_receptor_definition_v2.0.0.pkl"
INCLUDE_COLUMNS = [f"hpc_include{i}" for i in range(1, 9)]  # producer biosynthesis genes (AND)
EXCLUDE_COLUMNS = ["hpc_exclude1", "hpc_exclude2"]  # producer exclude genes (OR gate)
RECEPTOR_COLUMNS = ["receptorgene1", "receptorgene2"]
NOT_A_TYPE = {"Unassigned", "<NA>", "nan", "NA", "None", ""}
RECEIVER_COLUMNS = [
    "Hormone",
    "Celltype_tissue",
    "Strength",
    "Tissue",
    "hormone_display",
    "hormone_short",
    "Tier",
    "assay",
    "Hormone_CT",
]
GATE_COLUMNS = [
    "system",
    "hormone",
    "male_strength",
    "female_strength",
    "mean_or_max_strength_R",
    "diff_male_minus_female",
    "z_male_minus_female",
    "relevance_pass",
    "shown_in_figure",
]
UNTESTABLE_COLUMNS = [
    "mean_male",
    "mean_female",
    "delta_male_minus_female",
    "log2FC_like",
    "hedges_g",
    "mwu_p",
    "ols_beta",
    "ols_p",
    "cohort_ols_p",
    "n_mixed_datasets",
    "n_datasets_agree",
    "n_single_sex_datasets",
    "frac_agree",
]
SEX_DIFFERENCE_COLUMNS = [
    "panel",
    "system",
    "hormone",
    "n_donors",
    "n_male",
    "n_female",
    "median_nsys_cells",
    "mean_male",
    "mean_female",
    "delta_male_minus_female",
    "log2FC_like",
    "hedges_g",
    "fig_diff_male_minus_female",
    "mwu_p",
    "ols_p",
    "ols_q",
    "cohort_ols_p",
    "mwu_q",
    "n_mixed_datasets",
    "n_datasets_agree",
    "frac_agree",
    "n_single_sex_datasets",
    "n_datasets",
    "testable",
    "tier",
    "tier_note",
]

# ---------------------------------------------------------------------------------------------------------------
# definitions of the Hormone-Receptor Reference Database


def package_file(package, name) -> str:
    """Path of a data file shipped inside an installed package (found without importing the package)."""
    spec = importlib.util.find_spec(package)
    return os.path.join(spec.submodule_search_locations[0], name)


def hormone_definitions(producing_pickle=None, receptor_pickle=None):
    """Producing and receptor definitions of the HRDB v2.0.0, by default the tables shipped with hormone2cell.

    Returns
    -------
    The producing definition (one row per biosynthesis route: genes that must all be expressed,
    ``hpc_include1-8``, and genes that must not, ``hpc_exclude1-2``) and the receptor definition
    (``receptorgene1-2`` per hormone).
    """
    producing = pd.read_pickle(producing_pickle or package_file("hormone2cell", PRODUCING_PICKLE))
    receptors = pd.read_pickle(receptor_pickle or package_file("hormone2cell", RECEPTOR_PICKLE))
    return producing, receptors


def established_receptors(receptors) -> pd.DataFrame:
    """Rows of the receptor definition with an established receptor (all rows if the status column is absent)."""
    if "receptor_status" not in receptors.columns:
        return receptors
    status = receptors["receptor_status"].astype(str)
    return receptors[status.str.contains("establ", case=False, na=False)]


def definition_genes(definition, columns) -> list:
    """Sorted gene symbols in the given columns of a definition."""
    genes = pd.unique(definition[columns].to_numpy().ravel())
    return sorted({str(g) for g in genes if pd.notna(g) and str(g) != "nan"})


def hormone_genes(producing, receptors):
    """Genes named by the definitions: the biosynthesis and exclude genes of the producers, and the receptor genes."""
    return (
        definition_genes(producing, INCLUDE_COLUMNS + EXCLUDE_COLUMNS),
        definition_genes(receptors, RECEPTOR_COLUMNS),
    )


def hormone_gene_sets(producing, receptors):
    """The definitions as two flat tables, one row per hormone.

    Producers: every route as its co-required genes joined with ``+``, alternative routes joined with ``|``;
    the exclude genes of a route joined with ``,``. Receptors: the established receptor genes, joined with ``,``.

    Returns
    -------
    The producer table (``hormone``, ``hormone_display``, ``tier``, ``include_genes``, ``exclude_genes``) and the
    receptor table (``hormone``, ``hormone_display``, ``tier``, ``established_receptor_genes``).
    """
    producer_rows = []
    for hormone, rows in producing.groupby("hormone_short"):
        routes = []
        for _, row in rows.iterrows():
            and_genes = [str(g) for g in row[INCLUDE_COLUMNS].dropna().astype(str) if g and g != "nan"]
            exclude_genes = [str(g) for g in row[EXCLUDE_COLUMNS].dropna().astype(str) if g and g != "nan"]
            if and_genes:
                routes.append((list(dict.fromkeys(and_genes)), list(dict.fromkeys(exclude_genes))))
        if routes:
            producer_rows.append(
                {
                    "hormone": hormone,
                    "hormone_display": str(rows["hormone_display"].iloc[0]),
                    "tier": str(rows["Tier"].iloc[0]),
                    "include_genes": "|".join("+".join(and_genes) for and_genes, _ in routes),
                    "exclude_genes": "|".join(",".join(exclude_genes) for _, exclude_genes in routes if exclude_genes),
                }
            )
    receptor_rows = []
    for hormone, rows in established_receptors(receptors).groupby("hormone_short"):
        genes = pd.unique(rows[RECEPTOR_COLUMNS].values.ravel())
        genes = [str(g) for g in genes if isinstance(g, str) and g and g != "nan"]
        if genes:
            receptor_rows.append(
                {
                    "hormone": hormone,
                    "hormone_display": str(rows["hormone_display"].iloc[0]),
                    "tier": str(rows["Tier"].iloc[0]),
                    "established_receptor_genes": ",".join(list(dict.fromkeys(genes))),
                }
            )
    return pd.DataFrame(producer_rows), pd.DataFrame(receptor_rows)


# ---------------------------------------------------------------------------------------------------------------
# pseudobulk of the Level-4 cell states per donor subgroup


def read_obs(atlas, columns) -> pd.DataFrame:
    """Selected ``obs`` columns of the atlas h5ad as a table (categoricals decoded), without loading the counts."""
    out = {}
    with h5py.File(atlas, "r") as h5:
        obs = h5["obs"]
        index_key = decode(obs.attrs.get("_index", "_index"))
        out["_index"] = np.array([decode(v) for v in obs[index_key][:]])
        for column in columns:
            node = obs[column]
            if isinstance(node, h5py.Group) and "categories" in node:
                categories = np.array([decode(x) for x in node["categories"][:]], dtype=object)
                codes = node["codes"][:]
                values = np.where(codes >= 0, categories[np.clip(codes, 0, len(categories) - 1)], None)
                out[column] = pd.Categorical(values, categories=list(categories))
            else:
                values = node[:]
                if values.dtype.kind == "S" or values.dtype == object:
                    out[column] = np.array([decode(v) for v in values], dtype=object)
                else:
                    out[column] = values
    return pd.DataFrame(out).set_index("_index")


def read_profile_header(profile):
    """Level-4 types (column order) and the gene panel of the Level-4 atlas profile."""
    with open(profile) as handle:
        level4_types = handle.readline().rstrip("\n").split("\t")[1:]
        panel_genes = {line.split("\t", 1)[0] for line in handle}
    return level4_types, panel_genes


def donor_subgroups(obs, left_ventricle="Left Ventricle"):
    """Nuclei of the donor subgroups: all, by sex, left ventricle, left ventricle by sex and by age of the women.

    The female donors with left-ventricle nuclei are split at their median age (donors at the median are in
    the older group).

    Parameters
    ----------
    obs
        Table with one row per nucleus and columns ``donor_id``, ``sex``, ``development_stage`` and
        ``tissue_general``.

    Returns
    -------
    Mapping of subgroup -> boolean mask over the nuclei, a table with the nuclei and donors of every subgroup,
    and the age cut-off.
    """
    donor = obs["donor_id"].astype(str).values
    sex = obs["sex"].astype(str).values
    chamber = obs["tissue_general"].astype(str).values
    age = np.array([parse_age(v) for v in obs["development_stage"].astype(str).values])
    per_nucleus = pd.DataFrame({"donor": donor, "sex": sex, "age": age})
    donor_sex = per_nucleus.groupby("donor")["sex"].agg(lambda s: s.value_counts().index[0])
    donor_age = per_nucleus.groupby("donor")["age"].median()

    is_male = sex == "male"
    is_female = sex == "female"
    is_lv = chamber == left_ventricle
    is_lv_female = is_lv & is_female
    lv_female_donors = sorted(set(pd.unique(donor[is_lv_female])))
    lv_female_ages = donor_age.reindex(lv_female_donors).dropna()
    cutoff = float(np.median(lv_female_ages.values))
    younger = set(lv_female_ages.index[lv_female_ages < cutoff])
    older = set(lv_female_ages.index[lv_female_ages >= cutoff])
    is_younger_donor = np.array([d in younger for d in donor])
    is_older_donor = np.array([d in older for d in donor])

    subgroups = {
        "all": np.ones(len(donor), bool),
        "male": is_male,
        "female": is_female,
        "LV": is_lv,
        "LV_male": is_lv & is_male,
        "LV_female": is_lv_female,
        "LV_female_young": is_lv_female & is_younger_donor,
        "LV_female_old": is_lv_female & is_older_donor,
    }
    rows = []
    for name, mask in subgroups.items():
        donors = set(pd.unique(donor[mask]))
        rows.append(
            {
                "subgroup": name,
                "n_cells": int(mask.sum()),
                "n_donors": len(donors),
                "n_male_donors": sum(donor_sex.get(d) == "male" for d in donors),
                "n_female_donors": sum(donor_sex.get(d) == "female" for d in donors),
            }
        )
    return subgroups, pd.DataFrame(rows), cutoff


def _gene_index(genes, atlas):
    """Column index of every gene found in the atlas (case-insensitive)."""
    with h5py.File(atlas, "r") as h5:
        atlas_genes = var_names(h5)
    position = {g: i for i, g in enumerate(atlas_genes)}
    position_upper = {g.upper(): i for i, g in enumerate(atlas_genes)}
    out = {}
    for g in genes:
        if g in position:
            out[g] = position[g]
        elif g.upper() in position_upper:
            out[g] = position_upper[g.upper()]
    return out


def _retry(read, what, log, tries=6, wait=10):
    """Call read(), retrying with a growing pause after a failed read (transient file-system errors)."""
    for i in range(tries):
        try:
            return read()
        except (OSError, RuntimeError) as error:
            if log:
                log(f"  [retry {i + 1}/{tries}] {what} failed: {error}")
            time.sleep(wait * (i + 1))
    raise RuntimeError(f"{what}: exhausted retries")


def _load_checkpoint(path, names):
    """Accumulators and the next block of a streaming checkpoint."""
    saved = np.load(path, allow_pickle=True)
    sums = {name: saved[f"S__{name}"] for name in names}
    detected = {name: saved[f"NZ__{name}"] for name in names}
    nuclei = {name: saved[f"NC__{name}"] for name in names}
    return sums, detected, nuclei, int(saved["next_block"])


def _stream_subgroups(atlas, subgroups, type_code, n_types, kept_idx, layer, block_size, checkpoint_dir, every, log):
    """One pass over the counts, accumulating per (subgroup, cell type, gene).

    Returns three dicts by subgroup: summed full-library CP10k, nuclei with a non-zero count and nuclei per type.
    With a checkpoint folder the pass is saved every `every` blocks and resumes from the saved state.
    """
    names = list(subgroups)
    n_obs = len(type_code)
    sums = {name: np.zeros((n_types, len(kept_idx))) for name in names}
    detected = {name: np.zeros((n_types, len(kept_idx))) for name in names}
    nuclei = {name: np.zeros(n_types, dtype=np.int64) for name in names}
    start_block = 0
    if checkpoint_dir:
        checkpoint = checkpoint_dir + "/pseudobulk_accum.npz"
        partial = checkpoint_dir + "/pseudobulk_accum.tmp.npz"
        if os.path.exists(checkpoint):
            sums, detected, nuclei, start_block = _load_checkpoint(checkpoint, names)
        elif os.path.exists(partial):
            sums, detected, nuclei, start_block = _load_checkpoint(partial, names)
        if log and start_block:
            log(f"streaming pass resumes at block {start_block}")
    if start_block * block_size >= n_obs:
        return sums, detected, nuclei

    n_blocks = (n_obs + block_size - 1) // block_size
    with h5py.File(atlas, "r") as h5:
        counts = h5["layers"][layer]
        n_var = int(counts.attrs["shape"][1])
        indptr = _retry(lambda: counts["indptr"][:], "read indptr", log)
        data, indices = counts["data"], counts["indices"]
        for block in range(start_block, n_blocks):
            i0 = block * block_size
            i1 = min(n_obs, i0 + block_size)
            p0, p1 = int(indptr[i0]), int(indptr[i1])
            if p1 > p0:
                values = _retry(lambda p0=p0, p1=p1: data[p0:p1].astype(np.float64), f"read data blk{block}", log)
                columns = _retry(lambda p0=p0, p1=p1: indices[p0:p1], f"read indices blk{block}", log)
                X = sp.csr_matrix((values, columns, indptr[i0 : i1 + 1] - p0), shape=(i1 - i0, n_var))
                library = np.asarray(X.sum(1)).ravel()
                library[library == 0] = 1.0
                X = sp.diags(1e4 / library) @ X  # full-library CP10k
                X_kept = X[:, kept_idx].tocsc()  # nuclei x kept genes
                X_detected = (X_kept > 0).astype(np.float64)  # non-zero (detection) indicator
                block_types = type_code[i0:i1]
                for name in names:
                    types = np.where(subgroups[name][i0:i1], block_types, -1)
                    m = types >= 0
                    if not m.any():
                        continue
                    one_hot = sp.csr_matrix((np.ones(m.sum()), (types[m], np.nonzero(m)[0])), shape=(n_types, i1 - i0))
                    sums[name] += np.asarray((one_hot @ X_kept).todense())
                    detected[name] += np.asarray((one_hot @ X_detected).todense())
                    np.add.at(nuclei[name], types[m], 1)
            if (block + 1) % every == 0 or block == n_blocks - 1:
                if checkpoint_dir:
                    save = {f"S__{name}": sums[name] for name in names}
                    save.update({f"NZ__{name}": detected[name] for name in names})
                    save.update({f"NC__{name}": nuclei[name] for name in names})
                    save["next_block"] = block + 1
                    np.savez(partial, **save)
                    os.replace(partial, checkpoint)
                if log:
                    log(f"  streamed block {block + 1}/{n_blocks} (cells {i1:,})")
    return sums, detected, nuclei


def subgroup_pseudobulk(
    atlas, subgroups, cell_types, types, genes, layer="counts", block=50_000, checkpoint_dir=None, every=5, log=None
) -> dict:
    """Mean expression of selected genes per cell type, separately for every donor subgroup.

    Every nucleus is scaled to 10,000 counts over all genes; the mean is taken over the nuclei of a type in a
    subgroup. One pass over the atlas accumulates all subgroups.

    Parameters
    ----------
    atlas
        Path of the atlas h5ad.
    subgroups
        Mapping of subgroup -> boolean mask over the nuclei (:func:`donor_subgroups`).
    cell_types
        Cell-type label of every nucleus.
    types
        Cell types to keep, in output order; nuclei of other labels are left out.
    genes
        Genes to keep (matched to the atlas ignoring case).
    layer, block
        Layer with the counts and the number of nuclei read at a time.
    checkpoint_dir
        Folder in which the pass is saved every `every` blocks, so that an interrupted run resumes.

    Returns
    -------
    Mapping of subgroup -> AnnData of cell types x genes: ``X`` is the mean CP10k, ``layers["expressing"]`` the
    number of nuclei with a count above 0 and ``obs["n_nuclei"]`` the nuclei of the type.
    """
    import anndata as ad

    code_of_type = {c: i for i, c in enumerate(types)}
    type_code = np.array([code_of_type.get(c, -1) for c in cell_types], dtype=np.int64)
    atlas_column = _gene_index(genes, atlas)
    kept_genes = [g for g in genes if g in atlas_column]
    kept_idx = np.array([atlas_column[g] for g in kept_genes], dtype=np.int64)
    if log:
        log(f"{len(kept_genes)} of {len(genes)} genes in the atlas")
    sums, detected, nuclei = _stream_subgroups(
        atlas, subgroups, type_code, len(types), kept_idx, layer, block, checkpoint_dir, every, log
    )
    out = {}
    for name in subgroups:
        mean = sums[name] / np.maximum(nuclei[name][:, None], 1)  # cell types x genes
        pseudobulk = pd.DataFrame(mean.T, index=kept_genes, columns=types)
        pseudobulk = pseudobulk.groupby(level=0).sum()
        expressing = pd.DataFrame(detected[name].T, index=kept_genes, columns=types)
        expressing = expressing.groupby(level=0).max()  # expressing nuclei (duplicate symbols: max)
        out[name] = ad.AnnData(
            X=pseudobulk.to_numpy().T,
            obs=pd.DataFrame({"n_nuclei": nuclei[name]}, index=list(types)),
            var=pd.DataFrame(index=list(pseudobulk.index)),
            layers={"expressing": expressing.to_numpy().T},
        )
    return out


# ---------------------------------------------------------------------------------------------------------------
# hormone2cell strength


def _expression_table(pseudobulk, genes, celltype_column, nuclei_per_type, expressing=None):
    """Long expression table (`ave_all`) that hormone2cell expects, from a genes x cell types mean-CP10k frame.

    With `expressing` (genes x cell types counts of expressing nuclei) the detection percentage is the measured
    one. Without it every gene counts as detected (Percentage = 100).
    """
    if genes is not None:
        present = [g for g in genes if g in pseudobulk.index]
        pseudobulk = pseudobulk.loc[present]
    long = pseudobulk.reset_index().melt(id_vars="Gene", var_name="ct", value_name="meanExpr")
    long = long[long["meanExpr"] > 0].copy()
    long["logExpression"] = np.log1p(long["meanExpr"])
    long["Tissue"] = "Heart"
    long[celltype_column] = "Heart____" + long["ct"].astype(str)
    long["TotalCellNumber"] = long["ct"].map(nuclei_per_type).fillna(0).astype(int)
    if expressing is None:
        long["ExpressedCellNumber"] = long["TotalCellNumber"]
        long["Percentage"] = 100.0
    else:
        detected_long = (
            expressing.loc[list(pseudobulk.index)].reset_index().melt(id_vars="Gene", var_name="ct", value_name="nz")
        )
        long = long.merge(detected_long, on=["Gene", "ct"], how="left")
        long["ExpressedCellNumber"] = np.minimum(long["nz"].fillna(0), long["TotalCellNumber"]).astype(int)
        long["Percentage"] = np.where(
            long["TotalCellNumber"] > 0, 100.0 * long["ExpressedCellNumber"] / long["TotalCellNumber"], 0.0
        )
    return long[
        ["Gene", "logExpression", "Tissue", celltype_column, "TotalCellNumber", "ExpressedCellNumber", "Percentage"]
    ]


def _frames(pseudobulk):
    """Genes x cell types frames of mean expression and expressing nuclei, and the nuclei per type."""
    types = list(pseudobulk.obs_names)
    mean = pd.DataFrame(np.asarray(pseudobulk.X).T, index=pd.Index(pseudobulk.var_names, name="Gene"), columns=types)
    expressing = None
    if "expressing" in pseudobulk.layers:
        expressing = pd.DataFrame(np.asarray(pseudobulk.layers["expressing"]).T, index=mean.index, columns=types)
    nuclei_per_type = {ct: int(n) for ct, n in zip(types, pseudobulk.obs["n_nuclei"])}
    return mean, expressing, nuclei_per_type


def _strength(max_expression_file=None, **arguments):
    """Run ``hormone2cell.hormone_strength`` without its progress messages.

    Its table of per-gene maxima goes to `max_expression_file`, or to a temporary file.
    """
    from hormone2cell.hormone_strength import hormone_strength

    with contextlib.redirect_stdout(io.StringIO()), tempfile.TemporaryDirectory() as folder:
        path = max_expression_file or os.path.join(folder, "gene_max_expression.csv")
        return hormone_strength(max_expression_file=path, **arguments)


def _strongest_first(table, celltype_column):
    """One row per (hormone, cell type), strongest first; pairs of equal strength by hormone and cell type."""
    table = table.sort_values(["Strength", "Hormone", celltype_column], ascending=[False, True, True], kind="stable")
    return table.drop_duplicates(["Hormone", celltype_column])


def producer_strength(
    pseudobulk, producing, genes=None, strongest_first=True, max_expression_file=None
) -> pd.DataFrame:
    """Strength with which every cell type produces every hormone (hormone2cell, producing definition).

    A cell type produces a hormone when it expresses all genes of a biosynthesis route above the per-gene
    threshold of hormone2cell and none of the exclude genes; the strength follows the limiting gene. A pseudobulk
    has no detection rate, so every gene counts as detected (assay ``cell``) and the calls rest on the expression
    thresholds.

    Parameters
    ----------
    pseudobulk
        AnnData of cell types x genes with mean CP10k in ``X`` and ``obs["n_nuclei"]``.
    producing
        Producing definition (:func:`hormone_definitions`).
    genes
        Genes offered to hormone2cell (default: all genes of `pseudobulk`).
    strongest_first
        Sort the pairs by decreasing strength and keep one row per hormone and cell type.

    Returns
    -------
    One row per (hormone, cell type) with ``Strength`` above 0.
    """
    mean, _, nuclei_per_type = _frames(pseudobulk)
    producers = _strength(
        max_expression_file,
        ave_all=_expression_table(mean, genes, "Celltype_unique", nuclei_per_type),
        geneset_definition=producing,
        celltype_column="Celltype_unique",
        tissue_col="Tissue",
        adjustment=False,
        assay="cell",
        include_cols=INCLUDE_COLUMNS,
        exclude_cols=EXCLUDE_COLUMNS,
        use_precomputed=None,
    )
    producers["Celltype"] = producers["Celltype_unique"].str.replace("Heart____", "", regex=False)
    producers = producers[producers["Strength"] > 0].copy()
    if strongest_first:
        producers = _strongest_first(producers, "Celltype_unique")
    return producers


def receiver_strength(pseudobulk, receptors, genes=None, max_expression_file=None) -> pd.DataFrame:
    """Strength with which every cell type receives every hormone (hormone2cell, receptor definition).

    Receptor genes are taken from all atlas genes, with the measured share of expressing nuclei as detection rate
    (assay ``nucleus_only``, 5 % detection filter of hormone2cell).

    Parameters
    ----------
    pseudobulk
        AnnData of :func:`subgroup_pseudobulk` (one subgroup).
    receptors
        Receptor definition (:func:`hormone_definitions`).

    Returns
    -------
    One row per (hormone, cell type) with ``Strength`` above 0, strongest first.
    """
    mean, expressing, nuclei_per_type = _frames(pseudobulk)
    receivers = _strength(
        max_expression_file,
        ave_all=_expression_table(mean, genes, "Celltype_tissue", nuclei_per_type, expressing),
        geneset_definition=receptors,
        celltype_column="Celltype_tissue",
        tissue_col="Tissue",
        adjustment=False,
        assay="nucleus_only",
        include_cols=RECEPTOR_COLUMNS,
        exclude_cols=[],
        use_precomputed=None,
    )
    receivers = receivers[receivers["Strength"] > 0].copy()
    receivers = _strongest_first(receivers, "Celltype_tissue")
    receivers["Hormone_CT"] = receivers["Hormone"].astype(str) + "___" + receivers["Celltype_tissue"].astype(str)
    return receivers[[c for c in RECEIVER_COLUMNS if c in receivers.columns]]


def profile_pseudobulk(profile, nuclei_per_type, missing_nuclei=100):
    """A genes x cell types table of mean expression as the AnnData that :func:`producer_strength` takes."""
    import anndata as ad

    return ad.AnnData(
        X=profile.to_numpy().T,
        obs=pd.DataFrame(
            {"n_nuclei": [int(nuclei_per_type.get(ct, missing_nuclei)) for ct in profile.columns]},
            index=list(profile.columns),
        ),
        var=pd.DataFrame(index=list(profile.index)),
    )


def receptor_expression(profile, receptors) -> pd.DataFrame:
    """Mean log1p expression of a hormone's established receptor genes per cell type (OR logic across genes).

    Parameters
    ----------
    profile
        Genes x cell types table of mean expression.
    """
    log_expression = np.log1p(profile)  # genes x cell types
    rows = []
    for hormone, definition_rows in established_receptors(receptors).groupby("hormone_short"):
        genes = pd.unique(definition_rows[RECEPTOR_COLUMNS].values.ravel())
        genes = [g for g in genes if isinstance(g, str) and g in log_expression.index]
        if not genes:
            continue
        score = log_expression.loc[genes].mean(axis=0)
        for celltype, value in score.items():
            rows.append(
                {
                    "Hormone": hormone,
                    "Celltype": celltype,
                    "ReceiverStrength": float(value),
                    "receptor_genes": ",".join(genes),
                }
            )
    return pd.DataFrame(rows)


def focus_producers(producers, hormones) -> pd.DataFrame:
    """All producing cell types of the given hormones, strongest first within a hormone."""
    rows = []
    for hormone in hormones:
        ranked = producers[producers["Hormone"] == hormone].sort_values("Strength", ascending=False)
        for _, r in ranked.iterrows():
            rows.append(
                {
                    "Hormone": hormone,
                    "display": r["hormone_display"],
                    "Celltype": r["Celltype"],
                    "Strength": r["Strength"],
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------
# relevance gate


def cell_system(name, systems):
    """Cell system of a Level-4 type name, or None; `systems` maps system -> patterns on the name."""
    for system, patterns in systems.items():
        if any(re.search(pattern, name, re.IGNORECASE) for pattern in patterns):
            return system
    return None


def _system_strength(table, systems):
    """Mean strength per (system, hormone) of a producer or receiver table."""
    table = table.copy()
    if "Celltype_tissue" in table.columns:
        table["celltype"] = table["Celltype_tissue"].str.replace("Heart____", "", regex=False)
    else:
        table["celltype"] = table["Celltype"].astype(str)
    table["system"] = table["celltype"].map(lambda name: cell_system(name, systems))
    table = table[table["system"].notna()].copy()
    table["hormone_display"] = table["hormone_display"].astype(str)
    means = table.groupby(["system", "hormone_display"])["Strength"].mean().reset_index()
    return means.pivot(index="system", columns="hormone_display", values="Strength").reindex(list(systems))


def relevance_gate(male, female, systems, quantile=0.25, top_n=22, force_include=()) -> pd.DataFrame:
    """Which (hormone, cell system) pairs are relevant enough to be compared between the sexes.

    Per sex, the strength of a hormone in a system is the mean strength over the Level-4 types of the system
    (0 when the hormone is absent). The relevance R is the larger of the two. A pair passes when R reaches the
    `quantile` of the non-zero R values and is above 0; it is shown when its hormone is among the `top_n`
    hormones by their largest R (`force_include` are always kept).

    Parameters
    ----------
    male, female
        Strength tables of the male and the female donors (:func:`producer_strength` or
        :func:`receiver_strength`).
    systems
        Mapping of cell system -> patterns on the Level-4 type names.

    Returns
    -------
    One row per (system, hormone): the strengths, R, the difference male minus female and its z-score over the
    panel, ``relevance_pass`` and ``shown_in_figure``.
    """
    order = list(systems)
    male, female = _system_strength(male, systems), _system_strength(female, systems)
    hormones = sorted(set(male.columns) | set(female.columns))
    male = male.reindex(columns=hormones).fillna(0.0)
    female = female.reindex(columns=hormones).fillna(0.0)
    diff = male - female
    relevance = pd.DataFrame(np.maximum(male.values, female.values), index=male.index, columns=male.columns)
    values = diff.values.astype(float)
    mu, sd = np.nanmean(values), np.nanstd(values)
    z = (diff - mu) / (sd if sd > 0 else 1.0)

    nonzero = relevance.values[relevance.values > 0]
    threshold = float(np.quantile(nonzero, quantile)) if nonzero.size else 0.0
    passes = (relevance >= threshold).any(axis=0)  # hormone relevant in any system
    candidates = relevance.columns[passes.values]
    best = relevance[candidates].max(axis=0)
    # ties broken by hormone name, so the choice at the top_n boundary does not depend on the input row order
    shown_hormones = sorted(candidates, key=lambda c: (-best[c], c))[:top_n]
    for hormone in force_include:
        if hormone in relevance.columns and hormone not in shown_hormones:
            shown_hormones.append(hormone)

    rows = []
    for system in order:
        for hormone in relevance.columns:
            r = relevance.loc[system, hormone]
            passed = bool(r >= threshold and r > 0)
            shown = bool(passed and (hormone in shown_hormones))
            rows.append(
                (
                    system,
                    hormone,
                    male.loc[system, hormone],
                    female.loc[system, hormone],
                    r,
                    diff.loc[system, hormone],
                    z.loc[system, hormone],
                    passed,
                    shown,
                )
            )
    gate = pd.DataFrame(rows, columns=GATE_COLUMNS)
    gate.attrs["threshold"] = threshold
    return gate


# ---------------------------------------------------------------------------------------------------------------
# per-donor expression of the hormone genes in the left ventricle


def _split_genes(text, separator):
    """Gene symbols of a separated list."""
    return [g.strip() for g in str(text).split(separator) if g.strip()]


def _is_missing(value):
    """True for an absent table entry (None or NaN)."""
    return value is None or (isinstance(value, float) and pd.isna(value))


def shown_gene_sets(gates, producer_sets, receptor_sets):
    """Gene sets of the hormones that the relevance gate shows.

    Parameters
    ----------
    gates
        ``{"received": gate table, "produced": gate table}`` (:func:`relevance_gate`).
    producer_sets, receptor_sets
        Tables of :func:`hormone_gene_sets`.

    Returns
    -------
    ``{"received": hormone -> {"kind": "receptor_OR", "genes": [...]}, "produced": hormone -> {"kind":
    "biosynthesis", "or_groups": [[...], ...], "exclude": [...]}}`` and the sorted union of their genes.
    """
    receptors_of = dict(zip(receptor_sets.hormone_display, receptor_sets.established_receptor_genes))
    include_of = dict(zip(producer_sets.hormone_display, producer_sets.include_genes))
    exclude_of = dict(zip(producer_sets.hormone_display, producer_sets.exclude_genes))
    shown = {panel: gate[gate.shown_in_figure] for panel, gate in gates.items()}

    genesets = {"received": {}, "produced": {}}
    for hormone in sorted(shown["received"].hormone.unique()):
        genes = receptors_of.get(hormone)
        genesets["received"][hormone] = {
            "kind": "receptor_OR",
            "genes": [] if pd.isna(genes) else _split_genes(genes, ","),
        }
    for hormone in sorted(shown["produced"].hormone.unique()):
        include, or_groups, exclude = include_of.get(hormone), [], []
        if not _is_missing(include):
            for route in str(include).split("|"):
                and_genes = _split_genes(route, "+")
                if and_genes:
                    or_groups.append(and_genes)
            exclude = exclude_of.get(hormone)
            exclude = [] if _is_missing(exclude) else _split_genes(exclude, ",")
        genesets["produced"][hormone] = {"kind": "biosynthesis", "or_groups": or_groups, "exclude": exclude}

    gene_union = set()
    for definition in genesets["received"].values():
        gene_union.update(definition["genes"])
    for definition in genesets["produced"].values():
        for route in definition["or_groups"]:
            gene_union.update(route)
        gene_union.update(definition["exclude"])
    return genesets, sorted(gene_union)


def _donor_mode(categories, codes, valid, donor_codes, n_donors):
    """Most frequent category over each donor's valid nuclei ("NA" for a donor without one)."""
    out = np.array(["NA"] * n_donors, dtype=object)
    for d in range(n_donors):
        m = valid & (donor_codes == d)
        if m.sum():
            values = codes[m]
            values = values[values >= 0]
            if len(values):
                out[d] = categories[np.bincount(values).argmax()]
    return out


def lv_expression(atlas, genes, level, tissue="Left Ventricle", layer="normcounts", block=50_000, log=None):
    """Mean expression of selected genes per donor and cell type, over the nuclei of one tissue.

    Parameters
    ----------
    atlas
        Path of the atlas h5ad.
    genes
        Genes to keep; those missing from the atlas are left out and listed in ``uns["genes_missing"]``.
    level
        ``obs`` column with the cell-type labels.
    tissue
        Value of ``tissue_general`` to keep.
    layer, block
        Layer to average and the number of nuclei read at a time.

    Returns
    -------
    AnnData with one row per (donor, cell type) and one column per gene: ``X`` is the mean over the nuclei of the
    group (0 for an empty group), ``obs["n_nuclei"]`` their number, and ``obs["dataset"]`` and ``obs["sex"]``
    the most frequent dataset and sex over the donor's nuclei of the tissue.
    """
    with h5py.File(atlas, "r") as h5:
        atlas_genes = var_names(h5)
        atlas_column = {g: i for i, g in enumerate(atlas_genes)}
        present = [g for g in genes if g in atlas_column]
        missing = [g for g in genes if g not in atlas_column]
        n_genes = len(present)
        column_to_gene = -np.ones(len(atlas_genes), dtype=np.int64)
        column_to_gene[np.array([atlas_column[g] for g in present], dtype=np.int64)] = np.arange(n_genes)

        obs = h5["obs"]
        donor_cats, donor_codes = read_categorical(obs, "donor_id")
        donor_codes = donor_codes.astype(np.int64)
        celltype_cats, celltype_codes = read_categorical(obs, level)
        tissue_cats, tissue_codes = read_categorical(obs, "tissue_general")
        dataset_cats, dataset_codes = read_categorical(obs, "dataset")
        sex_cats, sex_codes = read_categorical(obs, "sex")
        celltype_codes, tissue_codes = celltype_codes.astype(np.int64), tissue_codes.astype(np.int64)
        dataset_codes, sex_codes = dataset_codes.astype(np.int64), sex_codes.astype(np.int64)
        n_donors, n_celltypes = len(donor_cats), len(celltype_cats)
        n_obs = len(donor_codes)

        in_tissue = tissue_codes == int(np.where(tissue_cats == tissue)[0][0])
        if log:
            log(f"nuclei {n_obs}, {tissue} nuclei {int(in_tissue.sum())}, donors {n_donors}, types {n_celltypes}")
        # nuclei of the tissue with a donor and a cell type
        valid = in_tissue & (donor_codes >= 0) & (celltype_codes >= 0)
        group = donor_codes * n_celltypes + celltype_codes
        n_groups = n_donors * n_celltypes
        ncells = np.bincount(group[valid], minlength=n_groups).astype(np.int64)
        donor_dataset = _donor_mode(dataset_cats, dataset_codes, valid, donor_codes, n_donors)
        donor_sex = _donor_mode(sex_cats, sex_codes, valid, donor_codes, n_donors)

        counts = h5["layers"][layer]
        indptr = counts["indptr"][:]
        data, indices = counts["data"], counts["indices"]
        sums = np.zeros((n_groups, n_genes), dtype=np.float64)
        for r0 in range(0, n_obs, block):
            r1 = min(r0 + block, n_obs)
            p0, p1 = int(indptr[r0]), int(indptr[r1])
            if p1 == p0:
                continue
            values, columns = data[p0:p1], indices[p0:p1]
            per_row = np.diff(indptr[r0 : r1 + 1])
            keep = column_to_gene[columns] >= 0
            if keep.any():
                rows = np.repeat(np.arange(r1 - r0), per_row)[keep]
                in_group = valid[r0 + rows]
                if in_group.any():
                    rows = rows[in_group]
                    gene_of = column_to_gene[columns[keep][in_group]]
                    weights = values[keep][in_group].astype(np.float64)
                    flat = group[r0 + rows] * n_genes + gene_of
                    sums += np.bincount(flat, weights=weights, minlength=n_groups * n_genes).reshape(n_groups, n_genes)
            if log and (r0 // block) % 5 == 0:
                log(f"  rows {r1}/{n_obs}")

    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(ncells[:, None] > 0, sums / np.maximum(ncells[:, None], 1), 0.0)
    expression = grouped(
        means.astype(np.float32),
        donor_cats,
        celltype_cats,
        ncells.astype(np.int32),
        pd.DataFrame(index=present),
        dataset=np.repeat(donor_dataset.astype(str).astype(object), n_celltypes),
        sex=np.repeat(donor_sex.astype(str).astype(object), n_celltypes),
    )
    expression.uns["genes_missing"] = missing
    return expression


def _lv_blocks(expression):
    """Means (donors x cell types x genes), nuclei (donors x cell types), donors, cell types, dataset and sex."""
    donors = np.array(expression.obs["donor"].cat.categories)
    cell_types = list(expression.obs["cell_type"].cat.categories)
    means = np.asarray(expression.X).reshape(len(donors), len(cell_types), expression.n_vars)
    ncells = expression.obs["n_nuclei"].to_numpy().reshape(len(donors), len(cell_types))
    dataset = expression.obs["dataset"].to_numpy()[:: len(cell_types)]
    sex = expression.obs["sex"].to_numpy()[:: len(cell_types)]
    return means, ncells, donors, cell_types, dataset, sex


def pool_lv_expression(expression, donor_map):
    """Pool the per-donor expression (:func:`lv_expression`) of the atlas donor ids that belong to the same donor.

    The pooled mean is the nucleus-weighted mean of its parts and the nuclei are added up. Dataset and sex of a
    pooled donor are those of its id with the most nuclei.
    """
    means, ncells, donors, cell_types, dataset, sex = _lv_blocks(expression)
    pooled_donors, index_lists, _ = pool_plan(donors, donor_map)
    n_pooled, n_celltypes, n_genes = len(index_lists), means.shape[1], means.shape[2]
    pooled_means = np.zeros((n_pooled, n_celltypes, n_genes), dtype=means.dtype)
    pooled_ncells = np.zeros((n_pooled, n_celltypes), dtype=ncells.dtype)
    for j, indices in enumerate(index_lists):
        n = ncells[indices].astype(np.float64)  # (rows of the donor, cell types)
        total = n.sum(0)  # (cell types,)
        weighted = (means[indices].astype(np.float64) * n[:, :, None]).sum(0)  # (cell types, genes)
        pooled_ncells[j] = total.astype(ncells.dtype)
        with np.errstate(invalid="ignore", divide="ignore"):
            pooled_means[j] = np.where(total[:, None] > 0, weighted / np.maximum(total[:, None], 1), 0.0)
    nuclei = ncells.sum(1)
    representatives = np.array([indices[int(np.argmax(nuclei[indices]))] for indices in index_lists])
    pooled = grouped(
        pooled_means.reshape(-1, n_genes),
        pooled_donors,
        cell_types,
        pooled_ncells.reshape(-1),
        expression.var.copy(),
        dataset=np.repeat(dataset[representatives], n_celltypes),
        sex=np.repeat(sex[representatives], n_celltypes),
    )
    pooled.uns.update(expression.uns)
    return pooled


def cohort_composition(expression) -> pd.DataFrame:
    """Donors with nuclei per dataset and sex (rows: dataset, columns: sex)."""
    _, ncells, donors, _, dataset, sex = _lv_blocks(expression)
    composition = pd.DataFrame({"donor": donors, "dataset": dataset, "sex": sex, "lv_cells": ncells.sum(1)})
    composition = composition[composition.lv_cells > 0]
    return composition.groupby(["dataset", "sex"]).size().unstack(fill_value=0)


# ---------------------------------------------------------------------------------------------------------------
# sex differences per donor


def system_expression(expression, cell_types):
    """Nucleus-weighted mean expression per donor over the cell types of one cell system.

    Parameters
    ----------
    expression
        AnnData of :func:`lv_expression` or :func:`pool_lv_expression`.
    cell_types
        Cell types of the system.

    Returns
    -------
    Donors x genes table (NaN for a donor without nuclei of the system) and the nuclei of the system per donor.
    """
    means, ncells, donors, all_types, _, _ = _lv_blocks(expression)
    celltype_index = {str(c): i for i, c in enumerate(all_types)}
    types = [celltype_index[c] for c in cell_types if c in celltype_index]
    nuclei = ncells[:, types]  # (donors, types)
    total = nuclei.sum(1)  # (donors,)
    weighted = (means[:, types, :] * nuclei[:, :, None]).sum(1)  # (donors, genes) sum over types of mean * nuclei
    with np.errstate(invalid="ignore", divide="ignore"):
        system_mean = np.where(total[:, None] > 0, weighted / np.maximum(total[:, None], 1), np.nan)
    index = pd.Index([str(d) for d in donors], name="donor")
    return pd.DataFrame(system_mean, index=index, columns=list(expression.var_names)), pd.Series(total, index=index)


def hormone_score(system_mean, definition):
    """Per-donor score of one hormone in one cell system.

    Reception: the mean over the hormone's receptor genes. Production: the logic of hormone2cell, the best route
    (max) of the limiting gene (min) of each route.

    Parameters
    ----------
    system_mean
        Donors x genes table of :func:`system_expression`.
    definition
        Gene set of the hormone (:func:`shown_gene_sets`).

    Returns
    -------
    Array with one score per donor, or None when none of the genes was measured.
    """
    gene_index = {g: i for i, g in enumerate(system_mean.columns)}
    values = system_mean.to_numpy()
    if definition["kind"] == "receptor_OR":
        columns = [gene_index[g] for g in definition["genes"] if g in gene_index]
        if not columns:
            return None
        return values[:, columns].mean(axis=1)
    routes = []
    for route in definition["or_groups"]:
        columns = [gene_index[g] for g in route if g in gene_index]
        if not columns:
            continue
        routes.append(np.nanmin(values[:, columns], axis=1))  # co-required genes: the limiting one
    if not routes:
        return None
    return np.nanmax(np.vstack(routes), axis=0)  # alternative routes: the best one


def _ols_p(df, formula, term="C(sex)"):
    """Coefficient and p-value of the sex term; NaN when the model cannot be fitted."""
    import statsmodels.formula.api as smf

    try:
        fit = smf.ols(formula, data=df).fit()
        for name in fit.params.index:
            if name.startswith(term):
                return float(fit.params[name]), float(fit.pvalues[name])
    except model_fit_errors():
        return np.nan, np.nan
    return np.nan, np.nan


def _dataset_agreement(d, overall_sign):
    """Direction agreement across the datasets with at least two donors of each sex.

    Returns the number of such datasets, how many agree with the overall direction, and the number of
    datasets with donors of one sex only.
    """
    mixed = agree = single_sex = 0
    for _, donors in d.groupby("dataset"):
        male = donors[donors.sex == "male"].score.values
        female = donors[donors.sex == "female"].score.values
        if (len(male) == 0) ^ (len(female) == 0):
            single_sex += 1
        if len(male) < 2 or len(female) < 2:
            continue
        mixed += 1
        if np.sign(male.mean() - female.mean()) == overall_sign and overall_sign != 0:
            agree += 1
    return mixed, agree, single_sex


def _assign_tier(result, min_datasets):
    """Robustness tier of every pair: FDR, dataset-adjusted model, agreement across datasets."""
    tiers, notes = [], []
    for _, r in result.iterrows():
        if not r["testable"]:
            tiers.append("NA_underpowered")
            notes.append("<5 donors in a sex")
            continue
        fdr_significant = r["ols_q"] < 0.05
        cohort_ok = r["cohort_ols_p"] < 0.05
        enough_datasets = r["n_mixed_datasets"] >= min_datasets
        direction_ok = (r["frac_agree"] >= 0.5) if pd.notna(r["frac_agree"]) else False
        if fdr_significant and cohort_ok and enough_datasets and direction_ok:
            tiers.append("A_robust")
            notes.append("FDR + cohort-adjusted + majority datasets agree")
        elif fdr_significant and cohort_ok:
            tiers.append("B_cohort_ok_weakdir")
            notes.append("FDR + cohort-adjusted; dataset agreement weak/insufficient")
        elif fdr_significant:
            tiers.append("C_fdr_only")
            notes.append("FDR-sig on naive donor test; NOT robust to dataset")
        else:
            tiers.append("D_ns")
            notes.append("not FDR-significant")
    result["tier"] = tiers
    result["tier_note"] = notes
    return result


def sex_differences(
    expression,
    donors,
    gate,
    gene_sets,
    systems,
    panel,
    min_cells=10,
    min_per_group=5,
    adult_min_age=20,
    min_datasets=3,
) -> pd.DataFrame:
    """Male-female difference of every shown (hormone, cell system) pair of one panel, with donors as replicates.

    Donors are the adults of known sex with at least `min_cells` nuclei of the system. Per pair: the means of
    the two sexes, Hedges g (male minus female), a Mann-Whitney U test, OLS ``score ~ C(sex) + age`` (``ols_p``)
    and the same model with ``C(dataset)`` (``cohort_ols_p``), and the agreement of the direction across the
    datasets that hold both sexes. ``ols_q`` and ``mwu_q`` are Benjamini-Hochberg adjustments within the panel,
    and ``tier`` grades the robustness. Pairs with fewer than `min_per_group` donors of a sex are not tested.

    Parameters
    ----------
    expression
        Pooled per-donor expression (:func:`pool_lv_expression`).
    donors
        Donor table with ``sex`` and ``age``, indexed by donor.
    gate
        Gate table of the panel (:func:`relevance_gate`); its shown pairs are tested.
    gene_sets
        Hormone -> gene set of the panel (:func:`shown_gene_sets`).
    systems
        Mapping of cell system -> cell types of `expression`.
    panel
        Name of the panel (``received`` or ``produced``), stored in the table.

    Returns
    -------
    One row per shown pair.
    """
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    _, _, donor_ids, _, dataset, _ = _lv_blocks(expression)
    meta = pd.DataFrame({"donor": [str(x) for x in donor_ids]})
    meta["sex"] = meta.donor.map(donors["sex"])
    meta["age"] = meta.donor.map(donors["age"])
    meta["dataset"] = np.array([str(x) for x in dataset])
    meta = meta.reset_index(drop=True)
    system_mean, system_nuclei = {}, {}
    for system, cell_types in systems.items():
        mean, total = system_expression(expression, cell_types)
        system_mean[system], system_nuclei[system] = mean, total.to_numpy()

    shown = gate[gate.shown_in_figure][["system", "hormone", "diff_male_minus_female"]]
    rows = []
    for _, dot in shown.iterrows():
        system, hormone = dot["system"], dot["hormone"]
        score = hormone_score(system_mean[system], gene_sets[hormone])
        d = meta.copy()
        d["score"] = score if score is not None else np.nan
        d["nsys"] = system_nuclei[system]
        d = d[(d.nsys >= min_cells) & d.age.notna() & d.sex.notna() & (d.age >= adult_min_age) & d.score.notna()].copy()
        male = d[d.sex == "male"].score.values
        female = d[d.sex == "female"].score.values
        row = {
            "panel": panel,
            "system": system,
            "hormone": hormone,
            "fig_diff_male_minus_female": dot["diff_male_minus_female"],
            "n_donors": len(d),
            "n_male": len(male),
            "n_female": len(female),
            "median_nsys_cells": int(np.median(d.nsys)) if len(d) else 0,
            "n_datasets": d.dataset.nunique(),
        }
        if len(male) >= min_per_group and len(female) >= min_per_group:
            row["mean_male"] = float(male.mean())
            row["mean_female"] = float(female.mean())
            row["delta_male_minus_female"] = float(male.mean() - female.mean())
            row["log2FC_like"] = float((male.mean() - female.mean()) / np.log(2))
            row["hedges_g"] = hedges_g(male, female)
            row["mwu_p"] = float(stats.mannwhitneyu(male, female, alternative="two-sided")[1])
            row["ols_beta"], row["ols_p"] = _ols_p(d, "score ~ C(sex) + age")
            _, row["cohort_ols_p"] = _ols_p(d, "score ~ C(sex) + age + C(dataset)")
            mixed, agree, single_sex = _dataset_agreement(d, np.sign(row["delta_male_minus_female"]))
            row["n_mixed_datasets"] = mixed
            row["n_datasets_agree"] = agree
            row["n_single_sex_datasets"] = single_sex
            row["frac_agree"] = (agree / mixed) if mixed else np.nan
            row["testable"] = True
        else:
            for column in UNTESTABLE_COLUMNS:
                row[column] = np.nan
            row["testable"] = False
        rows.append(row)
    result = pd.DataFrame(rows)
    for p_column, q_column in (("ols_p", "ols_q"), ("mwu_p", "mwu_q")):
        tested = result[p_column].notna()
        result[q_column] = np.nan
        if tested.sum():
            result.loc[tested, q_column] = multipletests(result.loc[tested, p_column], method="fdr_bh")[1]
    return _assign_tier(result, min_datasets)[SEX_DIFFERENCE_COLUMNS]


# ---------------------------------------------------------------------------------------------------------------
# relay from producing cell states to vascular receivers


def cell_family(celltype, families) -> str:
    """Family of a Level-4 cell type: the first family of `families` (family -> patterns) with a matching pattern."""
    for family, patterns in families.items():
        if any(re.search(pattern, celltype, flags=re.IGNORECASE) for pattern in patterns):
            return family
    return "Other"


def _edge_table(table, tiers):
    """Hormone, cell type (without the tissue prefix), strength, display name and tier of the kept tiers."""
    table = table.copy()
    if "Celltype_tissue" in table.columns:
        celltype = table["Celltype_tissue"].astype(str).str.replace(r"^.*____", "", regex=True)
    elif "Celltype" in table.columns:
        celltype = table["Celltype"]
    else:
        celltype = table["Celltype_unique"].astype(str).str.replace(r"^.*____", "", regex=True)
    out = pd.DataFrame(
        {
            "Hormone": table["Hormone"].astype(str),
            "Celltype": celltype.astype(str).str.replace(r"^.*____", "", regex=True),
            "Strength": pd.to_numeric(table["Strength"], errors="coerce"),
            "hormone_display": table.get("hormone_display", table["Hormone"]).astype(str),
            "Tier": table.get("Tier", "1").astype(str),
        }
    )
    out = out.dropna(subset=["Strength"])
    out = out[out["Tier"].isin({str(t) for t in tiers})].copy()
    out["_faint"] = False
    return out


def _strongest_edges(edges, hormones, top_n):
    """The `top_n` strongest edges of each of the given hormones."""
    return (
        edges[edges["Hormone"].isin(hormones)]
        .sort_values("Strength", ascending=False)
        .groupby("Hormone", group_keys=False)
        .head(top_n)
        .copy()
    )


def relay_edges(
    receivers,
    producers,
    hormones,
    families,
    receiver_families,
    tiers=("1", "2", "3"),
    top_n_producers=2,
    top_n_receivers=6,
    display=None,
):
    """Edges of the hormone relay: producing cell states -> hormone -> receiving cell states of chosen families.

    Of the pairs in the mapping-confidence `tiers`, the strongest `top_n_receivers` receiving states of the
    `receiver_families` and the strongest `top_n_producers` producing states of any family are kept per hormone.
    Hormones without a receiving edge are left out.

    Parameters
    ----------
    receivers, producers
        Strength tables of :func:`receiver_strength` and :func:`producer_strength`.
    hormones
        Hormone ids (column ``Hormone``) of the relay.
    families
        Mapping of cell family -> patterns on the Level-4 type names (:func:`cell_family`).
    display
        Hormone id -> name shown; other hormones keep ``hormone_display``.

    Returns
    -------
    The producer edges and the receiver edges (``Hormone``, ``Celltype``, ``Strength``, ``hormone_display``,
    ``Tier``).
    """
    receivers = receivers.copy()
    names = receivers["Celltype_tissue"].astype(str).str.replace(r"^.*____", "", regex=True)
    receivers = receivers[names.map(lambda name: cell_family(name, families)).isin(set(receiver_families))].copy()
    receivers["hormone_display"] = receivers["Hormone"].map(display or {}).fillna(receivers["hormone_display"])
    receivers, producers = _edge_table(receivers, tiers), _edge_table(producers, tiers)
    received = set(receivers["Hormone"])
    shown = [h for h in hormones if h in received]
    return _strongest_edges(producers, shown, top_n_producers), _strongest_edges(receivers, shown, top_n_receivers)
