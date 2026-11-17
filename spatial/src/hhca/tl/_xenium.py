"""Xenium cells with transferred atlas labels: lymphoid subtypes, cell classes, the endothelial marker gate,
immune foci and gene-set scores per cell, cell type and biopsy core (Fig 6 c-l, ED9, ED6 i-j).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

POOLED_CARDIOMYOCYTES = "Cardiomyocytes"
STRUCTURAL = "structural / other"

# short names of Level-3 types on the healthy section (ED6 i)
SHORT_NAMES = {
    "Arterial Endothelial cells": "Arterial EC",
    "Capillary Endothelial cells": "Capillary EC",
    "Venous Endothelial cells": "Venous EC",
    "Endocardial Endothelial Cells": "Endocardial EC",
    "Lymphatic Endothelial Cells": "Lymphatic EC",
    "Vascular Smooth Muscle cells": "Smooth muscle",
    "Neural Cells": "Neural",
    "Plasma Cells": "Plasma cells",
    "Lymphoid Cells (Proliferating)": "Lymphoid (prolif.)",
}
LYMPHATIC = "Lymphatic Endothelial Cells"

# names of the cytokine programmes in the programme table
PROGRAMME_LABELS = {
    "IFN-beta": "IFN-β",
    "IFN-gamma": "IFN-γ",
    "IFN-alpha1": "IFN-α",
    "IL-1-beta": "IL-1β",
    "IL-1-alpha": "IL-1α",
    "TNF-alpha": "TNF-α",
    "GM-CSF": "GM-CSF",
    "M-CSF": "M-CSF",
    "OSM": "OSM",
}


# ---------------------------------------------------------------------------------------------------------------
# labels


def lymphoid_level4(
    latent, level3, reference_latent, reference_lineage, compartment, proliferating, names, k=25
) -> pd.DataFrame:
    """Lymphoid Level-4 label of the cells of the lymphoid compartment.

    The lineage (CD4 T, CD8 T, Treg, NK) is transferred from atlas cells by a distance-weighted k-nearest-neighbour
    vote in the standardised CellPin latent space; whether a cell is proliferating is taken from its Level-3 label.

    Parameters
    ----------
    latent
        Cells x latent dimensions of the Xenium cells, standardised like the reference.
    level3
        Transferred Level-3 label of each cell (a Series; its index names the cells).
    reference_latent, reference_lineage
        Latent position and lineage of the atlas cells.
    compartment
        Level-3 labels that make up the lymphoid compartment.
    proliferating
        The Level-3 label of proliferating lymphoid cells.
    names
        Lineage -> name of its Level-4 label.
    k
        Number of neighbours.

    Returns
    -------
    One row per cell of the compartment: ``L3``, ``lineage``, ``is_prolif``, ``label8`` (the lineage, with the
    prefix "Proliferating" for proliferating cells) and ``confidence`` (share of the neighbour weights of the
    winning lineage).
    """
    from sklearn.neighbors import KNeighborsClassifier

    level3 = pd.Series(level3)
    query = level3.isin(compartment).to_numpy()
    classifier = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)
    classifier.fit(reference_latent, reference_lineage)
    proba = classifier.predict_proba(latent[query])
    lineage = classifier.classes_[proba.argmax(1)]
    cells = pd.DataFrame({"L3": level3[query]})
    cells["lineage"] = [names[x] for x in lineage]
    cells["is_prolif"] = (cells["L3"] == proliferating).values
    cells["label8"] = np.where(cells.is_prolif, "Proliferating " + cells.lineage, cells.lineage)
    cells["confidence"] = np.round(proba.max(1), 4)
    return cells


def atlas_labels(adata, cellpin, pool=(), exclude=(), lymphoid=None) -> None:
    """Add the transferred labels of the label table to the cells.

    Adds ``obs["atlas_L3"]`` (the Level-3 label, with the types in `pool` joined into "Cardiomyocytes"),
    ``obs["atlas_immune"]``, ``obs["lymphoid_lineage"]``, ``obs["label8"]`` (lymphoid Level 4, missing outside the
    lymphoid compartment) and ``obs["label8_confidence"]``.

    Parameters
    ----------
    adata
        Cells of the rejection cohort.
    cellpin
        Label table of ``scripts/xenium/assemble_labels.py``, one row per cell in the order of `adata`.
    pool
        Level-3 types joined into one cardiomyocyte class.
    exclude
        Level-3 types that must not be flagged immune.
    lymphoid
        The lymphoid Level-4 labels that may occur.
    """
    assert (cellpin["barcode"].to_numpy() == adata.obs_names.to_numpy()).all(), "label table is not in cell order"
    level3 = cellpin["cellpin_L3"].astype(str)
    adata.obs["atlas_L3"] = np.where(level3.isin(pool), POOLED_CARDIOMYOCYTES, level3)
    adata.obs["atlas_immune"] = cellpin["immune"].astype(bool).to_numpy()
    assert not adata.obs.loc[adata.obs["atlas_L3"].isin(exclude), "atlas_immune"].any(), "excluded class marked immune"
    adata.obs["lymphoid_lineage"] = cellpin["lymphoid_lineage"].to_numpy()
    adata.obs["label8"] = cellpin["lymphoid8"].to_numpy()
    adata.obs["label8_confidence"] = cellpin["lymphoid8_conf"].to_numpy()
    if lymphoid is not None:
        found = set(adata.obs["label8"].dropna())
        assert found <= set(lymphoid), found - set(lymphoid)


def cell_classes(adata, level3, lymphoid=None, exclude=(), default=None, key_added="ct") -> None:
    """Group the cells into the classes of one analysis.

    Cells of the lymphoid compartment take the class of their lymphoid Level-4 label, the other cells the class
    of their Level-3 label; cells of neither stay without a class.

    Parameters
    ----------
    adata
        Cells with ``obs["atlas_L3"]`` and ``obs["label8"]`` (:func:`atlas_labels`).
    level3
        Level-3 label -> class.
    lymphoid
        Lymphoid Level-4 label -> class; without it every cell is classed by its Level-3 label.
    exclude
        Level-3 labels left without a class.
    default
        Class of the cells that are not covered (default: missing).
    key_added
        Name of the new ``obs`` column.
    """
    level3 = {label: name for label, name in level3.items() if label not in exclude}
    classes = adata.obs["atlas_L3"].astype(object).map(level3)
    if lymphoid is not None:
        level4 = adata.obs["label8"].astype(object)
        classes = np.where(level4.notna(), level4.map(lymphoid), classes)
    classes = pd.Series(classes, index=adata.obs_names, dtype=object)
    adata.obs[key_added] = classes if default is None else classes.fillna(default)


def spatial_classes(adata, immune_level3, lymphoid, lineages, exclude=()) -> pd.DataFrame:
    """Cells with their centroid and the classes drawn on tissue (Fig 6 c, ED6 j).

    Parameters
    ----------
    adata
        Cells with labels (:func:`atlas_labels`) and centroids in ``obsm["spatial"]``.
    immune_level3
        Level-3 immune types drawn next to the lymphoid Level-4 classes (Fig 6 c).
    lymphoid
        The lymphoid Level-4 labels.
    lineages
        Immune lineage -> its Level-3 types (ED6 j).
    exclude
        Level-3 types that are drawn as structural cells.

    Returns
    -------
    One row per cell: ``cell_id``, ``Sample``, ``slide``, ``x``, ``y``, ``core_grade``, ``fig6c`` (lymphoid
    Level-4 label or immune Level-3 type, missing for structural cells) and ``ed6j`` (immune lineage).
    """
    obs = adata.obs
    lineage_of = {level3: lineage for lineage, types in lineages.items() for level3 in types}
    cell_classes(adata, {name: name for name in immune_level3}, {name: name for name in lymphoid}, key_added="fig6c")
    cell_classes(adata, lineage_of, exclude=exclude, default=STRUCTURAL, key_added="ed6j")
    return pd.DataFrame(
        {
            "cell_id": obs["cell_id"].to_numpy(),
            "Sample": obs["Sample"].to_numpy(),
            "slide": obs["slide"].to_numpy(),
            "x": adata.obsm["spatial"][:, 0],
            "y": adata.obsm["spatial"][:, 1],
            "core_grade": obs["biopsy_cellular_grading"].to_numpy(),
            "fig6c": obs["fig6c"].to_numpy(),
            "ed6j": obs["ed6j"].to_numpy(),
        }
    )


def endothelial_gate(
    adata,
    level3,
    ec_classes,
    markers,
    lymphatic_markers,
    min_counts,
    min_fraction,
    pool=(),
    negative_label="EC label, marker-negative",
) -> None:
    """Display classes of the healthy section: endothelial subtypes only for cells that express endothelial genes.

    A cell passes the gate when its raw transcripts summed over the pan-endothelial `markers` (plus the
    `lymphatic_markers` for cells labelled lymphatic endothelium) are at least `min_counts` and at least
    `min_fraction` of all its transcripts. Adds ``obs["ec_marker_counts"]``, ``obs["total_counts"]``,
    ``obs["ec_gate"]`` (the gate, for every cell), ``obs["ec_gate_pass"]`` (endothelial label and gate) and
    ``obs["display"]``: the short name of the Level-3 label, with the types in `pool` joined into
    "Cardiomyocytes" and `negative_label` for cells with an endothelial label that fail the gate.

    Parameters
    ----------
    adata
        Cells x genes with raw counts in ``X``.
    level3
        Transferred Level-3 label of each cell.
    ec_classes
        The endothelial Level-3 types.
    """
    counts = adata.X.tocsc()
    gene_index = {gene: i for i, gene in enumerate(adata.var_names)}
    pan_markers = [gene for gene in markers if gene in gene_index]
    extra_markers = [gene for gene in lymphatic_markers if gene in gene_index]
    pan_counts = np.asarray(counts[:, [gene_index[gene] for gene in pan_markers]].sum(1)).ravel()
    lymphatic_counts = np.asarray(counts[:, [gene_index[gene] for gene in extra_markers]].sum(1)).ravel()
    total = np.maximum(np.asarray(counts.sum(1)).ravel(), 1)

    level3 = pd.Series(np.asarray(level3), index=adata.obs_names).astype(str)
    marker = np.where((level3 == LYMPHATIC).values, pan_counts + lymphatic_counts, pan_counts)
    is_ec = level3.isin(ec_classes).values
    gate = (marker >= min_counts) & (marker / total >= min_fraction)
    display = level3.map(lambda name: SHORT_NAMES.get(name, name)).to_numpy(dtype=object)
    display[level3.isin(pool).values] = POOLED_CARDIOMYOCYTES
    display[is_ec & ~gate] = negative_label

    adata.obs["ec_marker_counts"] = marker
    adata.obs["total_counts"] = total
    adata.obs["ec_gate"] = gate
    adata.obs["ec_gate_pass"] = gate & is_ec
    adata.obs["display"] = display
    adata.uns["endothelial_gate"] = {"markers": pan_markers, "lymphatic_markers": extra_markers}


# ---------------------------------------------------------------------------------------------------------------
# per-core features


def immune_foci(
    adata, immune_key="atlas_immune", groupby="Sample", eps=30.0, min_samples=5, key_added="in_focus"
) -> pd.DataFrame:
    """Immune foci of each biopsy core: DBSCAN clusters of the immune-cell centroids.

    Adds ``obs[key_added]``: whether a cell is an immune cell inside a focus.

    Parameters
    ----------
    adata
        Cells with centroids in ``obsm["spatial"]`` (um).
    immune_key
        Boolean ``obs`` column that marks the immune cells.
    groupby
        ``obs`` column with the core of each cell.
    eps, min_samples
        DBSCAN neighbourhood radius (um) and the number of cells that make a dense neighbourhood. Cores with
        fewer than `min_samples` immune cells get no foci.

    Returns
    -------
    One row per core: ``n_total``, ``n_immune``, ``immune_fraction``, ``dbscan_frac_in_agg`` (share of the
    immune cells that lie in a focus), ``dbscan_largest_agg_cells`` and ``dbscan_n_aggregates``.
    """
    from sklearn.cluster import DBSCAN

    cells = pd.DataFrame(
        {
            "core": adata.obs[groupby].astype(object).to_numpy(),
            "immune": adata.obs[immune_key].to_numpy(dtype=bool),
            "x": adata.obsm["spatial"][:, 0],
            "y": adata.obsm["spatial"][:, 1],
        }
    )
    rows = []
    in_any_focus = np.zeros(adata.n_obs, dtype=bool)
    for name, core in cells.groupby("core"):
        immune = core[core.immune]
        n_total, n_immune = len(core), len(immune)
        frac_in_foci = largest_cells = np.nan
        n_foci = 0
        if n_immune >= min_samples:
            xy = immune[["x", "y"]].to_numpy(float)
            clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xy)
            in_focus = clusters >= 0
            frac_in_foci = in_focus.mean()
            in_any_focus[immune.index[in_focus]] = True
            if in_focus.any():
                sizes = pd.Series(clusters[in_focus]).value_counts()
                largest_cells = int(sizes.iloc[0])
                n_foci = int(sizes.size)
            else:
                largest_cells = 0
        rows.append(
            {
                groupby: name,
                "n_total": n_total,
                "n_immune": n_immune,
                "immune_fraction": n_immune / n_total if n_total else np.nan,
                "dbscan_frac_in_agg": frac_in_foci,
                "dbscan_largest_agg_cells": largest_cells,
                "dbscan_n_aggregates": n_foci,
            }
        )
    adata.obs[key_added] = in_any_focus
    return pd.DataFrame(rows).set_index(groupby)


def core_features(adata, foci, nk_classes, class_key="ct", groupby="Sample") -> pd.DataFrame:
    """Clinical labels, cell-type fractions and foci of each biopsy core.

    Parameters
    ----------
    adata
        Cells with labels (:func:`atlas_labels`), the immune classes in ``obs[class_key]`` (:func:`cell_classes`)
        and the per-core columns ``biopsy_cellular_grading``, ``biopsy_antibody_grading`` and
        ``biopsy_rejection_type``.
    foci
        Per-core table of :func:`immune_foci`.
    nk_classes
        The classes counted as NK cells.

    Returns
    -------
    One row per core: ``Sample``, ``n_total``, ``n_immune``, the clinical labels, ``immune_fraction``,
    ``frac_Tcell`` and ``frac_Mac`` (Level-3 T cells and macrophages), ``frac_NK``, ``dbscan_n_aggregates``,
    ``dbscan_frac_in_agg`` and ``dbscan_largest_agg_frac`` (largest focus as a share of the immune cells).
    """
    obs = adata.obs
    core = obs[groupby].astype(object)
    level3 = obs["atlas_L3"].astype(object)
    first_cell = obs.assign(**{groupby: core}).drop_duplicates(groupby).set_index(groupby).loc[foci.index]
    return pd.DataFrame(
        {
            groupby: foci.index,
            "n_total": foci.n_total.values,
            "n_immune": foci.n_immune.values,
            "biopsy_cellular_grading": first_cell.biopsy_cellular_grading.values,
            "biopsy_antibody_grading": first_cell.biopsy_antibody_grading.astype(object).values,
            "biopsy_rejection_type": first_cell.biopsy_rejection_type.astype(str).str.strip().values,
            "immune_fraction": foci.immune_fraction.values,
            "frac_Tcell": (level3 == "T cells").groupby(core).mean().loc[foci.index].values,
            "frac_Mac": (level3 == "Macrophages").groupby(core).mean().loc[foci.index].values,
            "frac_NK": obs[class_key].isin(nk_classes).groupby(core).mean().loc[foci.index].values,
            "dbscan_n_aggregates": foci.dbscan_n_aggregates.values,
            "dbscan_frac_in_agg": foci.dbscan_frac_in_agg.values,
            "dbscan_largest_agg_frac": (foci.dbscan_largest_agg_cells / foci.n_immune).values,
        }
    )


# ---------------------------------------------------------------------------------------------------------------
# gene-set scores


def programme_genes(programmes, panel, cytokines) -> pd.DataFrame:
    """Cytokine-response programmes on a gene panel.

    The programme of a cytokine is the union of its response genes over the immune cell types of the programme
    dictionary, restricted to the genes of the panel.

    Parameters
    ----------
    programmes
        Cell type -> cytokine -> genes.
    panel
        Genes of the panel.
    cytokines
        The cytokines to list, in order.

    Returns
    -------
    One row per cytokine: ``cytokine``, ``label``, ``n_union``, ``n_on_panel`` and ``genes`` (joined by ``|``).
    """
    panel = set(panel)
    rows = []
    for cytokine in cytokines:
        union = set()
        for cell_type in programmes:
            union |= set(programmes[cell_type].get(cytokine, []))
        on_panel = sorted(union & panel)
        rows.append(
            {
                "cytokine": cytokine,
                "label": PROGRAMME_LABELS.get(cytokine, cytokine),
                "n_union": len(union),
                "n_on_panel": len(on_panel),
                "genes": "|".join(on_panel),
            }
        )
    return pd.DataFrame(rows)


def gene_sets(table) -> dict[str, list[str]]:
    """Cytokine -> genes of a programme table (:func:`programme_genes`)."""
    return {row.cytokine: str(row.genes).split("|") for row in table.itertuples()}


def score_programmes(adata, sets, prefix="c_", ctrl_size=50, random_state=0) -> list[str]:
    """Score every cell for each gene set with ``scanpy.tl.score_genes``.

    The score is the mean expression of the set minus the mean of `ctrl_size` control genes drawn from the same
    expression bins. Adds one ``obs`` column ``<prefix><name>`` per gene set and returns their names.

    Parameters
    ----------
    adata
        Cells x genes with log1p CP10k in ``X``.
    sets
        Name -> genes; genes that are not in `adata` are left out.
    """
    import scanpy as sc

    keys = []
    for name, genes in sets.items():
        present = [gene for gene in genes if gene in adata.var_names]
        sc.tl.score_genes(adata, present, score_name=f"{prefix}{name}", ctrl_size=ctrl_size, random_state=random_state)
        keys.append(f"{prefix}{name}")
    return keys


def programme_means(adata, sets, groupby=("Sample", "ct"), min_cells=10) -> pd.DataFrame:
    """Mean expression of gene sets per core and cell type.

    Per cell, the score of a set is the mean log1p CP10k over its genes (no control genes, so the scores are
    positive); per group it is the mean over the cells. Cells without a value in one of the `groupby` columns
    are left out.

    Parameters
    ----------
    adata
        Cells x genes with log1p CP10k in ``X``.
    sets
        Name -> genes.
    groupby
        ``obs`` columns that define the groups.
    min_cells
        Smallest group that is kept.

    Returns
    -------
    One row per group in the order of its first cell: one column per gene set, the `groupby` columns and
    ``n_cells``.
    """
    groupby = list(groupby)
    expression = adata.X.tocsc()
    gene_index = {gene: i for i, gene in enumerate(adata.var_names)}
    scores = pd.DataFrame(index=adata.obs_names)
    for name, genes in sets.items():
        columns = [gene_index[gene] for gene in genes if gene in gene_index]
        scores[name] = np.asarray(expression[:, columns].mean(axis=1)).ravel().astype(np.float32)
    for key in groupby:
        scores[key] = adata.obs[key].astype(object).to_numpy()
    scores = scores[scores[groupby[-1]].notna()]
    groups = scores.groupby(groupby, sort=False)
    table = groups[list(sets)].mean()
    table["n_cells"] = groups.size()
    table = table[table.n_cells >= min_cells].reset_index()
    return table[[*sets, *groupby, "n_cells"]]


def group_zscores(adata, keys, groupby, groups) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean of per-cell scores per group, and its z-score across the groups.

    Parameters
    ----------
    adata
        Cells with the scores in ``obs``.
    keys
        Score name -> ``obs`` column.
    groupby
        ``obs`` column with the group of each cell.
    groups
        The groups, in the order of the columns.

    Returns
    -------
    Two scores x groups tables: the means and the z-scores (each score scaled over the groups).
    """
    group_of = adata.obs[groupby].astype(object)
    means = np.zeros((len(keys), len(groups)))
    for i, column in enumerate(keys.values()):
        mean_by_group = adata.obs[column].groupby(group_of).mean()
        for j, group in enumerate(groups):
            means[i, j] = mean_by_group.get(group, np.nan)
    zscores = (means - np.nanmean(means, axis=1, keepdims=True)) / (np.nanstd(means, axis=1, keepdims=True) + 1e-9)
    names = list(keys)
    return pd.DataFrame(means, index=names, columns=groups), pd.DataFrame(zscores, index=names, columns=groups)
