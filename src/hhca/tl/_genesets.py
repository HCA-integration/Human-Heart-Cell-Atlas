"""Gene sets of the per-donor scores: drug targets, cardiac axes, hormone producers and hormone receivers."""

from __future__ import annotations

from collections import ChainMap

import pandas as pd

from hhca.tl._hormones import EXCLUDE_COLUMNS, INCLUDE_COLUMNS, RECEPTOR_COLUMNS, established_receptors


def _gene_symbols(values):
    """Non-empty gene symbols of a definition row."""
    return [str(g) for g in values if isinstance(g, str) and g and g != "nan"]


def _drug_and_axis_sets(gene_index, chembl_nested, axes):
    """DRUG:: and AXIS:: gene sets (targets present in the atlas, first-seen order) and their metadata."""
    groups, gmeta = {}, {}
    chembl = dict(ChainMap(*[chembl_nested[category] for category in chembl_nested]))
    drug_atc = {}
    for atc, drugs in chembl_nested.items():
        for drug in drugs:
            drug_atc.setdefault(drug, atc)
    sources = {}
    sources.update({name: list(targets) for name, targets in chembl.items()})
    sources.update({name: list(axis["genes"]) for name, axis in axes.items()})
    cardiac_genes = sorted({g for axis in axes.values() for g in axis["genes"] if g in gene_index})
    for label, targets in sources.items():
        present = [g for g in dict.fromkeys(targets) if g in gene_index]
        if not present:
            continue
        is_axis = label.startswith("AXIS:")
        name = ("AXIS::" if is_axis else "DRUG::") + label
        groups[name] = present
        gmeta[name] = {
            "kind": "cardiac_axis" if is_axis else "drug",
            "label": label,
            "atc": drug_atc.get(label, "AXIS" if is_axis else "NA"),
            "is_axis": is_axis,
            "n_cardiac_targets": len(set(sources[label]) & set(cardiac_genes)),
        }
    return groups, gmeta


def _producer_sets(producing, union_producers):
    """HPROD:: gene sets: include minus exclude genes of a hormone's last definition row (or of all its rows)."""
    groups, gmeta = {}, {}
    genes_of, display_of = {}, {}
    for _, row in producing.iterrows():
        hormone = str(row["hormone_short"])
        include = _gene_symbols(row[INCLUDE_COLUMNS].values)
        exclude = _gene_symbols(row[EXCLUDE_COLUMNS].values)
        if union_producers and hormone in genes_of:
            genes_of[hormone] = (genes_of[hormone][0] | set(include), genes_of[hormone][1] | set(exclude))
        else:
            genes_of[hormone] = (set(include), set(exclude))
        display_of[hormone] = (str(row["hormone_display"]), str(row["Tier"]))
    for hormone, (include, exclude) in genes_of.items():
        genes = sorted(include - exclude)
        if not genes:
            continue
        groups[f"HPROD::{hormone}"] = genes
        gmeta[f"HPROD::{hormone}"] = {
            "kind": "hormone_producer",
            "label": hormone,
            "hormone_display": display_of[hormone][0],
            "tier": display_of[hormone][1],
        }
    return groups, gmeta


def _receiver_sets(receptors, panel, producer_table):
    """HRECV:: gene sets: established receptors on the gene panel."""
    groups, gmeta = {}, {}
    produced = producer_table.drop_duplicates("Hormone").set_index("Hormone")
    for hormone, rows in established_receptors(receptors).groupby("hormone_short"):
        genes = pd.unique(rows[RECEPTOR_COLUMNS].values.ravel())
        genes = [g for g in genes if isinstance(g, str) and g in panel]
        if not genes:
            continue
        groups[f"HRECV::{hormone}"] = sorted(genes)
        if hormone in produced.index:
            display, tier = str(produced.loc[hormone, "hormone_display"]), str(produced.loc[hormone, "Tier"])
        else:
            display, tier = hormone, ""
        gmeta[f"HRECV::{hormone}"] = {
            "kind": "hormone_receiver",
            "label": hormone,
            "hormone_display": display,
            "tier": tier,
        }
    return groups, gmeta


def gene_set_collection(
    atlas_genes, chembl_nested, axes, producing, receptors, panel, producer_table, union_producers=False
) -> dict:
    """Assemble the gene sets of the per-donor scores from their four sources.

    ``DRUG::`` the ChEMBL drug-target dictionary of drug2cell (ATC-nested), drugs with at least one target in the
    atlas; ``AXIS::`` the curated cardiac axes; ``HPROD::`` the include minus exclude genes of the last producing
    definition row of a hormone (with `union_producers` the union over its rows); ``HRECV::`` the established
    receptors on the gene panel.

    Parameters
    ----------
    atlas_genes
        Gene names of the atlas.
    chembl_nested
        ATC category -> drug -> targets.
    axes
        Axis name -> ``{"genes": [...]}``.
    producing, receptors
        Definitions of :func:`hhca.tl.hormone_definitions`.
    panel
        Set of the genes of the Level-4 atlas profile.
    producer_table
        Producer strength table of the Level-4 profile (display name and tier of the produced hormones).

    Returns
    -------
    ``{"groups": name -> genes, "gmeta": name -> description}``.
    """
    gene_index = {g: i for i, g in enumerate(atlas_genes)}
    groups, gmeta = {}, {}
    for family_groups, family_meta in (
        _drug_and_axis_sets(gene_index, chembl_nested, axes),
        _producer_sets(producing, union_producers),
        _receiver_sets(receptors, panel, producer_table),
    ):
        groups.update(family_groups)
        gmeta.update(family_meta)
    return {"groups": groups, "gmeta": gmeta}
