"""drug2cell target scores of pseudobulk profiles and their comparison between two groups."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def _dense(matrix) -> np.ndarray:
    """A sparse or dense matrix as ndarray."""
    return np.asarray(matrix.todense()) if hasattr(matrix, "todense") else np.asarray(matrix)


def _chembl_targets(exclude=()) -> dict:
    """The ChEMBL dictionary of drug2cell (category -> drug -> target genes) without the genes in `exclude`."""
    import drug2cell.data

    drop = set(exclude)
    return {
        category: {drug: [gene for gene in genes if gene not in drop] for drug, genes in drugs.items()}
        for category, drugs in drug2cell.data.chembl().items()
    }


def score_drug_targets(adata, exclude=()) -> None:
    """Score the targets of every ChEMBL drug with drug2cell.

    The score of a drug is the mean expression of its target genes (drug2cell ``score``, method ``mean``, on
    ``adata.X``). Adds ``adata.uns["drug2cell"]``, an AnnData of observations x drugs.

    Parameters
    ----------
    adata
        Log-normalised profiles.
    exclude
        Genes removed from every target set before scoring.
    """
    import drug2cell

    drug2cell.score(adata, targets=_chembl_targets(exclude), nested=True)


class _Scores:
    """Expression and drug2cell scores of the profiles, by ChEMBL id."""

    def __init__(self, adata):
        self.column = {gene: i for i, gene in enumerate(adata.var_names)}
        self.expression = _dense(adata.X)
        self.drugs = adata.uns["drug2cell"]
        self.feature = {feature.split("|")[0]: feature for feature in self.drugs.var_names}
        self.scores = _dense(self.drugs.X)
        self.score_column = {feature: i for i, feature in enumerate(self.drugs.var_names)}

    def targets(self, chembl_id) -> list[str]:
        """Target genes of a drug as scored."""
        genes = str(self.drugs.var.loc[self.feature[chembl_id], "genes"]).split(",")
        return [gene.strip() for gene in genes if gene.strip()]

    def measured(self, chembl_id) -> list[str]:
        """Target genes of a drug among the genes of the profiles."""
        return [gene for gene in self.targets(chembl_id) if gene in self.column]

    def score(self, chembl_id):
        """Score of a drug per profile, or None if the drug or all of its targets are missing."""
        if chembl_id not in self.feature or not self.measured(chembl_id):
            return None
        return self.scores[:, self.score_column[self.feature[chembl_id]]]


def _per_drug(adata, chembl_ids) -> dict:
    """Score per profile of every drug of `chembl_ids` that has a measured target."""
    scores = _Scores(adata)
    per_drug = {chembl_id: scores.score(chembl_id) for chembl_id in chembl_ids}
    return {chembl_id: score for chembl_id, score in per_drug.items() if score is not None}


def drug_scores(adata, chembl_ids) -> pd.DataFrame:
    """drug2cell scores of a list of drugs.

    Parameters
    ----------
    adata
        Profiles scored with :func:`hhca.tl.score_drug_targets`.
    chembl_ids
        ChEMBL ids of the drugs; drugs without a measured target are left out.

    Returns
    -------
    Profiles x drugs.
    """
    return pd.DataFrame(_per_drug(adata, chembl_ids), index=adata.obs_names)


def mean_drug_score(adata, chembl_ids) -> np.ndarray:
    """Mean drug2cell score over a list of drugs, one value per profile (drugs without a measured target left out)."""
    return np.mean(np.vstack(list(_per_drug(adata, chembl_ids).values())), axis=0)


def target_mean_difference(adata, chembl_ids) -> float:
    """Largest difference between the drug2cell score and the plain mean over the measured targets of a drug."""
    scores = _Scores(adata)
    largest = 0.0
    for chembl_id in chembl_ids:
        score = scores.score(chembl_id)
        if score is not None:
            columns = [scores.column[gene] for gene in scores.measured(chembl_id)]
            target_mean = scores.expression[:, columns].mean(1)
            largest = max(largest, float(np.max(np.abs(target_mean.astype(np.float64) - score.astype(np.float64)))))
    return largest


def _drug_row(scores, chembl_id, name, drug_class, first, second, exclude, detected_frac) -> dict:
    """Score difference between two groups of profiles for one drug."""
    if chembl_id not in scores.feature:
        return {
            "drug": name,
            "cls": drug_class,
            "in_d2c": False,
            "n_targets": 0,
            "n_detected": 0,
            "diff": np.nan,
            "p": np.nan,
            "detected_genes": "",
        }
    expression, column = scores.expression, scores.column
    measured = scores.measured(chembl_id)
    detected = [gene for gene in measured if (expression[:, column[gene]] > 0).mean() >= detected_frac]
    if measured:
        score = scores.score(chembl_id)
        diff = float(score[first].mean() - score[second].mean())
        try:
            _, p = stats.mannwhitneyu(score[first], score[second], alternative="two-sided")
        except ValueError:
            p = np.nan
    else:
        diff, p = np.nan, np.nan
    # the same comparison on the plain mean over the targets that are not in `exclude`
    kept = [gene for gene in measured if gene not in set(exclude)]
    if kept and len(kept) < len(measured):
        score_kept = expression[:, [column[gene] for gene in kept]].mean(1)
        diff_kept = float(score_kept[first].mean() - score_kept[second].mean())
        p_kept = stats.mannwhitneyu(score_kept[first], score_kept[second], alternative="two-sided")[1]
    else:
        diff_kept, p_kept = diff, p
    return {
        "drug": name,
        "cls": drug_class,
        "in_d2c": True,
        "n_targets": len(measured),
        "n_detected": len(detected),
        "diff": diff,
        "p": p,
        "detected_genes": ",".join(detected),
        "myelin_targets": ",".join(gene for gene in measured if gene in set(exclude)),
        "diff_excl_myelin": diff_kept,
        "p_excl_myelin": p_kept,
    }


def _benjamini_hochberg(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p values of the tests that have one."""
    adjusted = pd.Series(np.nan, index=p.index)
    tested = p.notna()
    if tested.any():
        adjusted[tested] = multipletests(p[tested], method="fdr_bh")[1]
    return adjusted


def drug_differences(
    adata, drugs, groupby="group", groups=("myelinated", "non_myelinated"), exclude=(), detected_frac=0.10
) -> pd.DataFrame:
    """Compare the drug2cell score of every drug between two groups of profiles.

    Parameters
    ----------
    adata
        Profiles scored with :func:`hhca.tl.score_drug_targets`.
    drugs
        ``(ChEMBL id, name, class)`` per drug.
    groupby, groups
        Column of ``adata.obs`` and its two values; the difference is first minus second.
    exclude
        Genes that were removed from the target sets.
    detected_frac
        A target gene is detected when at least this share of the profiles has a count.

    Returns
    -------
    One row per drug: ``drug``, ``cls``, ``in_d2c``, the number of measured (``n_targets``) and detected targets
    (``n_detected``, ``detected_genes``), ``diff`` (difference of the mean scores), ``p`` (two-sided
    Mann-Whitney) and ``fdr`` (Benjamini-Hochberg over the drugs), followed by the same on the plain mean over the
    targets outside `exclude` (``*_excl_myelin``).
    """
    scores = _Scores(adata)
    first = (adata.obs[groupby] == groups[0]).to_numpy()
    second = (adata.obs[groupby] == groups[1]).to_numpy()
    table = pd.DataFrame(
        [
            _drug_row(scores, chembl_id, name, drug_class, first, second, exclude, detected_frac)
            for chembl_id, name, drug_class in drugs
        ]
    )
    table["fdr"] = _benjamini_hochberg(table.p)
    table["fdr_excl_myelin"] = _benjamini_hochberg(table.p_excl_myelin if "p_excl_myelin" in table else table.p)
    extra = ["myelin_targets", "diff_excl_myelin", "p_excl_myelin", "fdr_excl_myelin"]
    return table[[c for c in table.columns if c not in extra] + [c for c in extra if c in table.columns]]


def drug_target_coverage(adata, drugs, exclude=(), check_targets=()) -> dict:
    """Which target genes of a list of drugs are among the genes of the profiles.

    Returns
    -------
    ``n_genes``, ``check_targets`` (gene -> present), and per drug name the targets that are present
    (``drug_targets_present``) and missing (``drug_targets_missing``); genes in `exclude` are left out.
    """
    chembl = {}
    for category in _chembl_targets().values():
        chembl.update(category)
    targets_of_id = {feature.split("|")[0]: genes for feature, genes in chembl.items()}
    present = set(adata.var_names)
    targets = {
        name: [gene for gene in targets_of_id.get(chembl_id, []) if gene not in set(exclude)]
        for chembl_id, name, _ in drugs
    }
    return {
        "n_genes": int(adata.n_vars),
        "check_targets": {gene: gene in present for gene in check_targets},
        "drug_targets_present": {name: [g for g in genes if g in present] for name, genes in targets.items()},
        "drug_targets_missing": {name: [g for g in genes if g not in present] for name, genes in targets.items()},
    }
