"""CMFLAG clean-up: the spillover flags per 2 um bin and the 8 um bins without their flagged 2 um bins."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

from hhca.grid import SPILLOVER_FLAGS, bin_key

MAX_DIST_UM = 4.6  # 2 um bin centre to its 8 um block centre is <= 3*sqrt(2) = 4.24 um on an exact grid


def bin_flags(flags, adata) -> pd.DataFrame:
    """The four spillover flags per 2 um bin of a section.

    The flag table has one row per decode row, several per 2 um bin; they are collapsed on the 2 um grid key and
    a bin carries a flag when any of its rows does.

    Parameters
    ----------
    flags
        Flag table of the section (``X``, ``Y`` and the four flag columns).
    adata
        2 um bins of the section (``obs["x_um"]``, ``obs["y_um"]``).

    Returns
    -------
    One boolean column per flag, indexed like ``adata.obs``; ``attrs["frac_in_flag_table"]`` holds the share of
    the bins found in the flag table.
    """
    by_key = flags[SPILLOVER_FLAGS].groupby(bin_key(flags["X"].to_numpy(), flags["Y"].to_numpy())).any()
    keys = pd.Index(bin_key(adata.obs["x_um"].to_numpy(), adata.obs["y_um"].to_numpy()))
    if not keys.is_unique:
        raise ValueError("the 2 um bins do not have unique grid keys")
    frac_in_flag_table = float(keys.isin(by_key.index).mean())
    if frac_in_flag_table <= 0.99:
        raise ValueError(f"only {frac_in_flag_table:.3f} of the 2 um bins are in the flag table")
    per_bin = pd.DataFrame(
        by_key.reindex(keys).fillna(False).astype(bool).to_numpy(), index=adata.obs_names, columns=SPILLOVER_FLAGS
    )
    per_bin.attrs["frac_in_flag_table"] = frac_in_flag_table
    return per_bin


def _assign_blocks(obs_2um, obs_8um):
    """8 um bin (row, -1 if none) of every 2 um bin: nearest 8 um centre in physical um."""
    tree = cKDTree(np.c_[obs_8um.x_um.to_numpy(), obs_8um.y_um.to_numpy()])
    dist, block = tree.query(np.c_[obs_2um.x_um.to_numpy(), obs_2um.y_um.to_numpy()], distance_upper_bound=MAX_DIST_UM)
    assigned = np.isfinite(dist)
    block = np.where(assigned, block, -1)
    frac_assigned = float(assigned.mean())
    if frac_assigned <= 0.99:
        raise ValueError(f"only {frac_assigned:.3f} of the 2 um bins have an 8 um bin")
    return block, assigned, frac_assigned


def _block_sum(mask, block, assigned, n_blocks):
    """Sparse matrix that sums the selected 2 um bins into their 8 um bins."""
    idx = np.where(mask & assigned)[0]
    return sp.csr_matrix((np.ones(len(idx), dtype=np.float32), (block[idx], idx)), shape=(n_blocks, len(mask)))


def _full_block_agreement(X2, X8, block, assigned):
    """Fraction of the 8 um bins with all 16 bins whose counts equal the sum of their 2 um counts."""
    n_blocks = X8.shape[0]
    n_all = np.bincount(block[assigned], minlength=n_blocks)
    full = np.where(n_all == 16)[0]
    diff = (X8[full] - (_block_sum(np.ones(X2.shape[0], bool), block, assigned, n_blocks) @ X2)[full]).tocsr()
    diff.eliminate_zeros()
    frac_full_exact = float((np.diff(diff.indptr) == 0).mean())
    if frac_full_exact <= 0.99:
        raise ValueError(f"only {frac_full_exact:.3f} of the full 8 um bins equal the sum of their 2 um bins")
    return frac_full_exact


def clean_bins_8um(adata_2um, adata_8um, flagged):
    """Remove the flagged 2 um bins of a section from its 8 um bins.

    Every 2 um bin belongs to the nearest 8 um bin. The counts of the flagged 2 um bins are subtracted from
    their 8 um bin, and label, composition, certainty and bin count are taken from the unflagged 2 um bins. 8 um
    bins whose 2 um bins are all flagged are dropped.

    Parameters
    ----------
    adata_2um, adata_8um
        The 2 um and 8 um bins of the section before the clean-up (in memory).
    flagged
        Whether each 2 um bin carries one of the spillover flags.

    Returns
    -------
    The clean 8 um bins, with ``obs["n_2um_flagged"]`` and ``obs["frac_counts_removed"]`` added, and a dictionary
    with the statistics of the join (bins assigned, bins dropped, counts removed).
    """
    import anndata as ad

    flagged = np.asarray(flagged)
    if not adata_2um.var_names.equals(adata_8um.var_names):
        raise ValueError("the 2 um and 8 um bins have different features")
    X2, X8 = sp.csr_matrix(adata_2um.X), sp.csr_matrix(adata_8um.X)
    n_blocks = adata_8um.n_obs
    block, assigned, frac_assigned = _assign_blocks(adata_2um.obs, adata_8um.obs)
    frac_full_exact = _full_block_agreement(X2, X8, block, assigned)

    X8c = (X8 - _block_sum(flagged, block, assigned, n_blocks) @ X2).tocsr()
    n_negative = int((X8c.data < 0).sum())
    negative_counts = float(-X8c.data[X8c.data < 0].sum())
    X8c.data[X8c.data < 0] = 0
    X8c.eliminate_zeros()
    total_before = np.asarray(X8.sum(1)).ravel()
    total_after = np.asarray(X8c.sum(1)).ravel()

    order = list(adata_8um.uns["L3_composition_order"])
    code = pd.Categorical(adata_2um.obs.L3_label.astype(str), categories=order).codes
    keep = ~flagged & assigned & (code >= 0)
    n_per_type = np.bincount(block[keep] * len(order) + code[keep], minlength=n_blocks * len(order)).reshape(
        n_blocks, len(order)
    )
    n_clean = n_per_type.sum(1)
    n_flagged = np.bincount(block[flagged & assigned], minlength=n_blocks)
    with np.errstate(invalid="ignore", divide="ignore"):
        composition = np.where(n_clean[:, None] > 0, n_per_type / n_clean[:, None], 0.0).astype(np.float32)
        gcert_sum = np.bincount(
            block[keep], weights=adata_2um.obs.gcert.to_numpy()[keep].astype(float), minlength=n_blocks
        )
        gcert_mean = (gcert_sum / n_clean).astype(np.float32)
    top = composition.argmax(1)
    label = pd.Categorical(np.where(n_clean > 0, np.array(order, dtype=object)[top], None), categories=order)
    keep_block = ~((n_clean == 0) & (n_flagged > 0))
    obs = pd.DataFrame(
        {
            "L3_label": label,
            "L3_top1_frac": np.where(n_clean > 0, composition.max(1), np.nan).astype(np.float32),
            "gcert_mean": gcert_mean,
            "n_2um_bins": n_clean.astype(np.int32),
            "n_2um_flagged": n_flagged.astype(np.int32),
            "frac_counts_removed": np.where(
                total_before > 0, 1 - total_after / np.maximum(total_before, 1e-9), 0
            ).astype(np.float32),
            "x_um": adata_8um.obs.x_um.to_numpy(),
            "y_um": adata_8um.obs.y_um.to_numpy(),
            "slide": adata_8um.obs.slide.values,
        },
        index=adata_8um.obs_names,
    )
    clean = ad.AnnData(X=X8c[keep_block], obs=obs[keep_block], var=adata_8um.var.copy())
    clean.obsm["L3_composition"] = composition[keep_block]
    clean.obsm["spatial"] = np.asarray(adata_8um.obsm["spatial"])[keep_block]
    clean.uns["L3_composition_order"] = order
    clean.uns["slide"] = adata_8um.uns.get("slide", str(adata_8um.obs.slide.iloc[0]))
    clean.uns["build"] = adata_8um.uns["build"]
    stats = {
        "n_8um": int(n_blocks),
        "n_8um_dropped_all_flagged": int((~keep_block).sum()),
        "n_8um_clean": int(keep_block.sum()),
        "pct_8um_counts_removed": round(100 * float(1 - total_after.sum() / total_before.sum()), 3),
        "frac_2um_assigned_to_8um": round(frac_assigned, 5),
        "frac_full_blocks_exact_sum": round(frac_full_exact, 5),
        "n_8um_negative_entries_clipped": n_negative,
        "pct_8um_counts_clipped": round(100 * negative_counts / float(total_before.sum()), 4),
    }
    return clean, stats
