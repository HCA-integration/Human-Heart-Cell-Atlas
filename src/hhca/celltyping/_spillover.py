"""Spillover flags of CMFLAG on the decode rows of a Visium HD section.

Transcripts of cardiomyocytes, smooth muscle and arterial / venous endothelium diffuse out of their cells into
empty space. A decode row is flagged when it is assigned to one of these types, lies on H&E whitespace, the
marker density expected from spillover at its depth into the whitespace explains what is observed there, and it
carries neither a nucleus nor a marker of a competing type.

nucleus gate   nucleus_gate      rows whose haematoxylin optical density shows a nucleus are never flagged
marker depth   marker_depth      marker counts of the 2 um bin of each row and its distance into whitespace
spillover      spillover_flags   the four flags
lumen          vessel_lumen      whitespace inside the filled outline of a large vessel (a column of the flag table)
flag table     flag_table        the columns of <section>_clean_bins.parquet
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    distance_transform_edt,
    generate_binary_structure,
    uniform_filter,
)
from scipy.spatial import cKDTree

from hhca.grid import SPILLOVER_FLAGS

log = logging.getLogger("hhca.celltyping").info

GRID = 2.0
LARGE_TYPES = {"Arterial(muscular)", "Arteriole", "Venous", "Venous(muscular)"}
PAD = 16.0
CLOSE_ITER = 4


def flag_thresholds(tau, factor_order, unscaled, scale=0.3) -> np.ndarray:
    """Certainty threshold per type for the assignment behind the flags.

    The thresholds of all types other than `unscaled` (the cardiomyocytes) are multiplied by `scale`, which keeps
    more vessel and immune rows assigned.

    Parameters
    ----------
    tau
        Type -> threshold (``pertype_tau_recommended.json``).
    factor_order
        Level-3 types in factor order.
    """
    scaled = np.array([tau[name] for name in factor_order], float)
    scaled[[i for i, name in enumerate(factor_order) if name not in unscaled]] *= scale
    return scaled


def nucleus_gate(hema, covered, whitespace, fraction=0.35):
    """Decode rows with a nucleus, from their own haematoxylin optical density.

    A row holds a nucleus when its haematoxylin OD lies more than `fraction` of the way from the median of the
    whitespace rows to the median of the tissue rows. The two medians are taken per section, so the gate follows
    the staining intensity. No smoothing, which would carry tissue signal into gaps.

    Returns
    -------
    Boolean array per row, and the values of the gate.
    """
    hema, covered, whitespace = np.asarray(hema), np.asarray(covered), np.asarray(whitespace)
    floor = float(np.median(hema[covered & whitespace])) if (covered & whitespace).any() else 0.0
    tissue = float(np.median(hema[covered & (~whitespace)])) if (covered & (~whitespace)).any() else 0.0
    threshold = floor + fraction * max(tissue - floor, 1e-4)
    resident = covered & (hema > threshold)
    log(
        f"H&E white_frac={whitespace.mean():.3f} cov={covered.mean():.3f} | nucleus hfloor={floor:.4f} "
        f"htiss={tissue:.4f} thr={threshold:.4f} resident_frac={resident.mean():.3f}"
    )
    gate = {
        "nucleus_gate": "hematoxylin_HE_perbin",
        "H_NUC_FRAC": fraction,
        "nucleus_hfloor": floor,
        "nucleus_htiss": tissue,
        "nucleus_thr": threshold,
        "resident_frac": float(resident.mean()),
    }
    return resident, gate


def _gene_counts(bins, genes):
    """Summed counts of the genes present in `bins`, per bin."""
    present = [gene for gene in genes if gene in bins.var_names]
    if not present:
        return np.zeros(bins.n_obs, np.float32)
    return np.asarray(bins[:, present].X.sum(1)).ravel().astype(np.float32)


def marker_depth(bins, X, Y, whitespace, markers) -> pd.DataFrame:
    """Marker counts and depth into whitespace for every decode row.

    The Space Ranger bins and the decode rows are placed on one 2 um raster. A raster cell counts as tissue when
    at least half of its decode rows are not whitespace; the depth of a row is the distance of its cell to the
    nearest tissue cell.

    Parameters
    ----------
    bins
        Space Ranger 2 um bins of the section (:func:`hhca.celltyping.read_bins`).
    X, Y
        Coordinates of the decode rows in um.
    whitespace
        Whether each decode row lies on H&E whitespace.
    markers
        Name -> marker genes; one column ``<name>_bin`` is returned per entry.

    Returns
    -------
    One row per decode row: ``depth_bin`` (um) and the summed marker counts of its 2 um bin.
    """
    whitespace = np.asarray(whitespace)
    counts = {name: _gene_counts(bins, genes) for name, genes in markers.items()}
    umx, umy = bins.obsm["spatial"][:, 0], bins.obsm["spatial"][:, 1]
    x0 = min(X.min(), umx.min()) - 4
    y0 = min(Y.min(), umy.min()) - 4
    x1 = max(X.max(), umx.max()) + 4
    y1 = max(Y.max(), umy.max()) + 4
    Wg = int(np.ceil((x1 - x0) / GRID))
    Hg = int(np.ceil((y1 - y0) / GRID))
    ri = np.clip(((umy - y0) / GRID).astype(int), 0, Hg - 1)
    ci = np.clip(((umx - x0) / GRID).astype(int), 0, Wg - 1)
    occ = np.zeros((Hg, Wg), np.float32)
    np.add.at(occ, (ri, ci), 1.0)

    def rmean(vals):
        s = np.zeros((Hg, Wg), np.float32)
        np.add.at(s, (ri, ci), vals)
        return np.where(occ > 0, s / np.maximum(occ, 1), 0.0).astype(np.float32)

    G = {name: rmean(vals) for name, vals in counts.items()}
    bj = np.clip(((Y - y0) / GRID).astype(int), 0, Hg - 1)
    bi = np.clip(((X - x0) / GRID).astype(int), 0, Wg - 1)
    # distance into whitespace from the tissue edge
    wsum = np.zeros((Hg, Wg), np.float32)
    tsum = np.zeros((Hg, Wg), np.float32)
    np.add.at(wsum, (bj, bi), whitespace.astype(np.float32))
    np.add.at(tsum, (bj, bi), (~whitespace).astype(np.float32))
    he_tissue = (tsum > 0) & (tsum >= wsum)
    depth_grid = (distance_transform_edt(~he_tissue) * GRID).astype(np.float32)
    out = {"depth_bin": depth_grid[bj, bi]}
    for k, grid in G.items():
        out[k + "_bin"] = grid[bj, bi]
    return pd.DataFrame(out)


def decay_envelope(depth, fit):
    """Expected spillover marker density at `depth` (um) into whitespace: ``A exp(-depth / lam) + b``."""
    return fit["A"] * np.exp(-depth / fit["lam"]) + fit["b"]


def _spillover_flag(is_type, observed, depth, whitespace, guard, fit, min_share):
    """Rows of a type in whitespace whose marker density is explained by spillover."""
    gap = depth > 0
    expected = decay_envelope(depth, fit)
    share = np.where(gap, np.clip(expected / np.maximum(observed, 1e-3), 0, 1), 0.0).astype(np.float32)
    share[gap & (observed <= 0)] = 1.0
    return is_type & (share >= min_share) & whitespace & (~guard)


def spillover_flags(assign, factor_order, marker, whitespace, nucleus, rules, fits, min_share=0.80) -> pd.DataFrame:
    """The four spillover flags per decode row.

    For each flag, `rules` names the cell types it applies to, the marker set whose density is compared with the
    decay of that marker into whitespace, and the marker sets of competing types that guard a row. A row is
    flagged when it is assigned to one of the types, lies on whitespace, the expected spillover makes up at least
    `min_share` of the observed marker density, and neither a guard marker nor a nucleus is present.

    Parameters
    ----------
    assign
        Assigned factor index per row (-1 = unassigned).
    factor_order
        Level-3 types in factor order.
    marker
        Marker counts and depth per row (:func:`marker_depth`).
    whitespace, nucleus
        Whether each row lies on H&E whitespace, and whether it holds a nucleus (:func:`nucleus_gate`).
    rules
        Flag -> ``{"types": [...], "markers": name, "guards": [names], "decay": name}``.
    fits
        Decay name -> ``{"A": ..., "lam": ..., "b": ...}``.

    Returns
    -------
    One boolean column per flag.
    """
    assign, whitespace, nucleus = np.asarray(assign), np.asarray(whitespace), np.asarray(nucleus)
    depth = marker["depth_bin"].to_numpy()
    flags = {}
    for flag, rule in rules.items():
        is_type = np.isin(assign, [factor_order.index(name) for name in rule["types"]])
        guard = nucleus.copy()
        for name in rule["guards"]:
            guard |= marker[f"{name}_bin"].to_numpy() > 0
        observed = marker[f"{rule['markers']}_bin"].to_numpy()
        flags[flag] = _spillover_flag(is_type, observed, depth, whitespace, guard, fits[rule["decay"]], min_share)
    return pd.DataFrame(flags)


def _concave_lumen(X, Y, whitespace, structs, cells):
    """Whitespace rows inside the closed and filled outline of each large vessel."""
    large_ids = set(structs.loc[structs["vessel_type"].isin(LARGE_TYPES), "struct"].tolist())
    cg = cells[cells["struct"].isin(large_ids)]
    in_lumen = np.zeros(len(X), bool)
    st = generate_binary_structure(2, 2)
    tree = cKDTree(np.column_stack([X, Y]))
    n_ok = 0
    n_deg = 0
    for _, sub in cg.groupby("struct"):
        pts = sub[["x_um", "y_um"]].to_numpy(float)
        if len(pts) < 4:
            n_deg += 1
            continue
        wx0, wx1 = pts[:, 0].min() - PAD, pts[:, 0].max() + PAD
        wy0, wy1 = pts[:, 1].min() - PAD, pts[:, 1].max() + PAD
        cxc, cyc = 0.5 * (wx0 + wx1), 0.5 * (wy0 + wy1)
        rad = 0.5 * np.hypot(wx1 - wx0, wy1 - wy0) + 2
        cand = np.array(tree.query_ball_point([cxc, cyc], rad), dtype=np.int64)
        if len(cand) == 0:
            n_deg += 1
            continue
        cx = X[cand]
        cy = Y[cand]
        inw = (cx >= wx0) & (cx < wx1) & (cy >= wy0) & (cy < wy1)
        cand = cand[inw]
        cx = cx[inw]
        cy = cy[inw]
        if len(cand) < 10:
            n_deg += 1
            continue
        W = int(np.ceil((wx1 - wx0) / GRID)) + 1
        H = int(np.ceil((wy1 - wy0) / GRID)) + 1
        mr = np.clip(((pts[:, 1] - wy0) / GRID).astype(int), 0, H - 1)
        mc = np.clip(((pts[:, 0] - wx0) / GRID).astype(int), 0, W - 1)
        okp = (pts[:, 0] >= wx0) & (pts[:, 0] < wx1) & (pts[:, 1] >= wy0) & (pts[:, 1] < wy1)
        mem = np.zeros((H, W), bool)
        mem[mr[okp], mc[okp]] = True
        fpr = binary_fill_holes(binary_closing(mem, structure=st, iterations=CLOSE_ITER))
        ri = np.clip(((cy - wy0) / GRID).astype(int), 0, H - 1)
        ci = np.clip(((cx - wx0) / GRID).astype(int), 0, W - 1)
        sel = fpr[ri, ci] & whitespace[cand]
        if sel.any():
            in_lumen[cand[sel]] = True
        n_ok += 1
    log(f"lumen: {n_ok} vessels ({n_deg} too small); lumen bins={int(in_lumen.sum()):,}")
    return in_lumen


def vessel_lumen(X, Y, whitespace, structs_path, cells_path) -> np.ndarray:
    """Lumen rows of the large vessels of the vessel tables; none if the section has no tables."""
    if not (structs_path and cells_path and os.path.exists(structs_path) and os.path.exists(cells_path)):
        log("no vessel structures: lumen not computed")
        return np.zeros(len(X), bool)
    structs = pd.read_csv(structs_path)
    cells = pd.read_parquet(cells_path, columns=["struct", "x_um", "y_um"])
    return _concave_lumen(X, Y, np.asarray(whitespace), structs, cells)


def local_density(X, Y, vals, win_um) -> np.ndarray:
    """Mean of `vals` over a square neighbourhood of win_um around each row (2 um raster, uniform filter)."""
    x0, y0 = X.min(), Y.min()
    Wg = int(np.ceil((X.max() - x0) / GRID)) + 1
    Hg = int(np.ceil((Y.max() - y0) / GRID)) + 1
    col = np.clip(((X - x0) / GRID).astype(int), 0, Wg - 1)
    row = np.clip(((Y - y0) / GRID).astype(int), 0, Hg - 1)
    sg = np.zeros((Hg, Wg), np.float32)
    og = np.zeros((Hg, Wg), np.float32)
    np.add.at(sg, (row, col), vals.astype(np.float32))
    np.add.at(og, (row, col), 1.0)
    k = int(round(win_um / GRID)) * 2 + 1
    ss = uniform_filter(sg, k, mode="constant")
    oo = uniform_filter(og, k, mode="constant")
    return np.where(oo > 0, ss / np.maximum(oo, 1e-6), 0).astype(np.float32)[row, col]


def flag_table(X, Y, assign, flags, lumen, resident, grey, eosin, hema, covered) -> pd.DataFrame:
    """Table of ``<section>_clean_bins.parquet``: one row per decode row with its flags and H&E values."""
    flag = {name: np.asarray(flags[name]) for name in SPILLOVER_FLAGS}
    exclude = (
        flag["flag_cm_spillover"]
        | flag["flag_vsmc_spillover"]
        | flag["flag_spill_artEC"]
        | flag["flag_spill_venEC"]
        | lumen
    )
    keep = ~exclude
    log(
        f"flagged {int(exclude.sum()):,}/{len(X):,} = {100 * exclude.mean():.3f}%  "
        f"(CM {int(flag['flag_cm_spillover'].sum()):,} VSMC {int(flag['flag_vsmc_spillover'].sum()):,} "
        f"artEC {int(flag['flag_spill_artEC'].sum()):,} venEC {int(flag['flag_spill_venEC'].sum()):,} "
        f"lumen {int(lumen.sum()):,})"
    )
    return pd.DataFrame(
        {
            "X": X.astype(np.float32),
            "Y": Y.astype(np.float32),
            "assign": assign.astype(np.int16),
            "exclude_spillover": exclude,
            "keep_layer1": keep,
            "flag_cm_spillover": flag["flag_cm_spillover"],
            "flag_vsmc_spillover": flag["flag_vsmc_spillover"],
            "flag_spill_artEC": flag["flag_spill_artEC"],
            "flag_spill_venEC": flag["flag_spill_venEC"],
            "in_large_vessel_lumen": lumen,
            "resident_nucleus": resident,
            "grey_eff": np.round(grey, 1),
            "eosin_od": np.round(eosin, 4),
            "hema_od": np.round(hema, 4),
            "micro_covered": covered,
        }
    )


def percent(count, total) -> float:
    """Percentage with four decimals."""
    return round(100.0 * int(count) / total, 4)


def flag_counts(flags, lumen) -> dict:
    """Number and percentage of rows per flag, of lumen rows and of all excluded rows (``<section>_counts.json``)."""
    n = len(lumen)
    flag_cm, flag_vs = np.asarray(flags["flag_cm_spillover"]), np.asarray(flags["flag_vsmc_spillover"])
    flag_art, flag_ven = np.asarray(flags["flag_spill_artEC"]), np.asarray(flags["flag_spill_venEC"])
    exclude = flag_cm | flag_vs | flag_art | flag_ven | lumen
    return {
        "n_cm": int(flag_cm.sum()),
        "pct_cm": percent(flag_cm.sum(), n),
        "n_vsmc": int(flag_vs.sum()),
        "pct_vsmc": percent(flag_vs.sum(), n),
        "n_artEC": int(flag_art.sum()),
        "pct_artEC": percent(flag_art.sum(), n),
        "n_venEC": int(flag_ven.sum()),
        "pct_venEC": percent(flag_ven.sum(), n),
        "n_lumen": int(lumen.sum()),
        "pct_lumen": percent(lumen.sum(), n),
        "n_excluded": int(exclude.sum()),
    }
